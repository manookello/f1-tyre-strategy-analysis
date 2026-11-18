"""Load race lap timing with FastF1 and combine all races into tidy parquet tables.

Data sources
------------
* **FastF1** (this module) reads the official F1 live-timing archive. It is used for 2022. When
  this project was built the archive had started returning HTTP 401 to anonymous users, so newer
  seasons need ``F1_USERNAME`` / ``F1_PASSWORD`` (an F1 account) to load this way.
* **OpenF1** (:mod:`f1tyres.openf1`) serves the same timing data for every session since 2023,
  free and unauthenticated. It is used for 2023-2025 and converted to the same schema.

Each race is written to ``data/raw/races/{season}_{round}_laps.parquet`` (+ ``_results``), and
:func:`combine` merges every race file into ``data/processed/``.

The FastF1 bulk download is *paced*: FastF1 enforces a self-imposed ceiling of 500 API calls per
hour, and one race needs roughly a dozen calls. Each race is loaded in its own short-lived child
process (so a stalled connection can be timed out), race starts are spaced at least
``min_seconds_per_race`` apart, and on a rate-limit error we back off before retrying.

Usage::

    python -m f1tyres.data --seasons 2022         # FastF1 races, then combine
    python -m f1tyres.openf1 --seasons 2023 2024 2025
    python -m f1tyres.data --combine-only         # rebuild data/processed from all race files
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from f1tyres.config import CACHE, DATA_PROCESSED, DATA_RAW, FASTF1_SEASONS

try:
    from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError
except ImportError:  # older FastF1
    from fastf1.core import DataNotLoadedError
    from fastf1.req import RateLimitExceededError

RATE_LIMIT_EXIT_CODE = 3

LAP_COLUMNS = [
    "Driver",
    "DriverNumber",
    "Team",
    "LapNumber",
    "LapTime",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    "IsAccurate",
    "Position",
    "Deleted",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedST",
    "LapStartTime",
]
WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]
RESULT_COLUMNS = [
    "Abbreviation",
    "FullName",
    "TeamName",
    "Position",
    "ClassifiedPosition",
    "GridPosition",
    "Status",
    "Points",
]


def enable_cache(path: Path = CACHE) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(fastf1.Cache, "configure"):  # FastF1 >= 3.9
        fastf1.Cache.configure(cache_dir=str(path))
    else:
        fastf1.Cache.enable_cache(str(path))
    fastf1.set_log_level(logging.ERROR)


def race_schedule(year: int) -> pd.DataFrame:
    sched = fastf1.get_event_schedule(year, include_testing=False)
    return sched[sched["RoundNumber"] > 0][
        ["RoundNumber", "EventName", "Location", "Country", "EventDate", "EventFormat"]
    ].reset_index(drop=True)


def _seconds(series: pd.Series) -> pd.Series:
    return series.dt.total_seconds()


def load_race(year: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (laps, results) for one Grand Prix with weather joined onto each lap."""
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = session.laps
    weather = laps.get_weather_data().reset_index(drop=True)
    df = laps[LAP_COLUMNS].reset_index(drop=True).copy()
    for col in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "LapStartTime"):
        df[col] = _seconds(df[col])
    df["PitIn"] = df.pop("PitInTime").notna()
    df["PitOut"] = df.pop("PitOutTime").notna()
    df = pd.concat([df, weather[WEATHER_COLUMNS]], axis=1)
    df.insert(0, "Season", year)
    df.insert(1, "Round", rnd)
    df.insert(2, "EventName", session.event["EventName"])
    df["TotalLaps"] = int(df["LapNumber"].max())

    results = session.results[RESULT_COLUMNS].reset_index(drop=True).copy()
    results.insert(0, "Season", year)
    results.insert(1, "Round", rnd)
    results.insert(2, "EventName", session.event["EventName"])
    return df, results


def _race_files(raw_dir: Path, year: int, rnd: int) -> tuple[Path, Path]:
    return (
        raw_dir / f"{year}_{rnd:02d}_laps.parquet",
        raw_dir / f"{year}_{rnd:02d}_results.parquet",
    )


def save_race(year: int, rnd: int, raw_dir: Path = DATA_RAW / "races") -> int:
    """Load one race and write its parquet files. Returns a process exit code."""
    enable_cache()
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        laps, results = load_race(year, rnd)
    except RateLimitExceededError as exc:
        print(f"  ! rate limit: {exc}", flush=True)
        return RATE_LIMIT_EXIT_CODE
    except DataNotLoadedError:
        print(
            f"  ! {year} round {rnd}: timing data unavailable. From 2023 the F1 live-timing "
            "archive requires an F1 account (set F1_USERNAME / F1_PASSWORD).",
            flush=True,
        )
        return 1
    except Exception as exc:  # a missing/corrupt session should not stop the season
        print(f"  ! {year} round {rnd}: {type(exc).__name__}: {exc}", flush=True)
        return 1
    lap_file, res_file = _race_files(raw_dir, year, rnd)
    laps.to_parquet(lap_file, index=False)
    results.to_parquet(res_file, index=False)
    print(f"{year} R{rnd:02d} {laps['EventName'].iloc[0]}: {len(laps)} laps", flush=True)
    return 0


def download_fastf1(
    seasons: list[int] = FASTF1_SEASONS,
    raw_dir: Path = DATA_RAW / "races",
    min_seconds_per_race: float = 90.0,
    race_timeout: float = 300.0,
) -> None:
    """Load every race of ``seasons`` with FastF1 (paced and resumable)."""
    enable_cache()
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[int, int]] = []
    for year in seasons:
        sched = race_schedule(year)
        todo += [
            (year, int(rnd))
            for rnd in sched["RoundNumber"]
            if not all(p.exists() for p in _race_files(raw_dir, year, int(rnd)))
        ]
    print(f"{len(todo)} races to download", flush=True)

    attempts: dict[tuple[int, int], int] = {}
    while todo:
        year, rnd = todo.pop(0)
        started = time.monotonic()
        cmd = [sys.executable, "-m", "f1tyres.data", "--race", str(year), str(rnd)]
        try:
            code = subprocess.run(cmd, stdin=subprocess.DEVNULL, timeout=race_timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"  ! {year} R{rnd:02d} timed out", flush=True)
            code = 1
        attempts[(year, rnd)] = attempts.get((year, rnd), 0) + 1
        if code == RATE_LIMIT_EXIT_CODE:
            print("  rate limit reached - pausing 15 minutes", flush=True)
            todo.insert(0, (year, rnd))
            time.sleep(15 * 60)
            continue
        if code != 0 and attempts[(year, rnd)] < 3:
            todo.append((year, rnd))  # retry later
        time.sleep(max(0.0, min_seconds_per_race - (time.monotonic() - started)))


def combine(
    raw_dir: Path = DATA_RAW / "races", out_dir: Path = DATA_PROCESSED
) -> dict[str, pd.DataFrame]:
    """Merge every per-race file (FastF1 or OpenF1) into laps / results / events tables."""
    laps = pd.concat(
        [pd.read_parquet(p) for p in sorted(raw_dir.glob("*_laps.parquet"))], ignore_index=True
    )
    results = pd.concat(
        [pd.read_parquet(p) for p in sorted(raw_dir.glob("*_results.parquet"))], ignore_index=True
    )
    if "Source" not in laps:
        laps["Source"] = None
    inferred = pd.Series(np.where(laps["Season"] >= 2023, "OpenF1", "FastF1"), index=laps.index)
    laps["Source"] = laps["Source"].where(laps["Source"].notna(), inferred)
    laps["Deleted"] = laps["Deleted"].astype("boolean")
    events = (
        laps.groupby(["Season", "Round", "EventName"], as_index=False)
        .agg(
            TotalLaps=("TotalLaps", "max"),
            Drivers=("Driver", "nunique"),
            Source=("Source", "first"),
        )
        .sort_values(["Season", "Round"])
    )
    frames = {"laps": laps, "results": results, "events": events}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return frames


def load_tables(data_dir: Path = DATA_PROCESSED) -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(data_dir / f"{n}.parquet") for n in ("laps", "results", "events")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=FASTF1_SEASONS)
    parser.add_argument(
        "--race",
        type=int,
        nargs=2,
        metavar=("YEAR", "ROUND"),
        help="load a single race (used internally by the paced bulk loader)",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="skip downloading; rebuild data/processed from the race files",
    )
    args = parser.parse_args()
    if args.race:
        sys.exit(save_race(*args.race))
    if not args.combine_only:
        download_fastf1(args.seasons)
    combine()


if __name__ == "__main__":
    main()
