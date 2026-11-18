"""Load 2023+ race laps from the OpenF1 API (https://openf1.org) in the same schema as FastF1.

Why two sources? From 2023 onward the official F1 live-timing archive that FastF1 reads requires
an F1 account. OpenF1 serves the same timing data for every session since 2023, free and without
authentication (limits: 3 requests/s, 30 requests/min). This module converts OpenF1's laps,
stints, pit stops, race-control messages and weather into per-lap rows with the *same columns*
as :func:`f1tyres.data.load_race`, so the analysis treats every season identically.

Mapping
-------
* ``Compound``, ``Stint``, ``TyreLife``, ``FreshTyre`` from ``/stints`` (tyre age at stint
  start + laps into the stint, counting the current lap like FastF1).
* ``PitOut`` from ``is_pit_out_lap``; ``PitIn`` = the lap before a pit-out lap.
* ``TrackStatus`` per lap from ``/race_control``: safety car (``4``), red flag (``5``) and VSC
  (``6``) from deployment until it ends; ``1`` otherwise.
* ``Position`` at the end of each lap from lap completion times.
* Weather joined to each lap by time (as-of the lap start).

Usage::

    python -m f1tyres.openf1 --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from f1tyres.config import DATA_RAW

BASE_URL = "https://api.openf1.org/v1"


class OpenF1Client:
    def __init__(self, cache_dir: Path = DATA_RAW / "openf1", min_interval: float = 2.2) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval  # 2.2 s ~ 27 requests/minute, under the 30/min limit
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "f1-tyre-strategy-analysis/0.1 (open-source research)"
        self._last = 0.0

    def get(self, endpoint: str, **params) -> list[dict]:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return self.get_query(endpoint, query)

    def get_query(self, endpoint: str, query: str) -> list[dict]:
        """GET with a raw query string (needed for OpenF1's ``date>=`` / ``date<=`` filters)."""
        key = f"{endpoint}?{query}"
        cache = self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(f"{BASE_URL}/{endpoint}?{query}", timeout=60)
            except requests.RequestException:
                time.sleep(2 ** (attempt + 2))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(15 * (attempt + 1))
                continue
            if resp.status_code == 404:
                data: list[dict] = []
            else:
                resp.raise_for_status()
                data = resp.json()
            cache.write_text(json.dumps(data), encoding="utf-8")
            return data
        raise RuntimeError(f"Giving up on {key}")


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="ISO8601")


def race_sessions(client: OpenF1Client, year: int) -> pd.DataFrame:
    sessions = pd.DataFrame(client.get("sessions", year=year, session_name="Race"))
    if "is_cancelled" in sessions:
        sessions = sessions[~sessions["is_cancelled"].fillna(False).astype(bool)]
    meetings = pd.DataFrame(client.get("meetings", year=year))[["meeting_key", "meeting_name"]]
    sessions = sessions.merge(meetings, on="meeting_key", how="left")
    sessions = sessions.sort_values("date_start").reset_index(drop=True)
    sessions["round"] = np.arange(1, len(sessions) + 1)
    return sessions


def _track_status_by_lap(rc: pd.DataFrame, lap_of_time, total_laps: int) -> dict[int, str]:
    """Per-lap status string ('1' green, '4' SC, '5' red, '6' VSC) from race-control messages."""
    status = {lap: set() for lap in range(1, total_laps + 1)}
    if rc.empty:
        return {lap: "1" for lap in status}
    rc = rc.sort_values("date")
    rc["lap"] = rc["lap_number"].fillna(rc["date"].map(lap_of_time)).astype(int)
    msg = rc["message"].fillna("").str.upper()

    def mark(start_mask, end_mask, code):
        start = None
        for lap, s, e in zip(rc["lap"], start_mask, end_mask, strict=True):
            if s and start is None:
                start = lap
            elif e and start is not None:
                for lp in range(start, lap + 1):
                    status.setdefault(lp, set()).add(code)
                start = None
        if start is not None:
            for lp in range(start, total_laps + 1):
                status.setdefault(lp, set()).add(code)

    is_vsc = msg.str.contains("VIRTUAL SAFETY CAR")
    mark(is_vsc & msg.str.contains("DEPLOYED"), is_vsc & msg.str.contains("ENDING"), "6")
    is_sc = msg.str.contains("SAFETY CAR") & ~is_vsc
    mark(is_sc & msg.str.contains("DEPLOYED"), is_sc & msg.str.contains("IN THIS LAP"), "4")
    # only explicit red-flag messages (a "RED" flag alone is also used for the pit-exit light)
    red = msg.str.contains("RED FLAG")
    for lap in rc.loc[red, "lap"]:
        status.setdefault(int(lap), set()).add("5")
    return {lap: "".join(sorted(codes)) or "1" for lap, codes in status.items()}


def load_race(client: OpenF1Client, session: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Laps (FastF1-like schema) and results for one race session."""
    key = int(session["session_key"])
    laps = pd.DataFrame(client.get("laps", session_key=key))
    stints = pd.DataFrame(client.get("stints", session_key=key))
    rc = pd.DataFrame(client.get("race_control", session_key=key))
    weather = pd.DataFrame(client.get("weather", session_key=key))
    drivers = pd.DataFrame(client.get("drivers", session_key=key))
    result = pd.DataFrame(client.get("session_result", session_key=key))
    if laps.empty or stints.empty:
        raise ValueError(f"no lap/stint data for session {key}")

    laps = laps.dropna(subset=["lap_number"]).copy()
    laps["lap_number"] = laps["lap_number"].astype(int)
    laps["date_start"] = _to_dt(laps["date_start"])
    laps = laps.sort_values(["driver_number", "lap_number"])
    total_laps = int(laps["lap_number"].max())

    # tyres
    rows = []
    for st in stints.itertuples():
        if pd.isna(st.lap_start) or pd.isna(st.lap_end):
            continue
        age0 = 0 if pd.isna(st.tyre_age_at_start) else int(st.tyre_age_at_start)
        for lap in range(int(st.lap_start), int(st.lap_end) + 1):
            rows.append(
                {
                    "driver_number": st.driver_number,
                    "lap_number": lap,
                    "Stint": float(st.stint_number),
                    "Compound": st.compound,
                    "TyreLife": float(age0 + lap - int(st.lap_start) + 1),
                    "FreshTyre": age0 == 0,
                }
            )
    tyres = pd.DataFrame(rows).drop_duplicates(["driver_number", "lap_number"], keep="last")
    df = laps.merge(tyres, on=["driver_number", "lap_number"], how="left")

    # pit in/out
    df["PitOut"] = df["is_pit_out_lap"].eq(True)
    df["PitIn"] = df.groupby("driver_number")["PitOut"].shift(-1).eq(True)

    # lap end times -> running position
    nxt = df.groupby("driver_number")["date_start"].shift(-1)
    df["lap_end"] = nxt.fillna(df["date_start"] + pd.to_timedelta(df["lap_duration"], unit="s"))
    df["Position"] = df.groupby("lap_number")["lap_end"].rank(method="first")

    # track status
    lap_starts = df.groupby("lap_number")["date_start"].min().sort_index()

    def lap_of_time(ts) -> int:
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        idx = lap_starts.index[lap_starts <= ts]
        return int(idx.max()) if len(idx) else 1

    if not rc.empty:
        rc["date"] = _to_dt(rc["date"])
    status = _track_status_by_lap(rc, lap_of_time, total_laps)
    df["TrackStatus"] = df["lap_number"].map(status).fillna("1")

    # weather (as-of each lap's start; laps without a start time, e.g. around red flags, take
    # the previous lap's conditions)
    if not weather.empty:
        weather["date"] = _to_dt(weather["date"])
        w = weather.sort_values("date").rename(
            columns={
                "air_temperature": "AirTemp",
                "track_temperature": "TrackTemp",
                "humidity": "Humidity",
                "rainfall": "Rainfall",
                "wind_speed": "WindSpeed",
            }
        )
        wcols = ["AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]
        timed = df[df["date_start"].notna()].sort_values("date_start")
        timed = pd.merge_asof(
            timed, w[["date", *wcols]], left_on="date_start", right_on="date", direction="backward"
        )
        df = pd.concat([timed, df[df["date_start"].isna()]], ignore_index=True)
        df = df.sort_values(["driver_number", "lap_number"])
        df[wcols] = df.groupby("driver_number")[wcols].ffill().bfill()
        df["Rainfall"] = df["Rainfall"].fillna(0).astype(float) > 0
    else:
        for col in ("AirTemp", "TrackTemp", "Humidity", "WindSpeed"):
            df[col] = np.nan
        df["Rainfall"] = False

    names = drivers.drop_duplicates("driver_number").set_index("driver_number")
    session_start = _to_dt(pd.Series([session["date_start"]])).iloc[0]
    out = (
        pd.DataFrame(
            {
                "Season": int(session["year"]),
                "Round": int(session["round"]),
                "EventName": session["meeting_name"],
                "Driver": df["driver_number"].map(names["name_acronym"]),
                "DriverNumber": df["driver_number"].astype(str),
                "Team": df["driver_number"].map(names["team_name"]),
                "LapNumber": df["lap_number"].astype(float),
                "LapTime": df["lap_duration"].astype(float),
                "Stint": df["Stint"],
                "Compound": df["Compound"],
                "TyreLife": df["TyreLife"],
                "FreshTyre": df["FreshTyre"].eq(True),
                "TrackStatus": df["TrackStatus"],
                "IsAccurate": df["lap_duration"].notna()
                & ~df["PitOut"]
                & ~df["PitIn"]
                & (df["lap_number"] > 1),
                "Position": df["Position"],
                "Deleted": None,
                "Sector1Time": df["duration_sector_1"].astype(float),
                "Sector2Time": df["duration_sector_2"].astype(float),
                "Sector3Time": df["duration_sector_3"].astype(float),
                "SpeedST": df["st_speed"].astype(float),
                "LapStartTime": (df["date_start"] - session_start).dt.total_seconds(),
                "PitIn": df["PitIn"],
                "PitOut": df["PitOut"],
                "AirTemp": df["AirTemp"],
                "TrackTemp": df["TrackTemp"],
                "Humidity": df["Humidity"],
                "Rainfall": df["Rainfall"],
                "WindSpeed": df["WindSpeed"],
                "TotalLaps": total_laps,
                "Source": "OpenF1",
            }
        )
        .sort_values(["Driver", "LapNumber"])
        .reset_index(drop=True)
    )

    res = pd.DataFrame()
    if not result.empty:
        r = result.copy()

        def flag(col: str) -> pd.Series:
            return r[col].eq(True) if col in r else pd.Series(False, index=r.index)

        status = np.select(
            [flag("dsq"), flag("dns"), flag("dnf")],
            ["Disqualified", "Did not start", "Retired"],
            "Finished",
        )
        res = pd.DataFrame(
            {
                "Season": int(session["year"]),
                "Round": int(session["round"]),
                "EventName": session["meeting_name"],
                "Abbreviation": r["driver_number"].map(names["name_acronym"]),
                "FullName": r["driver_number"].map(names["full_name"]),
                "TeamName": r["driver_number"].map(names["team_name"]),
                "Position": pd.to_numeric(r["position"], errors="coerce"),
                "ClassifiedPosition": r["position"].astype(str),
                "GridPosition": np.nan,
                "Status": status,
                "Points": np.nan,
            }
        )
    return out, res


def fastest_lap(client: OpenF1Client, session_key: int, driver_number: int) -> pd.Series:
    laps = pd.DataFrame(client.get("laps", session_key=session_key))
    d = laps[(laps["driver_number"] == driver_number) & ~laps["is_pit_out_lap"].eq(True)]
    return d.loc[d["lap_duration"].idxmin()]


def lap_telemetry(
    client: OpenF1Client, session_key: int, driver_number: int, lap_number: int | None = None
) -> pd.DataFrame:
    """Speed / throttle / brake / gear / RPM and X-Y position for one lap (fastest by default).

    Distance along the lap is integrated from speed, so traces from two drivers can be overlaid.
    """
    laps = pd.DataFrame(client.get("laps", session_key=session_key))
    if lap_number is None:
        lap = fastest_lap(client, session_key, driver_number)
    else:
        lap = laps[
            (laps["driver_number"] == driver_number) & (laps["lap_number"] == lap_number)
        ].iloc[0]
    start = pd.Timestamp(lap["date_start"])
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    end = start + pd.Timedelta(seconds=float(lap["lap_duration"]))
    q = (
        f"session_key={session_key}&driver_number={driver_number}"
        f"&date>={quote(start.isoformat())}&date<={quote(end.isoformat())}"
    )
    car = pd.DataFrame(client.get_query("car_data", q))
    loc = pd.DataFrame(client.get_query("location", q))
    car["date"] = _to_dt(car["date"])
    car = car.sort_values("date").reset_index(drop=True)
    dt = car["date"].diff().dt.total_seconds().fillna(0)
    car["Distance"] = (car["speed"] / 3.6 * dt).cumsum()
    if not loc.empty:
        loc["date"] = _to_dt(loc["date"])
        car = pd.merge_asof(
            car, loc.sort_values("date")[["date", "x", "y"]], on="date", direction="nearest"
        )
    return pd.DataFrame(
        {
            "DriverNumber": driver_number,
            "LapNumber": int(lap["lap_number"]),
            "LapTime": float(lap["lap_duration"]),
            "Time": (car["date"] - start).dt.total_seconds(),
            "Distance": car["Distance"],
            "Speed": car["speed"],
            "Throttle": car["throttle"],
            "Brake": (car["brake"] > 0).astype(int),
            "Gear": car["n_gear"],
            "RPM": car["rpm"],
            "X": car.get("x"),
            "Y": car.get("y"),
        }
    )


def build_seasons(
    seasons: list[int], raw_dir: Path = DATA_RAW / "races", client: OpenF1Client | None = None
) -> None:
    """Write ``{year}_{round}_laps.parquet`` / ``_results.parquet`` for each OpenF1 race."""
    client = client or OpenF1Client()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in seasons:
        for _, sess in race_sessions(client, year).iterrows():
            rnd = int(sess["round"])
            lap_file = raw_dir / f"{year}_{rnd:02d}_laps.parquet"
            res_file = raw_dir / f"{year}_{rnd:02d}_results.parquet"
            if lap_file.exists() and res_file.exists():
                continue
            try:
                laps, results = load_race(client, sess)
            except Exception as exc:
                print(f"  ! {year} R{rnd:02d} {sess['meeting_name']}: {exc}", flush=True)
                continue
            laps.to_parquet(lap_file, index=False)
            results.to_parquet(res_file, index=False)
            print(f"{year} R{rnd:02d} {sess['meeting_name']}: {len(laps)} laps", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()
    build_seasons(args.seasons)


if __name__ == "__main__":
    main()
