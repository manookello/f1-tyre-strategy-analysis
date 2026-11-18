"""Lap cleaning, stints, pit loss, tyre degradation and a race-strategy simulator.

Conventions
-----------
* **Clean lap** – dry compound, not an in/out lap, not lap 1, green-flag track status throughout,
  FastF1 ``IsAccurate``, no rainfall, and within 107 % of the driver's stint median.
* **Fuel correction** – lap time minus ``fuel_effect × laps remaining`` (≈ 0.03 s per lap of fuel),
  i.e. every lap expressed as if run with the final-lap fuel load. What is left of the trend
  within a stint is mostly tyre degradation (plus track evolution).
* **Race tyre model** – per race, a fixed-effects regression on clean, fuel-corrected laps:
  ``time = driver_effect + compound_offset + compound_slope × tyre_age + β × race_lap``. Driver
  effects absorb car/driver pace; the race-lap term absorbs track evolution (rubbering-in) and any
  error in the assumed fuel effect, so later-race compounds are not credited with the track
  getting faster. Compound offsets and slopes are identified from drivers who ran more than one
  compound (which the regulations force).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1tyres.config import DRY_COMPOUNDS, FUEL_EFFECT_S_PER_LAP

NON_GREEN = set("4567")  # 4 = safety car, 5 = red flag, 6/7 = virtual safety car


def race_id(df: pd.DataFrame) -> pd.Series:
    return df["Season"].astype(str) + "-" + df["Round"].astype(str).str.zfill(2)


def is_green(track_status) -> bool:
    return isinstance(track_status, str) and not (set(track_status) & NON_GREEN)


def clean_laps(laps: pd.DataFrame, fuel_effect: float = FUEL_EFFECT_S_PER_LAP) -> pd.DataFrame:
    """Return representative green-flag dry laps with a fuel-corrected lap time."""
    df = laps.copy()
    df["RaceId"] = race_id(df)
    keep = (
        df["Compound"].isin(DRY_COMPOUNDS)
        & ~df["PitIn"]
        & ~df["PitOut"]
        & (df["LapNumber"] > 1)
        & df["TrackStatus"].map(is_green)
        & df["IsAccurate"]
        & ~df["Rainfall"].fillna(False).astype(bool)
        & df["LapTime"].notna()
        & df["TyreLife"].notna()
    )
    df = df[keep].copy()
    stint_median = df.groupby(["RaceId", "Driver", "Stint"])["LapTime"].transform("median")
    df = df[df["LapTime"] <= 1.07 * stint_median].copy()
    df["LapsRemaining"] = df["TotalLaps"] - df["LapNumber"]
    df["FuelCorrected"] = df["LapTime"] - fuel_effect * df["LapsRemaining"]
    race_ref = df.groupby("RaceId")["LapTime"].transform("median")
    df["RaceRef"] = race_ref
    df["DeltaPct"] = (df["LapTime"] / race_ref - 1) * 100
    return df.reset_index(drop=True)


def stints(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, driver, stint): compound and lap range (all laps, not only clean)."""
    df = laps.dropna(subset=["Stint"]).copy()
    df["RaceId"] = race_id(df)
    out = (
        df.groupby(["RaceId", "Season", "Round", "EventName", "Driver", "Team", "Stint"])
        .agg(
            Compound=("Compound", "first"),
            StartLap=("LapNumber", "min"),
            EndLap=("LapNumber", "max"),
            FreshTyre=("FreshTyre", "first"),
            StartTyreLife=("TyreLife", "min"),
        )
        .reset_index()
    )
    out["Laps"] = out["EndLap"] - out["StartLap"] + 1
    return out


def driver_strategies(stint_table: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Per driver and race: number of stops and compound sequence, joined to the result."""
    seq = (
        stint_table.sort_values("Stint")
        .groupby(["RaceId", "Season", "Round", "EventName", "Driver"])
        .agg(
            Stops=("Stint", lambda s: s.nunique() - 1),
            Sequence=(
                "Compound",
                lambda c: "-".join(x[0] if isinstance(x, str) else "?" for x in c),
            ),
        )
        .reset_index()
    )
    res = results.rename(columns={"Abbreviation": "Driver"})[
        ["Season", "Round", "Driver", "Position", "GridPosition", "Status"]
    ]
    return seq.merge(res, on=["Season", "Round", "Driver"], how="left")


def pit_loss(laps: pd.DataFrame) -> pd.DataFrame:
    """Time lost to a green-flag pit stop, per race (median over stops).

    For each stop: (in-lap + out-lap) − 2 × the driver's median clean lap in that race.
    Stops where either lap was not fully green are excluded (SC/VSC stops are cheap).
    """
    df = laps.reset_index(drop=True)
    df["RaceId"] = race_id(df)
    df = df.sort_values(["RaceId", "Driver", "LapNumber"])
    clean = clean_laps(laps)
    ref = clean.groupby(["RaceId", "Driver"])["LapTime"].median().rename("RefLap")
    nxt = df.groupby(["RaceId", "Driver"]).shift(-1)
    stops = df[df["PitIn"] & nxt["PitOut"].eq(True)].copy()
    stops["OutLapTime"] = nxt.loc[stops.index, "LapTime"]
    stops["OutStatus"] = nxt.loc[stops.index, "TrackStatus"]
    stops["OutCompound"] = nxt.loc[stops.index, "Compound"]
    stops["OutRain"] = nxt.loc[stops.index, "Rainfall"].eq(True)
    stops = stops[
        stops["TrackStatus"].map(is_green)
        & stops["OutStatus"].map(is_green)
        & stops["Compound"].isin(DRY_COMPOUNDS)
        & stops["OutCompound"].isin(DRY_COMPOUNDS)
        & ~stops["Rainfall"].eq(True)
        & ~stops["OutRain"]
    ]
    stops = stops.join(ref, on=["RaceId", "Driver"])
    stops["Loss"] = stops["LapTime"] + stops["OutLapTime"] - 2 * stops["RefLap"]
    stops = stops[stops["Loss"].between(5, 60)]  # drop drive-throughs / damage / outliers
    return (
        stops.groupby(["RaceId", "Season", "Round", "EventName"])["Loss"]
        .agg(PitLoss="median", Stops="size")
        .reset_index()
    )


# --------------------------------------------------------------------------------------
# Per-race tyre model
# --------------------------------------------------------------------------------------


@dataclass
class RaceTyreModel:
    race_id: str
    compounds: dict[str, tuple[float, float]]  # compound -> (offset s, slope s/lap)
    base_lap: float  # fuel-corrected lap of a front-running car on the fastest compound, new tyre
    track_evolution: float  # s per race lap (negative = track getting faster)
    driver_effects: pd.Series
    max_stint: dict[str, int]
    n_laps: int
    rmse: float


def observed_max_stint(stint_table: pd.DataFrame) -> dict[str, int]:
    """Longest stint actually run on each compound (from the full stint table of one race)."""
    s = stint_table[stint_table["Compound"].isin(DRY_COMPOUNDS)]
    return s.groupby("Compound")["Laps"].max().astype(int).to_dict()


def fit_race_tyre_model(
    clean: pd.DataFrame,
    min_laps_per_compound: int = 30,
    max_stint: dict[str, int] | None = None,
    min_drivers_per_compound: int = 3,
) -> RaceTyreModel:
    """Fixed-effects regression for one race's clean laps (see module docstring).

    A compound is modelled only if at least ``min_drivers_per_compound`` drivers ran it for
    ``min_laps_per_compound`` clean laps in total. With fewer drivers its offset cannot be
    separated from those drivers' own pace, and the fit becomes meaningless.
    """
    df = clean.copy()
    counts = df["Compound"].value_counts()
    n_drivers = df.groupby("Compound")["Driver"].nunique()
    comps = [
        c
        for c in DRY_COMPOUNDS
        if counts.get(c, 0) >= min_laps_per_compound
        and n_drivers.get(c, 0) >= min_drivers_per_compound
    ]
    if not comps:
        raise ValueError("no compound has enough clean laps to model")
    df = df[df["Compound"].isin(comps)]
    drivers = sorted(df["Driver"].unique())
    base_comp = comps[0]

    def design(frame: pd.DataFrame) -> np.ndarray:
        cols = [(frame["Driver"] == d).astype(float) for d in drivers]
        cols += [(frame["Compound"] == c).astype(float) for c in comps[1:]]
        cols += [(frame["Compound"] == c) * frame["TyreLife"] for c in comps]
        cols += [frame["LapNumber"].astype(float)]
        return np.column_stack(cols)

    y = df["FuelCorrected"].to_numpy()
    X = design(df)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    # one robust refit without gross outliers (traffic, lock-ups)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    ok = np.abs(resid) <= 3 * max(mad, 1e-6)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    rmse = float(np.sqrt(np.mean((y[ok] - X[ok] @ coef) ** 2)))

    nd = len(drivers)
    driver_effects = pd.Series(coef[:nd], index=drivers)
    raw = {base_comp: 0.0} | {c: float(coef[nd + i]) for i, c in enumerate(comps[1:])}
    slopes = {c: float(coef[nd + len(comps) - 1 + i]) for i, c in enumerate(comps)}
    fastest = min(raw, key=raw.get)
    offsets = {c: raw[c] - raw[fastest] for c in comps}
    track_evolution = float(coef[-1])
    # a front-running car (mean of the 3 quickest driver effects) on the fastest compound
    base_lap = float(driver_effects.nsmallest(3).mean() + raw[fastest])
    if max_stint is None:
        max_stint = (
            df.groupby(["Driver", "Stint", "Compound"])["TyreLife"]
            .max()
            .groupby("Compound")
            .max()
            .astype(int)
            .to_dict()
        )
    return RaceTyreModel(
        race_id=str(df["RaceId"].iloc[0]),
        compounds={c: (offsets[c], slopes[c]) for c in comps},
        base_lap=base_lap,
        track_evolution=track_evolution,
        driver_effects=driver_effects,
        max_stint=max_stint,
        n_laps=int(ok.sum()),
        rmse=rmse,
    )


def degradation_table(
    clean: pd.DataFrame, stint_table: pd.DataFrame | None = None, min_laps_per_compound: int = 30
) -> pd.DataFrame:
    """Compound offsets and degradation slopes for every race.

    Fits with physically implausible values (|slope| ≥ 0.5 s/lap or |offset| ≥ 5 s) are dropped.
    They come from compounds run by too few drivers to separate tyre from driver.
    """
    rows = []
    for rid, race in clean.groupby("RaceId"):
        if race["Compound"].nunique() < 2:
            continue
        ms = None
        if stint_table is not None:
            ms = observed_max_stint(stint_table[stint_table["RaceId"] == rid])
        try:
            m = fit_race_tyre_model(race, min_laps_per_compound, ms)
        except ValueError:
            continue
        if len(m.compounds) < 2:
            continue
        for comp, (offset, slope) in m.compounds.items():
            if abs(slope) >= 0.5 or abs(offset) >= 5:
                continue
            rows.append(
                {
                    "RaceId": rid,
                    "Season": race["Season"].iloc[0],
                    "Round": race["Round"].iloc[0],
                    "EventName": race["EventName"].iloc[0],
                    "Compound": comp,
                    "Offset": offset,
                    "DegSlope": slope,
                    "MaxStint": m.max_stint.get(comp),
                    "TrackEvolution": m.track_evolution,
                    "TrackTemp": race["TrackTemp"].median(),
                    "ModelRMSE": m.rmse,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Strategy simulator
# --------------------------------------------------------------------------------------


def stint_time(
    base: float, offset: float, slope: float, length: np.ndarray, start_age: int = 0
) -> np.ndarray:
    """Total time of a stint of ``length`` laps on a tyre that starts at ``start_age`` laps."""
    length = np.asarray(length, dtype=float)
    ages = start_age * length + length * (length + 1) / 2  # sum of ages 1..L (+ start age)
    return length * (base + offset) + max(slope, 0.0) * ages


def simulate_strategies(
    total_laps: int,
    base_lap: float,
    compounds: dict[str, tuple[float, float]],
    pit_loss: float,
    max_stint: dict[str, int] | None = None,
    max_stops: int = 2,
    min_stint: int = 5,
    fuel_effect: float = FUEL_EFFECT_S_PER_LAP,
) -> pd.DataFrame:
    """Enumerate 1- and 2-stop plans (≥ 2 different compounds) and their race times.

    Lap time = base + compound offset + slope × tyre age + fuel effect × laps remaining. The fuel
    term is identical for every plan, so it only shifts totals. One row per plan, best first.
    """
    max_stint = max_stint or {}
    fuel_total = fuel_effect * total_laps * (total_laps - 1) / 2
    rows = []
    comps = list(compounds)
    for stops in range(1, max_stops + 1):
        for seq in itertools.product(comps, repeat=stops + 1):
            if len(set(seq)) < 2:
                continue
            for pits in itertools.combinations(range(min_stint, total_laps - min_stint + 1), stops):
                bounds = (0, *pits, total_laps)
                lengths = np.diff(bounds)
                if (lengths < min_stint).any():
                    continue
                if any(
                    L > max_stint.get(c, total_laps) + 3 for c, L in zip(seq, lengths, strict=True)
                ):
                    continue
                t = sum(
                    float(stint_time(base_lap, *compounds[c], L))
                    for c, L in zip(seq, lengths, strict=True)
                )
                rows.append(
                    {
                        "Stops": stops,
                        "Sequence": "-".join(c[0] for c in seq),
                        "PitLaps": list(pits),
                        "StintLengths": lengths.tolist(),
                        # stint order does not change a linear model's total -> dedupe key
                        "Key": (stops, tuple(sorted(zip(seq, lengths.tolist(), strict=True)))),
                        "RaceTime": t + stops * pit_loss + fuel_total,
                    }
                )
    out = (
        pd.DataFrame(rows)
        .sort_values(["RaceTime", "Sequence"])
        .drop_duplicates("Key")
        .drop(columns="Key")
        .reset_index(drop=True)
    )
    out["Gap"] = out["RaceTime"] - out["RaceTime"].iloc[0]
    return out


def best_by_stops(plans: pd.DataFrame) -> pd.DataFrame:
    return plans.loc[plans.groupby("Stops")["RaceTime"].idxmin()].reset_index(drop=True)


def had_neutralisation(race_laps: pd.DataFrame) -> bool:
    """True if the race had a safety car, VSC or red flag at any point."""
    return bool(race_laps["TrackStatus"].dropna().map(lambda s: bool(set(s) & NON_GREEN)).any())


def strategy_review(
    laps: pd.DataFrame, results: pd.DataFrame, pit_losses: pd.DataFrame
) -> pd.DataFrame:
    """For each dry race: the simulator's best plan vs. what the winner actually did."""
    laps = laps.reset_index(drop=True)
    laps["RaceId"] = race_id(laps)
    clean = clean_laps(laps)
    st = stints(laps)
    strat = driver_strategies(st, results)
    rows = []
    for rid, race_clean in clean.groupby("RaceId"):
        race_laps = laps[laps["RaceId"] == rid]
        wet = (~race_laps["Compound"].isin(DRY_COMPOUNDS)).mean() > 0.05
        loss = pit_losses.loc[pit_losses["RaceId"] == rid, "PitLoss"]
        if wet or race_clean["Compound"].nunique() < 2 or loss.empty:
            continue
        try:
            model = fit_race_tyre_model(
                race_clean, max_stint=observed_max_stint(st[st["RaceId"] == rid])
            )
        except ValueError:
            continue
        implausible = any(abs(s) >= 0.5 or abs(o) >= 5 for o, s in model.compounds.values())
        if len(model.compounds) < 2 or implausible:
            continue
        total = int(race_laps["TotalLaps"].iloc[0])
        plans = simulate_strategies(
            total, model.base_lap, model.compounds, float(loss.iloc[0]), model.max_stint
        )
        best = best_by_stops(plans).set_index("Stops")
        winner = strat[(strat["RaceId"] == rid) & (strat["Position"] == 1)]
        rows.append(
            {
                "RaceId": rid,
                "Season": int(race_laps["Season"].iloc[0]),
                "Round": int(race_laps["Round"].iloc[0]),
                "EventName": race_laps["EventName"].iloc[0],
                "TotalLaps": total,
                "PitLoss": float(loss.iloc[0]),
                "Neutralised": had_neutralisation(race_laps),
                "BestStops": int(plans["Stops"].iloc[0]),
                "BestPlan": plans["Sequence"].iloc[0],
                "BestPitLaps": plans["PitLaps"].iloc[0],
                "OneStopVsTwoStop": (
                    float(best.loc[1, "RaceTime"] - best.loc[2, "RaceTime"])
                    if {1, 2} <= set(best.index)
                    else np.nan
                ),
                "WinnerStops": int(winner["Stops"].iloc[0]) if not winner.empty else np.nan,
                "WinnerPlan": winner["Sequence"].iloc[0] if not winner.empty else None,
            }
        )
    return pd.DataFrame(rows)
