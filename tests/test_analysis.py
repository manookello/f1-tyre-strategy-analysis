import numpy as np
import pandas as pd
import pytest

from f1tyres.analysis import (
    clean_laps,
    fit_race_tyre_model,
    is_green,
    simulate_strategies,
    stint_time,
)


def test_is_green():
    assert is_green("1")
    assert is_green("12")  # local yellow is still racing
    assert not is_green("14")  # safety car
    assert not is_green("6")  # VSC
    assert not is_green(None)


def test_stint_time_formula():
    # 3 laps, base 90, offset 0.5, slope 0.1 -> (90.5+0.1) + (90.5+0.2) + (90.5+0.3)
    assert float(stint_time(90, 0.5, 0.1, 3)) == pytest.approx(271.5 + 0.6)
    # negative slopes are clipped: tyres never get faster with age
    assert float(stint_time(90, 0.0, -0.1, 3)) == pytest.approx(270.0)


TRUE = {"SOFT": (0.0, 0.08), "MEDIUM": (0.35, 0.05), "HARD": (0.7, 0.03)}


def _synthetic_race(seed: int = 0) -> pd.DataFrame:
    """Five drivers, each running two compounds, with known tyre parameters."""
    rng = np.random.default_rng(seed)
    plans = [
        ("SOFT", "HARD"),
        ("MEDIUM", "HARD"),
        ("SOFT", "MEDIUM"),
        ("HARD", "MEDIUM"),
        ("MEDIUM", "SOFT"),
    ]
    rows = []
    for i, (c1, c2) in enumerate(plans):
        effect = 90 + 0.2 * i
        for lap in range(2, 51):
            comp, stint, age = (c1, 1, lap) if lap <= 25 else (c2, 2, lap - 25)
            off, slope = TRUE[comp]
            t = effect + off + slope * age - 0.02 * lap + 0.03 * (50 - lap)
            rows.append(
                {
                    "Season": 2024,
                    "Round": 1,
                    "EventName": "Test GP",
                    "Driver": f"D{i}",
                    "Team": "T",
                    "LapNumber": float(lap),
                    "LapTime": t + rng.normal(0, 0.05),
                    "Stint": float(stint),
                    "Compound": comp,
                    "TyreLife": float(age),
                    "TrackStatus": "1",
                    "IsAccurate": True,
                    "Rainfall": False,
                    "PitIn": False,
                    "PitOut": False,
                    "TotalLaps": 50,
                    "TrackTemp": 30.0,
                }
            )
    return pd.DataFrame(rows)


def test_clean_laps_fuel_correction_and_filters():
    race = _synthetic_race()
    race.loc[0, "TrackStatus"] = "4"  # a safety-car lap must be dropped
    race.loc[1, "PitIn"] = True
    clean = clean_laps(race)
    assert len(clean) == len(race) - 2
    row = clean.iloc[0]
    assert row["FuelCorrected"] == pytest.approx(row["LapTime"] - 0.03 * (50 - row["LapNumber"]))


def test_tyre_model_recovers_known_parameters():
    model = fit_race_tyre_model(clean_laps(_synthetic_race()), min_laps_per_compound=10)
    for comp, (off, slope) in TRUE.items():
        assert model.compounds[comp][0] == pytest.approx(off, abs=0.05)
        assert model.compounds[comp][1] == pytest.approx(slope, abs=0.005)
    assert model.track_evolution == pytest.approx(-0.02, abs=0.005)


def test_simulator_respects_rules_and_sorts():
    plans = simulate_strategies(50, 90.0, TRUE, pit_loss=22.0, max_stops=2)
    assert plans["Gap"].iloc[0] == 0
    assert plans["RaceTime"].is_monotonic_increasing
    for seq, lengths in zip(plans["Sequence"], plans["StintLengths"], strict=True):
        assert len(set(seq.split("-"))) >= 2
        assert sum(lengths) == 50
        assert min(lengths) >= 5


def test_cheap_pit_stops_favour_more_stops():
    expensive = simulate_strategies(50, 90.0, TRUE, pit_loss=40.0)
    cheap = simulate_strategies(50, 90.0, TRUE, pit_loss=5.0)
    assert expensive["Stops"].iloc[0] <= cheap["Stops"].iloc[0]
