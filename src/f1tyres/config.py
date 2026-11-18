"""Project paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"  # FastF1 HTTP cache (gitignored)
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

# The ground-effect era (2022-2025), consistent SOFT/MEDIUM/HARD labelling throughout.
# 2022 is loaded with FastF1 from the F1 live-timing archive; that archive now requires an F1
# account for anonymous users, so 2023-2025 come from the free OpenF1 API instead (same data,
# converted to the same schema in f1tyres.openf1).
SEASONS = [2022, 2023, 2024, 2025]
FASTF1_SEASONS = [2022]
OPENF1_SEASONS = [2023, 2024, 2025]
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Fuel effect: a common rule of thumb is ~0.03 s per lap of fuel burned (roughly 1.5-1.8 kg/lap
# at ~0.02-0.035 s/kg). We use it to remove the fuel trend from lap times. It is an assumption,
# documented in the README and exposed as a parameter everywhere it is used.
FUEL_EFFECT_S_PER_LAP = 0.03
