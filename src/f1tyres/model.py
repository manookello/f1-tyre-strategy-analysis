"""Lap-time model: predict a clean race lap relative to that race's median lap.

Target ``DeltaPct`` = (lap time / race median clean lap − 1) × 100, which makes laps comparable
across circuits. Validation is ``GroupKFold`` by race, so every score is for races the model never
saw (it has seen the same teams, drivers and circuits in *other* races).

Car performance is encoded as ``TeamSeason`` ("2024 McLaren"), because a team's car changes from
year to year. A ridge regression with these effects is a strong baseline because lap-time effects
are largely additive. With all four seasons LightGBM edges it, mainly by making smaller typical
errors. (On a smaller subset of races the ridge model was ahead, and fitting LightGBM to the ridge
residuals did not help.)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from f1tyres.config import MODELS

NUMERIC = [
    "TyreLife",
    "Stint",
    "LapNumber",
    "LapsRemaining",
    "LapFraction",
    "TrackTemp",
    "AirTemp",
    "Humidity",
    "WindSpeed",
    "Position",
]
CATEGORICAL = ["Compound", "TeamSeason", "Driver", "EventName"]
FEATURES = NUMERIC + ["FreshTyre"] + CATEGORICAL
TARGET = "DeltaPct"

FEATURE_LABELS = {
    "TyreLife": "Tyre age (laps)",
    "Stint": "Stint number",
    "LapNumber": "Lap number",
    "LapsRemaining": "Laps remaining (fuel load)",
    "LapFraction": "Race progress",
    "TrackTemp": "Track temperature",
    "AirTemp": "Air temperature",
    "Humidity": "Humidity",
    "WindSpeed": "Wind speed",
    "Position": "Running position (traffic)",
    "FreshTyre": "New tyre set",
    "Compound": "Compound",
    "TeamSeason": "Car (team × season)",
    "Driver": "Driver",
    "EventName": "Circuit",
}


def prepare(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    df["LapFraction"] = df["LapNumber"] / df["TotalLaps"]
    df["FreshTyre"] = df["FreshTyre"].astype(int)
    df["TeamSeason"] = df["Season"].astype(str) + " " + df["Team"].astype(str)
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def make_lgbm(random_state: int = 7) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1,
    )


def make_linear() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["TyreLife", "LapsRemaining", "TrackTemp"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Compound", "TeamSeason", "Driver"]),
        ]
    )
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])


def cross_validate(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold predictions for: race median (0), linear model and LightGBM."""
    df = df.dropna(subset=["TrackTemp", "Position"]).reset_index(drop=True)
    out = df[
        [
            "RaceId",
            "Season",
            "EventName",
            "Driver",
            "Compound",
            "TyreLife",
            "LapNumber",
            "LapTime",
            "RaceRef",
            TARGET,
        ]
    ].copy()
    out["pred_median"] = 0.0
    out["pred_linear"] = np.nan
    out["pred_lgbm"] = np.nan
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(df, groups=df["RaceId"]):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        lin = make_linear().fit(train, train[TARGET])
        out.loc[test_idx, "pred_linear"] = lin.predict(test)
        gbm = make_lgbm().fit(train[FEATURES], train[TARGET])
        out.loc[test_idx, "pred_lgbm"] = gbm.predict(test[FEATURES])
    return out


def cv_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in [
        ("Race median lap", "pred_median"),
        ("Linear (ridge)", "pred_linear"),
        ("LightGBM", "pred_lgbm"),
    ]:
        err_s = (pred[col] - pred[TARGET]).abs() * pred["RaceRef"] / 100
        rows.append(
            {
                "model": name,
                "mae_seconds": err_s.mean(),
                "median_ae_seconds": err_s.median(),
                "mae_pct": mean_absolute_error(pred[TARGET], pred[col]),
                "r2": r2_score(pred[TARGET], pred[col]),
            }
        )
    return pd.DataFrame(rows)


def fit_final(df: pd.DataFrame) -> LGBMRegressor:
    """LightGBM on all laps (used for feature importance)."""
    df = df.dropna(subset=["TrackTemp", "Position"])
    return make_lgbm().fit(df[FEATURES], df[TARGET])


def fit_linear(df: pd.DataFrame) -> Pipeline:
    """The ridge model on all laps (the most accurate model on unseen races)."""
    df = df.dropna(subset=["TrackTemp", "Position"])
    return make_linear().fit(df, df[TARGET])


def importance(model: LGBMRegressor) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": model.booster_.feature_name(), "importance": gain})
    imp["importance"] = imp["importance"] / imp["importance"].sum()
    imp["label"] = imp["feature"].map(FEATURE_LABELS)
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model, path: Path = MODELS / "laptime_model.joblib") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "target": TARGET}, path)
    return path
