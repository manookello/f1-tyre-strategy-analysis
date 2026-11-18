"""Charts for the notebooks, README and app (same visual system as the companion projects).

Tyre compounds use the colours F1 viewers already know (soft = red, medium = yellow,
hard = grey, intermediate = green, wet = blue), drawn from the project palette, and every stint bar
also carries the compound letter so colour is never the only cue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1tyres.config import DRY_COMPOUNDS, FIGURES

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OTHER = "#c3c2b7"
COMPOUND_COLORS = {
    "SOFT": "#e34948",
    "MEDIUM": "#eda100",
    "HARD": "#8f8e88",
    "INTERMEDIATE": "#008300",
    "WET": "#2a78d6",
}
LABEL_ON = {
    "SOFT": "#ffffff",
    "MEDIUM": "#0b0b0b",
    "HARD": "#ffffff",
    "INTERMEDIATE": "#ffffff",
    "WET": "#ffffff",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )


def save(fig: plt.Figure, name: str, folder: Path = FIGURES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def _compound_legend(ax: plt.Axes, compounds, **kw) -> None:
    handles = [
        mpl.patches.Patch(color=COMPOUND_COLORS[c], label=c.title())
        for c in compounds
        if c in COMPOUND_COLORS
    ]
    ax.legend(handles=handles, **kw)


# --------------------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------------------


def strategy_timeline(stint_table: pd.DataFrame, order: list[str], title: str) -> plt.Figure:
    """Horizontal stint bars per driver (finishing order top to bottom)."""
    order = [d for d in order if d in set(stint_table["Driver"])]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 1.4))
    for y, drv in enumerate(order[::-1]):
        for s in stint_table[stint_table["Driver"] == drv].itertuples():
            color = COMPOUND_COLORS.get(s.Compound, OTHER)
            ax.barh(
                y,
                s.Laps,
                left=s.StartLap - 1,
                height=0.62,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
            )
            if s.Laps >= 4 and isinstance(s.Compound, str):
                ax.text(
                    s.StartLap - 1 + s.Laps / 2,
                    y,
                    s.Compound[0],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=LABEL_ON.get(s.Compound, INK),
                    fontweight="semibold",
                )
    ax.set_yticks(range(len(order)), order[::-1], fontsize=8.5)
    ax.set_xlabel("Lap")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, stint_table["EndLap"].max())
    _compound_legend(
        ax,
        [c for c in COMPOUND_COLORS if c in set(stint_table["Compound"])],
        loc="upper left",
        bbox_to_anchor=(0, -0.06 - 1.2 / (len(order) + 4)),
        ncol=5,
        fontsize=8.5,
        handlelength=1,
    )
    ax.set_title(title, pad=22)
    _subtitle(ax, "Tyre stints by driver, in finishing order")
    return fig


# --------------------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------------------


def degradation_summary(deg: pd.DataFrame) -> plt.Figure:
    """Per-race degradation slopes for each compound, with the median marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(3)
    for x, comp in enumerate(DRY_COMPOUNDS):
        d = deg[deg["Compound"] == comp]["DegSlope"] * 1000
        ax.scatter(
            x + rng.uniform(-0.18, 0.18, len(d)),
            d,
            s=22,
            color=COMPOUND_COLORS[comp],
            alpha=0.55,
            edgecolor="none",
        )
        med = d.median()
        ax.hlines(med, x - 0.3, x + 0.3, color=INK, linewidth=2)
        ax.text(x + 0.33, med, f"{med:.0f} ms/lap", va="center", fontsize=9, color=INK_2)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(range(len(DRY_COMPOUNDS)), [c.title() for c in DRY_COMPOUNDS])
    ax.set_xlim(-0.5, len(DRY_COMPOUNDS) - 0.1)
    ax.set_ylabel("Lap time lost per lap of tyre age (ms)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Softer tyres fade faster", pad=22)
    _subtitle(ax, "Fuel-corrected degradation per race (dots) and median (line)")
    return fig


def race_degradation_curves(clean_race: pd.DataFrame, model, title: str) -> plt.Figure:
    """Clean laps (driver pace removed) vs tyre age, with the fitted line per compound."""
    df = clean_race[clean_race["Compound"].isin(model.compounds)].copy()
    df["Adjusted"] = (
        df["FuelCorrected"]
        - df["Driver"].map(model.driver_effects)
        - model.track_evolution * df["LapNumber"]
    )
    base = min(o for o, _ in model.compounds.values())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for comp in [c for c in DRY_COMPOUNDS if c in model.compounds]:
        d = df[df["Compound"] == comp]
        ax.scatter(
            d["TyreLife"],
            d["Adjusted"] - base,
            s=10,
            alpha=0.25,
            color=COMPOUND_COLORS[comp],
            edgecolor="none",
        )
        off, slope = model.compounds[comp]
        ages = np.arange(1, int(d["TyreLife"].max()) + 1)
        ax.plot(
            ages,
            off + slope * ages,
            color=COMPOUND_COLORS[comp],
            linewidth=2.5,
            label=f"{comp.title()}: +{off:.2f} s, {slope * 1000:+.0f} ms/lap",
        )
    lo, hi = np.nanpercentile(df["Adjusted"] - base, [2, 98])
    ax.set_ylim(lo - 0.3, hi + 0.3)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Lap time vs. fastest compound (s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title, pad=22)
    _subtitle(ax, "Clean laps with fuel, driver pace and track evolution removed")
    return fig


def pit_loss_by_circuit(pl: pd.DataFrame, top: int = 25) -> plt.Figure:
    med = pl.groupby("EventName")["PitLoss"].median().sort_values()
    med = med.iloc[-top:] if len(med) > top else med
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(med) + 1.2))
    ax.barh(
        [n.replace(" Grand Prix", "") for n in med.index], med.values, height=0.6, color=PALETTE[0]
    )
    for y, v in enumerate(med.values):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Time lost to a green-flag pit stop (s)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What a pit stop costs at each circuit", pad=22)
    _subtitle(ax, "Median over stops and seasons: in-lap + out-lap minus two normal laps")
    return fig


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def model_comparison(metrics: pd.DataFrame) -> plt.Figure:
    m = metrics.set_index("model")["mae_seconds"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    colors = [PALETTE[0] if v == m.min() else OTHER for v in m]
    ax.barh(m.index, m.values, height=0.55, color=colors)
    for y, v in enumerate(m.values):
        ax.text(v + 0.01, y, f"{v:.2f} s", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Mean absolute error per lap (s), unseen races")
    ax.grid(axis="y", visible=False)
    ax.set_title("Predicting race lap times", pad=22)
    _subtitle(ax, "Grouped 5-fold cross-validation by race")
    return fig


def importance(imp: pd.DataFrame, top: int = 12) -> plt.Figure:
    imp = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.2))
    ax.barh(imp["label"], imp["importance"] * 100, height=0.55, color=PALETTE[0])
    ax.set_xlabel("Share of total split gain (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What moves a lap time", pad=22)
    _subtitle(ax, "LightGBM feature importance (gain)")
    return fig


def simulator_curves(
    plans: pd.DataFrame, title: str, actual_pits: list[int] | None = None
) -> plt.Figure:
    """Best race time for each first-stop lap, 1-stop vs 2-stop."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for stops, color in ((1, PALETTE[0]), (2, PALETTE[1])):
        p = plans[plans["Stops"] == stops].copy()
        if p.empty:
            continue
        p["FirstStop"] = p["PitLaps"].map(lambda x: x[0])
        best = p.groupby("FirstStop")["Gap"].min()
        ax.plot(best.index, best.values, color=color, label=f"{stops}-stop (best plan)")
    for lap in actual_pits or []:
        ax.axvline(lap, color=AXIS, linewidth=1)
        ax.text(lap, ax.get_ylim()[1], " winner pitted", fontsize=8, color=MUTED, va="top")
    ax.set_ylim(bottom=-1)
    ax.set_xlabel("Lap of the first pit stop")
    ax.set_ylabel("Slower than the optimum (s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, pad=22)
    _subtitle(ax, "Simulated race time vs. the best strategy found")
    return fig


# --------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------


def telemetry_overlay(tel: pd.DataFrame, drivers: list[str], title: str) -> plt.Figure:
    fig, axes = plt.subplots(
        3, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 0.8]}
    )
    for color, drv in zip(PALETTE, drivers, strict=False):
        d = tel[tel["Driver"] == drv]
        lap = d["LapTime"].iloc[0]
        label = f"{drv}  {int(lap // 60)}:{lap % 60:06.3f}"
        axes[0].plot(d["Distance"], d["Speed"], color=color, linewidth=1.6, label=label)
        axes[1].plot(d["Distance"], d["Throttle"], color=color, linewidth=1.4)
        axes[2].step(d["Distance"], d["Brake"], color=color, linewidth=1.4, where="post")
    axes[0].set_ylabel("Speed (km/h)")
    axes[1].set_ylabel("Throttle %")
    axes[2].set_ylabel("Brake")
    axes[2].set_yticks([0, 1], ["off", "on"])
    axes[2].set_xlabel("Distance along the lap (m)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title(title, pad=22)
    _subtitle(axes[0], "Fastest race laps compared")
    fig.tight_layout()
    return fig


def minisector_map(tel: pd.DataFrame, drivers: list[str], n_sectors: int = 25) -> plt.Figure:
    """Track map coloured by which driver was faster in each mini-sector."""
    a, b = (tel[tel["Driver"] == d].sort_values("Distance") for d in drivers)
    length = min(a["Distance"].max(), b["Distance"].max())
    edges = np.linspace(0, length, n_sectors + 1)

    def sector_times(d: pd.DataFrame) -> np.ndarray:
        t = np.interp(edges, d["Distance"], d["Time"])
        return np.diff(t)

    faster_a = sector_times(a) < sector_times(b)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    seg = np.clip(np.searchsorted(edges, a["Distance"], side="right") - 1, 0, n_sectors - 1)
    for s in range(n_sectors):
        pts = a[seg == s]
        if len(pts) < 2:
            continue
        ax.plot(
            pts["X"],
            pts["Y"],
            color=PALETTE[0] if faster_a[s] else PALETTE[1],
            linewidth=5,
            solid_capstyle="round",
        )
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=PALETTE[i],
            linewidth=5,
            label=f"{d} faster ({(faster_a if i == 0 else ~faster_a).sum()})",
        )
        for i, d in enumerate(drivers)
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9)
    ax.set_title("Who was quicker where", pad=12)
    return fig
