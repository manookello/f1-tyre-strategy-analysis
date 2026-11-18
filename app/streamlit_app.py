"""F1 tyre strategy explorer.

Run with:  streamlit run app/streamlit_app.py
Reads only the processed files committed under data/processed/ (no API calls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from f1tyres.analysis import (
    best_by_stops,
    clean_laps,
    fit_race_tyre_model,
    observed_max_stint,
    simulate_strategies,
)
from f1tyres.config import DATA_PROCESSED, DRY_COMPOUNDS
from f1tyres.plots import COMPOUND_COLORS, PALETTE

st.set_page_config(page_title="F1 Tyre Strategy", page_icon="🛞", layout="wide")


@st.cache_data
def load():
    def read(name):
        path = DATA_PROCESSED / name
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    laps = read("laps.parquet")
    laps["RaceId"] = laps["Season"].astype(str) + "-" + laps["Round"].astype(str).str.zfill(2)
    return {
        "laps": laps,
        "results": read("results.parquet"),
        "stints": read("stints.parquet"),
        "strategies": read("driver_strategies.parquet"),
        "degradation": read("degradation.parquet"),
        "pit_loss": read("pit_loss.parquet"),
        "review": read("strategy_review.parquet"),
        "telemetry": read("telemetry.parquet"),
        "metrics": read("laptime_metrics.csv"),
        "importance": read("laptime_importance.csv"),
    }


@st.cache_data
def race_model(race_id: str):
    d = load()
    race_laps = d["laps"][d["laps"]["RaceId"] == race_id]
    model = fit_race_tyre_model(
        clean_laps(race_laps),
        max_stint=observed_max_stint(d["stints"][d["stints"]["RaceId"] == race_id]),
    )
    return model, clean_laps(race_laps)


def tidy(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    return fig


def race_picker(key: str, race_ids: list[str] | None = None) -> str:
    ev = data["laps"].drop_duplicates("RaceId")[["RaceId", "Season", "EventName"]]
    if race_ids is not None:
        ev = ev[ev["RaceId"].isin(race_ids)]
    c1, c2 = st.columns([1, 3])
    season = c1.selectbox("Season", sorted(ev["Season"].unique(), reverse=True), key=f"s_{key}")
    ev = ev[ev["Season"] == season].sort_values("RaceId")
    names = dict(zip(ev["RaceId"], ev["EventName"], strict=True))
    return c2.selectbox("Race", list(names), format_func=names.get, key=f"r_{key}")


data = load()
st.title("F1 Tyre Strategy & Pace")
st.caption(
    "2022–2025 race laps (FastF1 for 2022, OpenF1 for 2023–2025): stints, fuel-corrected "
    "degradation, pit loss, a strategy simulator and telemetry."
)

tab_strat, tab_deg, tab_sim, tab_tel, tab_model = st.tabs(
    ["Race strategy", "Tyre degradation", "Strategy simulator", "Telemetry", "Lap-time model"]
)

# ---------------------------------------------------------------------------------------
with tab_strat:
    rid = race_picker("strat")
    stn = data["stints"][data["stints"]["RaceId"] == rid]
    season, rnd = int(rid[:4]), int(rid[5:])
    res = data["results"][(data["results"]["Season"] == season) & (data["results"]["Round"] == rnd)]
    order = res.sort_values("Position")["Abbreviation"].tolist()
    order = [d for d in order if d in set(stn["Driver"])] + sorted(set(stn["Driver"]) - set(order))
    fig = go.Figure()
    for comp in COMPOUND_COLORS:
        s = stn[stn["Compound"] == comp]
        if s.empty:
            continue
        fig.add_trace(
            go.Bar(
                y=s["Driver"],
                x=s["Laps"],
                base=s["StartLap"] - 1,
                orientation="h",
                name=comp.title(),
                marker=dict(color=COMPOUND_COLORS[comp], line=dict(color="rgba(0,0,0,0)", width=2)),
                text=s["Compound"].str[0],
                textposition="inside",
                insidetextanchor="middle",
                customdata=np.stack([s["StartLap"], s["EndLap"]], axis=1),
                hovertemplate="%{y}: laps %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra>"
                + comp.title()
                + "</extra>",
            )
        )
    fig.update_layout(barmode="stack", xaxis_title="Lap", bargap=0.3)
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
    st.plotly_chart(tidy(fig, 26 * len(order) + 120), use_container_width=True)
    strat = data["strategies"][data["strategies"]["RaceId"] == rid].sort_values("Position")
    st.dataframe(
        strat[["Position", "Driver", "Stops", "Sequence", "Status"]],
        hide_index=True,
        use_container_width=True,
    )

# ---------------------------------------------------------------------------------------
with tab_deg:
    rid = race_picker("deg", data["degradation"]["RaceId"].unique().tolist())
    model, clean = race_model(rid)
    clean = clean[clean["Compound"].isin(model.compounds)].copy()
    base = min(o for o, _ in model.compounds.values())
    clean["Adjusted"] = (
        clean["FuelCorrected"]
        - clean["Driver"].map(model.driver_effects)
        - model.track_evolution * clean["LapNumber"]
        - base
    )
    lo, hi = clean["Adjusted"].quantile([0.02, 0.98])
    fig = go.Figure()
    for comp in [c for c in DRY_COMPOUNDS if c in model.compounds]:
        d = clean[clean["Compound"] == comp]
        off, slope = model.compounds[comp]
        fig.add_trace(
            go.Scatter(
                x=d["TyreLife"],
                y=d["Adjusted"],
                mode="markers",
                name=f"{comp.title()} laps",
                opacity=0.3,
                marker=dict(color=COMPOUND_COLORS[comp], size=5),
                hovertemplate="%{text}<extra></extra>",
                text=d["Driver"],
            )
        )
        ages = np.arange(1, int(d["TyreLife"].max()) + 1)
        fig.add_trace(
            go.Scatter(
                x=ages,
                y=off + slope * ages,
                mode="lines",
                name=f"{comp.title()}: {slope * 1000:+.0f} ms/lap",
                line=dict(color=COMPOUND_COLORS[comp], width=3),
            )
        )
    fig.update_layout(xaxis_title="Tyre age (laps)", yaxis_title="Lap time vs fastest compound (s)")
    fig.update_yaxes(range=[lo - 0.3, hi + 0.3])
    st.plotly_chart(tidy(fig, 460), use_container_width=True)
    st.caption("Clean laps with fuel, driver pace and track evolution removed.")
    st.dataframe(
        data["degradation"]
        .groupby("Compound")
        .agg(
            races=("RaceId", "nunique"),
            median_offset_s=("Offset", "median"),
            median_deg_ms_per_lap=("DegSlope", lambda s: s.median() * 1000),
        )
        .reindex(DRY_COMPOUNDS)
        .round(2),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------------------
with tab_sim:
    rid = race_picker("sim", data["review"]["RaceId"].tolist())
    row = data["review"].set_index("RaceId").loc[rid]
    model, _ = race_model(rid)
    c1, c2, c3 = st.columns(3)
    pit = c1.slider("Pit loss (s)", 15.0, 35.0, float(round(row["PitLoss"], 1)), 0.5)
    deg_mult = c2.slider(
        "Degradation multiplier",
        0.5,
        2.0,
        1.0,
        0.1,
        help="Scale every compound's degradation slope",
    )
    max_stops = c3.radio("Stops to consider", [1, 2], index=1, horizontal=True)
    comps = {c: (o, s * deg_mult) for c, (o, s) in model.compounds.items()}
    plans = simulate_strategies(
        int(row["TotalLaps"]), model.base_lap, comps, pit, model.max_stint, max_stops=max_stops
    )
    best = best_by_stops(plans)
    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Simulated optimum",
        f"{plans['Sequence'].iloc[0]} · pit lap(s) {', '.join(map(str, plans['PitLaps'].iloc[0]))}",
    )
    k2.metric("Winner actually ran", f"{row['WinnerPlan']}")
    if len(best) == 2:
        gap = best.set_index("Stops")["RaceTime"]
        k3.metric(
            "1-stop vs 2-stop", f"{gap[1] - gap[2]:+.1f} s", help="Positive = the 2-stop is faster"
        )
    p = plans.copy()
    p["FirstStop"] = p["PitLaps"].map(lambda x: x[0])
    curve = p.groupby(["Stops", "FirstStop"])["Gap"].min().reset_index()
    curve["Plan"] = curve["Stops"].map({1: "1-stop", 2: "2-stop"})
    fig = px.line(
        curve,
        x="FirstStop",
        y="Gap",
        color="Plan",
        color_discrete_map={"1-stop": PALETTE[0], "2-stop": PALETTE[1]},
        labels={"FirstStop": "Lap of the first stop", "Gap": "Slower than optimum (s)"},
    )
    st.plotly_chart(tidy(fig, 380), use_container_width=True)
    st.dataframe(
        best[["Stops", "Sequence", "PitLaps", "StintLengths", "Gap"]].round(1),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Pre-race plan from this race's fitted tyre model; safety cars are not simulated.")

# ---------------------------------------------------------------------------------------
with tab_tel:
    tel = data["telemetry"]
    races = tel.drop_duplicates(["Season", "EventName"])
    labels = [f"{r.Season} {r.EventName}" for r in races.itertuples()]
    choice = st.selectbox("Race", labels)
    season, event = int(choice[:4]), choice[5:]
    t = tel[(tel["Season"] == season) & (tel["EventName"] == event)]
    drivers = t["Driver"].unique().tolist()
    channel = st.radio("Channel", ["Speed", "Throttle", "Brake", "Gear"], horizontal=True)
    fig = px.line(
        t,
        x="Distance",
        y=channel,
        color="Driver",
        color_discrete_sequence=PALETTE[:2],
        labels={"Distance": "Distance along the lap (m)"},
    )
    st.plotly_chart(tidy(fig, 380), use_container_width=True)
    if t["X"].notna().any():
        a, b = (t[t["Driver"] == d].sort_values("Distance") for d in drivers[:2])
        edges = np.linspace(0, min(a["Distance"].max(), b["Distance"].max()), 26)
        ta = np.diff(np.interp(edges, a["Distance"], a["Time"]))
        tb = np.diff(np.interp(edges, b["Distance"], b["Time"]))
        seg = np.clip(np.searchsorted(edges, a["Distance"], side="right") - 1, 0, 24)
        a = a.assign(Faster=np.where(ta[seg] < tb[seg], drivers[0], drivers[1]))
        fig = px.scatter(
            a,
            x="X",
            y="Y",
            color="Faster",
            color_discrete_sequence=PALETTE[:2],
            title="Faster driver per mini-sector",
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_yaxes(scaleanchor="x", visible=False)
        fig.update_xaxes(visible=False)
        st.plotly_chart(tidy(fig, 460), use_container_width=True)

# ---------------------------------------------------------------------------------------
with tab_model:
    st.dataframe(data["metrics"].round(3), hide_index=True, use_container_width=True)
    imp = data["importance"].head(12).iloc[::-1]
    fig = px.bar(
        imp,
        x="importance",
        y="label",
        orientation="h",
        color_discrete_sequence=[PALETTE[0]],
        labels={"importance": "Share of split gain", "label": ""},
        title="What moves a lap time (LightGBM, gain)",
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(tidy(fig, 440), use_container_width=True)
