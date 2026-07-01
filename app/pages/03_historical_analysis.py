"""
Historical Analysis Page

Shows overall model accuracy and lets the user browse a season race-by-race
to compare predicted vs actual finishing positions.
"""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import get_available_races, predict_race
from components.comparison_table import render_comparison_table, render_accuracy_metrics

st.set_page_config(page_title="Historical Analysis - F1 Race Predictor", page_icon="📊", layout="wide")

st.title("📊 Historical Analysis")


@st.cache_data
def load_model_metrics():
    with open("models/model_config.json") as f:
        return json.load(f)


@st.cache_data
def load_races():
    return get_available_races()


@st.cache_data
def load_season_predictions(race_ids: tuple):
    rows = []
    for race_id in race_ids:
        for p in predict_race(race_id):
            rows.append({**p, "race_id": race_id})
    return pd.DataFrame(rows)


@st.cache_data
def load_leaderboard(years: tuple):
    races_all = load_races()
    subset = races_all[races_all["year"].isin(years)]
    rows = []
    for _, race in subset.iterrows():
        race_id = int(race["raceId"])
        for p in predict_race(race_id):
            rows.append({**p, "race_id": race_id, "year": int(race["year"])})
    return pd.DataFrame(rows)


config = load_model_metrics()
perf = config["performance"]["test"]
races = load_races()

st.subheader("Overall Model Performance (Test Set)")
st.caption(
    f"Trained on {config['training_data']['train_set']}, "
    f"validated on {config['training_data']['val_set']}, "
    f"tested on {config['training_data']['test_set']}"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{perf['mae']:.2f} positions")
col2.metric("Within 1 Position", f"{perf['within_1_position']:.1f}%")
col3.metric("Within 2 Positions", f"{perf['within_2_positions']:.1f}%")
col4.metric("Within 3 Positions", f"{perf['within_3_positions']:.1f}%")

st.markdown("---")
st.subheader("Accuracy Across Seasons")
st.caption(
    f"{config['training_data']['train_set']} was used to **train** the model, so accuracy there is "
    f"expected to look better than real-world performance. {config['training_data']['val_set']} "
    f"(validation) and {config['training_data']['test_set']} (test) are the honest, held-out numbers."
)

leaderboard_years = tuple(sorted(y for y in races["year"].unique() if y >= 2020))
with st.spinner("Computing accuracy across seasons..."):
    leaderboard_df = load_leaderboard(leaderboard_years)

leaderboard_df["abs_error"] = (leaderboard_df["predicted_position"] - leaderboard_df["actual_position"]).abs()
season_summary = (
    leaderboard_df.groupby("year")
    .agg(MAE=("abs_error", "mean"), **{"Within 2 Positions (%)": ("abs_error", lambda s: (s <= 2).mean() * 100)})
    .reset_index()
)

lb_col1, lb_col2 = st.columns(2)
with lb_col1:
    st.markdown("**MAE by season**")
    st.bar_chart(season_summary.set_index("year")["MAE"])
with lb_col2:
    st.markdown("**Within 2 Positions (%) by season**")
    st.bar_chart(season_summary.set_index("year")["Within 2 Positions (%)"])

st.markdown("---")
st.subheader("Season Browser")

years = sorted(races["year"].unique(), reverse=True)
selected_year = st.selectbox("Season", years)

season_races = races[races["year"] == selected_year].sort_values("round")

if selected_year in leaderboard_years:
    # Already computed above for the cross-season leaderboard - avoid
    # redoing the (relatively expensive) prediction pass for this season.
    df = leaderboard_df[leaderboard_df["year"] == selected_year].copy()
else:
    with st.spinner(f"Loading predictions for {selected_year}..."):
        df = load_season_predictions(tuple(season_races["raceId"].tolist()))

if df.empty or df["actual_position"].isna().all():
    st.info("No actual results available yet for this season.")
else:
    df["abs_error"] = (df["predicted_position"] - df["actual_position"]).abs()

    render_accuracy_metrics(
        df.rename(columns={"abs_error": "Error"}),
        mae_label=f"{selected_year} MAE",
        within2_label=f"{selected_year} Within 2 Positions",
    )

    round_to_name = season_races.set_index("raceId")["name"]
    round_to_round = season_races.set_index("raceId")["round"]
    df["race_name"] = df["race_id"].map(round_to_name)
    df["round"] = df["race_id"].map(round_to_round)

    trend = (
        df.groupby(["round", "race_name"])["abs_error"]
        .mean()
        .reset_index()
        .sort_values("round")
    )
    st.markdown("#### Prediction Error by Race")
    st.line_chart(trend.set_index("race_name")["abs_error"])

    st.markdown("#### Race Detail")
    race_labels = [f"Round {row['round']} - {row['name']}" for _, row in season_races.iterrows()]
    selected_label = st.selectbox("Select a race", race_labels)
    selected_race = season_races.iloc[race_labels.index(selected_label)]

    race_predictions = (
        df[df["race_id"] == selected_race["raceId"]]
        .sort_values("predicted_position")
        .to_dict("records")
    )
    render_comparison_table(race_predictions)
