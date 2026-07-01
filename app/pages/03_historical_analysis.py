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

from src.predict import get_available_races, predict_race

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


config = load_model_metrics()
perf = config["performance"]["test"]

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
st.subheader("Season Browser")

races = load_races()
years = sorted(races["year"].unique(), reverse=True)
selected_year = st.selectbox("Season", years)

season_races = races[races["year"] == selected_year].sort_values("round")

with st.spinner(f"Loading predictions for {selected_year}..."):
    df = load_season_predictions(tuple(season_races["raceId"].tolist()))

if df.empty or df["actual_position"].isna().all():
    st.info("No actual results available yet for this season.")
else:
    df["abs_error"] = (df["predicted_position"] - df["actual_position"]).abs()

    season_mae = df["abs_error"].mean()
    season_within2 = (df["abs_error"] <= 2).mean() * 100

    c1, c2 = st.columns(2)
    c1.metric(f"{selected_year} MAE", f"{season_mae:.2f} positions")
    c2.metric(f"{selected_year} Within 2 Positions", f"{season_within2:.0f}%")

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

    race_detail = df[df["race_id"] == selected_race["raceId"]].sort_values("predicted_position")
    display_df = race_detail[["driver_name", "predicted_position", "actual_position", "abs_error"]].rename(
        columns={
            "driver_name": "Driver",
            "predicted_position": "Predicted",
            "actual_position": "Actual",
            "abs_error": "Error",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
