"""
Race Predictions Page

Lets the user pick a race and see predicted finishing order,
confidence, trait-based explanation, and (for past races) the actual result.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.predict import get_available_races, predict_race

st.set_page_config(page_title="Predictions - F1 Race Predictor", page_icon="🎯", layout="wide")

st.title("🎯 Race Predictions")


@st.cache_data
def load_races():
    return get_available_races()


@st.cache_data
def load_predictions(race_id: int):
    return predict_race(race_id)


races = load_races()

years = sorted(races['year'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("Season", years)

season_races = races[races['year'] == selected_year]
race_labels = [f"Round {row['round']} - {row['name']}" for _, row in season_races.iterrows()]
selected_label = st.sidebar.selectbox("Race", race_labels)
selected_race = season_races.iloc[race_labels.index(selected_label)]
race_id = int(selected_race['raceId'])

st.subheader(f"{selected_race['name']} ({selected_year})")

predictions = load_predictions(race_id)

if not predictions:
    st.warning("No prediction data available for this race.")
else:
    has_actuals = all(p['actual_position'] is not None for p in predictions)

    rows = []
    for rank, p in enumerate(predictions, start=1):
        row = {
            "Predicted Rank": rank,
            "Driver": p['driver_name'],
            "Predicted Position": round(p['predicted_position'], 1),
            "Confidence": f"{p['confidence']:.0%}",
        }
        if has_actuals:
            row["Actual Position"] = int(p['actual_position'])
            row["Error"] = round(abs(p['predicted_position'] - p['actual_position']), 1)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if has_actuals:
        mae = sum(abs(p['predicted_position'] - p['actual_position']) for p in predictions) / len(predictions)
        within_2 = sum(1 for p in predictions if abs(p['predicted_position'] - p['actual_position']) <= 2) / len(predictions)
        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error", f"{mae:.2f} positions")
        col2.metric("Within 2 Positions", f"{within_2:.0%}")

    st.markdown("---")
    st.subheader("Driver Explanations")

    driver_names = [p['driver_name'] for p in predictions]
    selected_driver = st.selectbox("Select a driver for a detailed breakdown", driver_names)
    detail = next(p for p in predictions if p['driver_name'] == selected_driver)

    st.write(f"**{detail['explanation']}**")

    trait_df = pd.DataFrame(detail['trait_influences'])
    if not trait_df.empty:
        st.bar_chart(trait_df.set_index('trait')['impact'])
