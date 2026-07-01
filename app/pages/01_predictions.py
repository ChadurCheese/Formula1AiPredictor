"""
Race Predictions Page

Lets the user pick a race and see predicted finishing order,
confidence, trait-based explanation, and (for past races) the actual result.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import get_available_races, predict_race
from components.comparison_table import render_comparison_table, render_accuracy_metrics
from components.trait_visualizer import render_trait_influence_chart
from components.prediction_card import render_podium_cards
from config import TRAIT_COLORS, CONSTRUCTOR_COLORS, DEFAULT_CONSTRUCTOR_COLOR, get_accuracy_caption

st.set_page_config(page_title="Predictions - F1 Race Predictor", page_icon="🎯", layout="wide")

st.title("🎯 Race Predictions")
st.caption(get_accuracy_caption())


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
    render_podium_cards(predictions, CONSTRUCTOR_COLORS, DEFAULT_CONSTRUCTOR_COLOR)

    st.markdown("---")
    comparison_df = render_comparison_table(predictions)
    render_accuracy_metrics(comparison_df)

    st.markdown("---")
    st.subheader("Driver Explanations")

    driver_names = [p['driver_name'] for p in predictions]
    selected_driver = st.selectbox("Select a driver for a detailed breakdown", driver_names)
    detail = next(p for p in predictions if p['driver_name'] == selected_driver)

    st.write(f"**{detail['explanation']}**")
    render_trait_influence_chart(detail['trait_influences'], TRAIT_COLORS)
