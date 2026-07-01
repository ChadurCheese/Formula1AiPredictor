"""
Prediction Card Component

Renders a single driver's prediction as a compact card: predicted position,
confidence, actual result (if known), and the human-readable explanation.
"""

from typing import Dict

import streamlit as st


def render_prediction_card(prediction: Dict) -> None:
    """
    Render one driver's prediction as a bordered card.

    Args:
        prediction: A single prediction dict from src.predict.predict_race
    """
    with st.container(border=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**{prediction['driver_name']}**")
            st.caption(prediction["explanation"])

        with col2:
            st.metric("Predicted", f"P{prediction['predicted_position']:.1f}")
            if prediction.get("actual_position") is not None:
                st.caption(f"Actual: P{int(prediction['actual_position'])}")
            st.caption(f"Confidence: {prediction['confidence']:.0%}")


def render_podium_cards(predictions: list) -> None:
    """
    Render the top 3 predicted finishers as side-by-side cards.

    Args:
        predictions: List of prediction dicts, sorted by predicted_position
    """
    podium = predictions[:3]
    cols = st.columns(len(podium)) if podium else []
    medals = ["🥇", "🥈", "🥉"]

    for col, medal, prediction in zip(cols, medals, podium):
        with col:
            st.markdown(f"### {medal} {prediction['driver_name']}")
            st.metric("Predicted", f"P{prediction['predicted_position']:.1f}")
            if prediction.get("actual_position") is not None:
                st.caption(f"Actual: P{int(prediction['actual_position'])}")
            st.caption(f"Confidence: {prediction['confidence']:.0%}")
