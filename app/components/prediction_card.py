"""
Prediction Card Component

Renders a single driver's prediction as a compact card: predicted position,
confidence, actual result (if known), and the human-readable explanation.
"""

from typing import Dict

import streamlit as st


def confidence_badge(confidence: float) -> str:
    """
    Turn the raw confidence score into a High/Medium/Low badge. The
    underlying score (src.explainer's SHAP-variance-based heuristic) isn't
    a calibrated probability, so presenting it as a precise percentage
    implies more precision than it has - a coarse badge is more honest.
    Thresholds are calibrated from the observed spread of confidence scores
    across real 2023-2024 predictions (roughly 0.57-0.91).

    Args:
        confidence: Raw confidence score (0-1)

    Returns:
        str: e.g. "🟢 High"
    """
    if confidence >= 0.80:
        return "🟢 High"
    if confidence >= 0.65:
        return "🟡 Medium"
    return "🔴 Low"


def _constructor_badge_html(constructor_name: str, constructor_colors: Dict[str, str] = None,
                             default_color: str = "#888888") -> str:
    """Small colored pill showing the constructor name in its team color."""
    if not constructor_name:
        return ""
    color = (constructor_colors or {}).get(constructor_name, default_color)
    return (
        f'<span style="background: {color}; color: white; padding: 1px 8px; '
        f'border-radius: 10px; font-size: 0.75rem;">{constructor_name}</span>'
    )


def render_prediction_card(prediction: Dict, constructor_colors: Dict[str, str] = None,
                            default_constructor_color: str = "#888888") -> None:
    """
    Render one driver's prediction as a bordered card.

    Args:
        prediction: A single prediction dict from src.predict.predict_race
        constructor_colors: Optional {constructor_name: hex color} lookup
        default_constructor_color: Fallback color for unmapped constructors
    """
    with st.container(border=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**{prediction['driver_name']}**")
            st.markdown(
                _constructor_badge_html(
                    prediction.get("constructor_name"), constructor_colors, default_constructor_color
                ),
                unsafe_allow_html=True,
            )
            st.caption(prediction["explanation"])

        with col2:
            st.metric("Predicted", f"P{int(prediction['predicted_position'])}")
            if prediction.get("actual_position") is not None:
                st.caption(f"Actual: P{int(prediction['actual_position'])}")
            st.caption(f"Confidence: {confidence_badge(prediction['confidence'])}")


def render_podium_cards(predictions: list, constructor_colors: Dict[str, str] = None,
                         default_constructor_color: str = "#888888") -> None:
    """
    Render the top 3 predicted finishers as side-by-side cards.

    Args:
        predictions: List of prediction dicts, sorted by predicted_position
        constructor_colors: Optional {constructor_name: hex color} lookup
        default_constructor_color: Fallback color for unmapped constructors
    """
    podium = predictions[:3]
    cols = st.columns(len(podium)) if podium else []
    medals = ["🥇", "🥈", "🥉"]

    for col, medal, prediction in zip(cols, medals, podium):
        with col:
            st.markdown(f"### {medal} {prediction['driver_name']}")
            st.markdown(
                _constructor_badge_html(
                    prediction.get("constructor_name"), constructor_colors, default_constructor_color
                ),
                unsafe_allow_html=True,
            )
            st.metric("Predicted", f"P{int(prediction['predicted_position'])}")
            if prediction.get("actual_position") is not None:
                st.caption(f"Actual: P{int(prediction['actual_position'])}")
            st.caption(f"Confidence: {confidence_badge(prediction['confidence'])}")
