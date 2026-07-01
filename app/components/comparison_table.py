"""
Comparison Table Component

Builds and renders predicted-vs-actual finishing position tables, shared by
the Predictions and Historical Analysis pages so both stay consistent.
"""

import pandas as pd
import streamlit as st
from typing import List, Dict

from components.prediction_card import confidence_badge


def build_comparison_dataframe(predictions: List[Dict]) -> pd.DataFrame:
    """
    Turn a list of prediction dicts (as returned by src.predict.predict_race)
    into a DataFrame with predicted position, actual position, and absolute
    error columns.

    Args:
        predictions: List of prediction dicts, sorted by predicted_position

    Returns:
        pd.DataFrame: One row per driver
    """
    rows = []
    has_actuals = all(p.get("actual_position") is not None for p in predictions)

    for p in predictions:
        row = {
            "Driver": p["driver_name"],
            "Starting Position": p["starting_position"],
            "Predicted Position": int(p["predicted_position"]),
            "Confidence": p["confidence"],
        }
        if has_actuals:
            row["Actual Position"] = int(p["actual_position"])
            row["Error"] = round(abs(p["predicted_position"] - p["actual_position"]), 1)
        rows.append(row)

    return pd.DataFrame(rows)


def _error_background(error: float) -> str:
    """Green for a close prediction, amber for a middling one, red for a miss."""
    if pd.isna(error):
        return ""
    if error <= 1:
        color = "rgba(26, 255, 26, 0.25)"
    elif error <= 2:
        color = "rgba(255, 171, 0, 0.25)"
    else:
        color = "rgba(255, 107, 107, 0.25)"
    return f"background-color: {color}"


def render_comparison_table(predictions: List[Dict]) -> pd.DataFrame:
    """
    Render a predicted-vs-actual table with a Streamlit dataframe widget.
    Rows with actual results are color-coded by error size (green = close,
    red = big miss) so accuracy is visible at a glance.

    Args:
        predictions: List of prediction dicts

    Returns:
        pd.DataFrame: The dataframe that was rendered (for reuse, e.g. metrics)
    """
    df = build_comparison_dataframe(predictions)
    display_df = df.copy()
    if "Confidence" in display_df.columns:
        display_df["Confidence"] = display_df["Confidence"].apply(confidence_badge)

    if "Error" in display_df.columns:
        styled = (
            display_df.style
            .applymap(_error_background, subset=["Error"])
            .format({"Error": "{:.1f}"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    return df


def render_accuracy_metrics(df: pd.DataFrame, mae_label: str = "Mean Absolute Error",
                             within2_label: str = "Within 2 Positions") -> None:
    """
    Render MAE / within-2-positions metrics from a comparison dataframe that
    has an "Error" column (i.e. actual positions were available).

    Args:
        df: DataFrame from build_comparison_dataframe with an "Error" column
        mae_label: Metric label for mean absolute error
        within2_label: Metric label for within-2-positions accuracy
    """
    if "Error" not in df.columns or df.empty:
        return

    mae = df["Error"].mean()
    within_2 = (df["Error"] <= 2).mean() * 100

    col1, col2 = st.columns(2)
    col1.metric(mae_label, f"{mae:.2f} positions")
    col2.metric(within2_label, f"{within_2:.0f}%")
