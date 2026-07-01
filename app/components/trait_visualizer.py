"""
Trait Visualizer Component

Reusable charts for driver FM-style traits: a radar chart of all 6 scores,
progress bars per trait, and a bar chart of a specific prediction's
trait-influence breakdown (from src.explainer).
"""

from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st
import pandas as pd


def render_trait_radar(driver_name: str, driver_traits: Dict[str, float], trait_names: Dict[str, str]) -> None:
    """
    Render a radar/spider chart of a driver's 6 trait scores.

    Args:
        driver_name: Display name for the chart trace
        driver_traits: {trait_key: score} for one driver
        trait_names: {trait_key: display_name} lookup for labels
    """
    categories = [trait_names.get(t, t) for t in driver_traits.keys()]
    values = list(driver_traits.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=driver_name,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trait_bars(driver_traits: Dict[str, float], trait_names: Dict[str, str]) -> None:
    """
    Render one progress bar per trait score.

    Args:
        driver_traits: {trait_key: score} for one driver
        trait_names: {trait_key: display_name} lookup for labels
    """
    for trait, score in driver_traits.items():
        st.write(trait_names.get(trait, trait))
        st.progress(score, text=f"{score:.2f}")


def render_trait_influence_chart(trait_influences: List[Dict]) -> None:
    """
    Render a bar chart of how much each trait pushed a specific prediction
    up or down, as produced by src.explainer.calculate_trait_influence.

    Args:
        trait_influences: List of {"trait", "score", "impact", "direction"} dicts
    """
    if not trait_influences:
        st.info("No trait influence data available.")
        return

    df = pd.DataFrame(trait_influences).set_index("trait")
    st.bar_chart(df["impact"])
