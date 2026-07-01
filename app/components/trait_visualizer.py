"""
Trait Visualizer Component

Reusable charts for driver FM-style traits: a radar chart of all 6 scores,
progress bars per trait, and a bar chart of a specific prediction's
trait-influence breakdown (from src.explainer).
"""

from typing import Dict, List

import plotly.graph_objects as go
import plotly.express as px
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


def render_multi_driver_radar(drivers_traits: Dict[str, Dict[str, float]], trait_names: Dict[str, str]) -> None:
    """
    Overlay multiple drivers' trait radars on one chart for easy comparison -
    much faster to eyeball than a wide table of numbers.

    Args:
        drivers_traits: {driver_name: {trait_key: score}} for each selected driver
        trait_names: {trait_key: display_name} lookup for labels
    """
    if not drivers_traits:
        st.info("Select at least one driver to compare.")
        return

    palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    for i, (driver_name, traits) in enumerate(drivers_traits.items()):
        categories = [trait_names.get(t, t) for t in traits.keys()]
        values = list(traits.values())
        color = palette[i % len(palette)]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=driver_name,
            line=dict(color=color),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


DEFAULT_TRAIT_COLOR = "#4F8BF9"


def render_trait_bars(driver_traits: Dict[str, float], trait_names: Dict[str, str],
                       trait_colors: Dict[str, str] = None) -> None:
    """
    Render one color-coded bar per trait score. Uses a small custom HTML bar
    instead of st.progress since Streamlit's built-in progress bar doesn't
    support per-widget colors.

    Args:
        driver_traits: {trait_key: score} for one driver
        trait_names: {trait_key: display_name} lookup for labels
        trait_colors: Optional {trait_key: hex color} lookup
    """
    trait_colors = trait_colors or {}
    for trait, score in driver_traits.items():
        label = trait_names.get(trait, trait)
        color = trait_colors.get(trait, DEFAULT_TRAIT_COLOR)
        pct = max(0, min(100, round(score * 100)))
        st.markdown(
            f"""
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                    <span>{label}</span><span>{score:.2f}</span>
                </div>
                <div style="background: rgba(128,128,128,0.25); border-radius: 4px; height: 10px; width: 100%;">
                    <div style="background: {color}; width: {pct}%; height: 10px; border-radius: 4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_trait_influence_chart(trait_influences: List[Dict], trait_colors: Dict[str, str] = None) -> None:
    """
    Render a bar chart of how much each trait pushed a specific prediction
    up or down, as produced by src.explainer.calculate_trait_influence.

    Args:
        trait_influences: List of {"trait", "score", "impact", "direction"} dicts
        trait_colors: Optional {trait_key: hex color} lookup (keys are
            snake_case, e.g. "qualifying_specialist"; trait_influences uses
            title-case display names, so they're matched by normalizing)
    """
    if not trait_influences:
        st.info("No trait influence data available.")
        return

    trait_colors = trait_colors or {}
    df = pd.DataFrame(trait_influences)
    colors = [
        trait_colors.get(name.lower().replace(" ", "_"), DEFAULT_TRAIT_COLOR)
        for name in df["trait"]
    ]

    fig = go.Figure(go.Bar(x=df["trait"], y=df["impact"], marker_color=colors))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Impact on prediction")
    st.plotly_chart(fig, use_container_width=True)
