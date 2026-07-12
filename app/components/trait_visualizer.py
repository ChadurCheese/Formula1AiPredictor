"""
Trait Visualizer Component

Reusable charts for driver FM-style traits: a radar chart of all 6 scores
(with an average-driver reference trace), color-coded bars scored relative
to the population baseline, and a bar chart of a specific prediction's
trait-influence breakdown (from src.explainer).
"""

from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd

# Status palette (bad -> superb), reserved for "how good is this stat"
# encoding - never reused for arbitrary series identity elsewhere.
STATUS_COLORS = {
    "Bad": "#d03b3b",
    "Average": "#fab219",
    "Above Average": "#0ca30c",
    "Superb": "#2a78d6",
}

# Traits where a *higher* raw score means *worse* performance - the color
# banding below inverts these so the color always reads as "how good",
# not "how big the number is".
INVERTED_TRAITS = {"inconsistent"}

AVERAGE_COLOR = "#898781"  # muted ink - reference/baseline, not a real driver
DEFAULT_TRAIT_COLOR = "#4F8BF9"


def compute_trait_baselines(all_traits: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute each trait's population mean/std across all drivers, so a single
    score can be shown relative to a real baseline instead of a bare number.

    Args:
        all_traits: {driver_id: {trait_key: score}} for the whole population
            (or whatever subset should define "average" - e.g. active drivers)

    Returns:
        dict: {trait_key: {"mean": float, "std": float}}
    """
    if not all_traits:
        return {}

    df = pd.DataFrame.from_dict(all_traits, orient="index")
    baselines = {}
    for trait in df.columns:
        std = df[trait].std()
        baselines[trait] = {
            "mean": float(df[trait].mean()),
            # A degenerate (zero/NaN) std would make every score look like
            # an extreme outlier - fall back to a reasonable default spread.
            "std": float(std) if std and std > 0 else 0.15,
        }
    return baselines


def trait_status(trait_key: str, score: float, baseline: Optional[Dict[str, float]]) -> Tuple[str, str]:
    """
    Classify a trait score relative to the population baseline into one of
    four tiers, each with a fixed status color: Bad (red), Average (orange),
    Above Average (green), Superb (blue). Bands are +/-0.5 and +1.5 standard
    deviations from the mean - roughly bottom ~30%, middle ~40%, next ~25%,
    top ~5% for a normal-ish distribution.

    Args:
        trait_key: snake_case trait key (used to check INVERTED_TRAITS)
        score: This driver's raw score (0-1)
        baseline: {"mean": float, "std": float} from compute_trait_baselines,
            or None to fall back to a neutral 0.5/0.15 baseline

    Returns:
        (label, color): e.g. ("Above Average", "#0ca30c")
    """
    baseline = baseline or {"mean": 0.5, "std": 0.15}
    mean = baseline.get("mean", 0.5)
    std = baseline.get("std") or 0.15

    z = (score - mean) / std
    if trait_key in INVERTED_TRAITS:
        z = -z

    if z < -0.5:
        label = "Bad"
    elif z < 0.5:
        label = "Average"
    elif z < 1.5:
        label = "Above Average"
    else:
        label = "Superb"

    return label, STATUS_COLORS[label]


def render_status_legend() -> None:
    """Small legend explaining the bad->superb color tiers (status colors are
    never allowed to carry meaning by color alone)."""
    swatches = "".join(
        f'<span style="display: inline-flex; align-items: center; margin-right: 16px;">'
        f'<span style="width: 10px; height: 10px; border-radius: 2px; background: {color}; '
        f'display: inline-block; margin-right: 6px;"></span>{label}</span>'
        for label, color in STATUS_COLORS.items()
    )
    st.markdown(
        f'<div style="font-size: 0.8rem; opacity: 0.85; margin-bottom: 8px;">{swatches}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Colors show each trait relative to the average driver, not an absolute scale.")


def render_trait_radar(driver_name: str, driver_traits: Dict[str, float], trait_names: Dict[str, str],
                        average_traits: Dict[str, float] = None) -> None:
    """
    Render a radar/spider chart of a driver's 6 trait scores, with an
    optional grey reference trace showing the population average so a lone
    number like "0.4" has something to be read against.

    Args:
        driver_name: Display name for the chart trace
        driver_traits: {trait_key: score} for one driver
        trait_names: {trait_key: display_name} lookup for labels
        average_traits: Optional {trait_key: mean_score} to overlay as a
            grey "Average" reference trace
    """
    categories = [trait_names.get(t, t) for t in driver_traits.keys()]
    values = list(driver_traits.values())

    fig = go.Figure()

    if average_traits:
        avg_values = [average_traits.get(t, 0.5) for t in driver_traits.keys()]
        fig.add_trace(go.Scatterpolar(
            r=avg_values + [avg_values[0]],
            theta=categories + [categories[0]],
            name="Average",
            line=dict(color=AVERAGE_COLOR, dash="dot"),
            fill=None,
        ))

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=driver_name,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=bool(average_traits),
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_multi_driver_radar(drivers_traits: Dict[str, Dict[str, float]], trait_names: Dict[str, str],
                               average_traits: Dict[str, float] = None) -> None:
    """
    Overlay multiple drivers' trait radars on one chart for easy comparison,
    optionally with a grey "Average" reference trace.

    Args:
        drivers_traits: {driver_name: {trait_key: score}} for each selected driver
        trait_names: {trait_key: display_name} lookup for labels
        average_traits: Optional {trait_key: mean_score} to overlay as a
            grey "Average" reference trace
    """
    if not drivers_traits:
        st.info("Select at least one driver to compare.")
        return

    palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    if average_traits:
        first_traits = next(iter(drivers_traits.values()))
        categories = [trait_names.get(t, t) for t in first_traits.keys()]
        avg_values = [average_traits.get(t, 0.5) for t in first_traits.keys()]
        fig.add_trace(go.Scatterpolar(
            r=avg_values + [avg_values[0]],
            theta=categories + [categories[0]],
            name="Average",
            line=dict(color=AVERAGE_COLOR, dash="dot"),
            fill=None,
        ))

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


def render_trait_bars(driver_traits: Dict[str, float], trait_names: Dict[str, str],
                       baselines: Dict[str, Dict[str, float]] = None) -> None:
    """
    Render one bar per trait score, color-coded by how it compares to the
    population average (red=bad, orange=average, green=above average,
    blue=superb) rather than a fixed per-trait color - a raw score like 0.4
    is otherwise meaningless without something to compare it to.

    Args:
        driver_traits: {trait_key: score} for one driver
        trait_names: {trait_key: display_name} lookup for labels
        baselines: {trait_key: {"mean": float, "std": float}} from
            compute_trait_baselines(); falls back to a neutral baseline if
            not provided
    """
    for trait, score in driver_traits.items():
        label = trait_names.get(trait, trait)
        status_label, color = trait_status(trait, score, (baselines or {}).get(trait))
        pct = max(0, min(100, round(score * 100)))
        st.markdown(
            f"""
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                    <span>{label}</span><span>{score:.2f} &middot; {status_label}</span>
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
