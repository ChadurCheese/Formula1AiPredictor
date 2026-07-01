"""
Streamlit Configuration

Global app settings and constants.
"""

import json

import streamlit as st

# App Configuration
APP_TITLE = "🏎️ Formula 1 Race Predictor"
APP_ICON = "🏎️"

# Page Configuration
RACES_TO_LOAD = 50  # Historical races to cache
MODEL_PATH = "models/model.pkl"
MODEL_CONFIG_PATH = "models/model_config.json"
TRAITS_CACHE = "data/cache/driver_traits.json"
FEATURES_CACHE = "data/processed/features.pkl"
PREDICTIONS_CACHE = "data/cache/predictions_cache.pkl"

# UI Settings
THEME = "light"
LAYOUT = "wide"

# Trait Display Names
TRAIT_NAMES = {
    "qualifying_specialist": "🎯 Qualifying Specialist",
    "wet_weather_master": "🌧️ Wet Weather Master",
    "strong_starter": "⚡ Strong Starter",
    "tire_management": "🛞 Tire Management",
    "inconsistent": "⚠️ Inconsistent",
    "track_expert": "🏁 Track Expert",
}

# Color scheme for traits
TRAIT_COLORS = {
    "qualifying_specialist": "#00D9FF",  # Cyan
    "wet_weather_master": "#0084FF",     # Blue
    "strong_starter": "#FFAB00",         # Amber
    "tire_management": "#1AFF1A",        # Green
    "inconsistent": "#FF6B6B",           # Red
    "track_expert": "#FF00FF",           # Magenta
}

# Approximate current-era team colors (2020-2024 constructor names as they
# appear in constructors.csv). The Kaggle dataset has no official color
# data and constructor names change across eras, so this only covers the
# recent grid; anything else falls back to DEFAULT_CONSTRUCTOR_COLOR.
DEFAULT_CONSTRUCTOR_COLOR = "#888888"
CONSTRUCTOR_COLORS = {
    "Red Bull": "#3671C6",
    "Ferrari": "#E8002D",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine F1 Team": "#FF87BC",
    "Williams": "#64C4FF",
    "AlphaTauri": "#5E8FAA",
    "RB F1 Team": "#6692FF",
    "Alfa Romeo": "#C92D4B",
    "Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
}


@st.cache_data
def get_configuration() -> dict:
    """Get cached configuration dictionary."""
    return {
        "model_path": MODEL_PATH,
        "traits_cache": TRAITS_CACHE,
        "features_cache": FEATURES_CACHE,
    }


@st.cache_data
def get_accuracy_caption() -> str:
    """
    Short, honest summary of real model accuracy on the 2024 test set, meant
    to be shown near the top of every page so predictions aren't mistaken
    for more reliable than they are.
    """
    try:
        with open(MODEL_CONFIG_PATH) as f:
            config = json.load(f)
        perf = config["performance"]["test"]
        return (
            f"📏 Model accuracy (2024 test set): ~{perf['within_2_positions']:.0f}% of predictions "
            f"land within 2 positions of the actual result (MAE {perf['mae']:.1f} positions). "
            "Treat predictions as a rough guide, not a guarantee."
        )
    except (FileNotFoundError, KeyError):
        return "📏 Model accuracy metrics not available yet - train the model first (scripts/train_model.py)."
