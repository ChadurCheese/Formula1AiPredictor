"""
Streamlit Configuration

Global app settings and constants.
"""

import streamlit as st

# App Configuration
APP_TITLE = "🏎️ Formula 1 Race Predictor"
APP_ICON = "🏎️"

# Page Configuration
RACES_TO_LOAD = 50  # Historical races to cache
MODEL_PATH = "models/model.pkl"
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


@st.cache_data
def get_configuration() -> dict:
    """Get cached configuration dictionary."""
    return {
        "model_path": MODEL_PATH,
        "traits_cache": TRAITS_CACHE,
        "features_cache": FEATURES_CACHE,
    }
