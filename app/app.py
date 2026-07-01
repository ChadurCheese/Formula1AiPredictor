"""
Main Streamlit Application

Multi-page web UI for F1 prediction system.
Entry point: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
import logging

sys.path.insert(0, str(Path(__file__).parent))
from config import get_accuracy_caption

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Streamlit page configuration
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main app entry point. Use the sidebar to navigate to other pages."""

    st.title("🏎️ Formula 1 Race Predictor")
    st.caption(get_accuracy_caption())

    st.markdown("""
    Predict F1 race results with explainable driver traits.

    **Features:**
    - Race position predictions based on historical data
    - Driver trait analysis (FM-game style)
    - Prediction explanations with trait breakdowns
    - Historical race analysis
    """)

    show_home()


def show_home():
    """Display home page."""
    
    st.subheader("Welcome!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 About")
        st.write("""
        This AI system predicts Formula 1 race results by analyzing:
        - Driver historical performance
        - Constructor/team strength
        - Track characteristics
        - Weather patterns
        - Driver traits (qualifying, wet weather, tire management, etc.)
        """)
    
    with col2:
        st.markdown("### 🔧 How It Works")
        st.write("""
        1. **Data**: Historical F1 data (2020-2024)
        2. **Features**: Driver stats, form, track experience
        3. **Model**: XGBoost with SHAP explanations
        4. **Traits**: FM-style player characteristics
        5. **Predictions**: Position + confidence + explanation
        """)
    
    st.markdown("---")

    st.markdown("### 🚀 Quick Start")
    render_page_links()


def render_page_links() -> None:
    """
    Direct links into the other pages. Uses plain HTML anchors instead of
    st.page_link since that API requires Streamlit >= 1.31 (this project
    pins 1.28 to avoid a numpy/shap version conflict - see docs/DATA_SOURCES.md).
    """
    links = [
        ("/predictions", "🎯", "Predictions", "See race forecasts"),
        ("/driver_traits", "👤", "Driver Traits", "Browse player-style profiles"),
        ("/historical_analysis", "📊", "Historical Analysis", "Review model performance"),
    ]
    cols = st.columns(len(links))
    for col, (href, icon, label, description) in zip(cols, links):
        with col:
            st.markdown(
                f"""
                <a href="{href}" target="_self" style="text-decoration: none;">
                    <div style="border: 1px solid rgba(128,128,128,0.3); border-radius: 8px;
                                padding: 16px; text-align: center;">
                        <div style="font-size: 1.8rem;">{icon}</div>
                        <div style="font-weight: 600; margin-top: 4px;">{label}</div>
                        <div style="font-size: 0.85rem; opacity: 0.7;">{description}</div>
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
