"""
Main Streamlit Application

Multi-page web UI for F1 prediction system.
Entry point: streamlit run app.py
"""

import streamlit as st
import logging

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
    """Main app entry point."""
    
    st.title("🏎️ Formula 1 Race Predictor")
    
    st.markdown("""
    Predict F1 race results with explainable driver traits.
    
    **Features:**
    - Race position predictions based on historical data
    - Driver trait analysis (FM-game style)
    - Prediction explanations with trait breakdowns
    - Historical race analysis
    """)
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🎯 Predictions", "👤 Driver Traits", "📊 Historical Analysis"],
        label_visibility="collapsed"
    )
    
    # TODO: Route to different pages based on selection
    if page == "🏠 Home":
        show_home()
    elif page == "🎯 Predictions":
        st.info("Predictions page - to be implemented")
    elif page == "👤 Driver Traits":
        st.info("Driver Traits page - to be implemented")
    elif page == "📊 Historical Analysis":
        st.info("Historical Analysis page - to be implemented")


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
    st.write("1. Go to **Predictions** to see race forecasts")
    st.write("2. Check **Driver Traits** for player profiles")
    st.write("3. Review **Historical Analysis** for model performance")


if __name__ == "__main__":
    main()
