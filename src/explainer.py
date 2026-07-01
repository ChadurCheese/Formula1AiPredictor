"""
Explainability Module

Generates human-readable explanations for model predictions using SHAP
and trait-based reasoning.

Functions:
    - explain_prediction(): Generate full explanation with traits
    - calculate_trait_influence(): How traits affect prediction
    - format_explanation(): Pretty-print explanation
"""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def explain_prediction(
    model: xgb.XGBRegressor,
    X_row: pd.Series,
    driver_traits: Dict,
    driver_id: int,
    feature_names: List[str] = None,
    explainer: "shap.TreeExplainer" = None,
) -> Dict:
    """
    Generate explanation for a single prediction.

    Combines:
        - Base prediction (finishing position)
        - SHAP feature importance
        - Driver trait influences
        - Confidence score

    Args:
        model: Trained XGBoost model
        X_row: Single sample feature vector
        driver_traits: Driver traits dictionary
        driver_id: Driver ID for trait lookup
        feature_names: List of feature names
        explainer: Optional pre-built shap.TreeExplainer for this model.
            Building a TreeExplainer is relatively expensive, so callers
            generating many explanations (e.g. a whole race or season)
            should build one once and pass it in rather than let this
            function build a fresh one on every call.

    Returns:
        dict: Complete explanation including:
            - predicted_position: Base prediction
            - confidence: Confidence score (0-1)
            - feature_importance: Top contributing features
            - trait_influences: How traits affect prediction
            - explanation_text: Human-readable summary
    """
    # Get base prediction
    base_pred = model.predict(X_row.values.reshape(1, -1))[0]

    # Get SHAP explanations
    if explainer is None:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row.values.reshape(1, -1))

    # Get feature importances
    feature_contrib = shap_values[0]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(feature_contrib))]
    
    # Sort by absolute contribution
    top_features = sorted(
        zip(feature_names, feature_contrib),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]  # Top 5 contributing features
    
    # Calculate trait influences
    trait_influences = calculate_trait_influence(
        driver_traits.get(driver_id, {}),
        base_pred
    )
    
    # Confidence (inversely related to absolute SHAP values std)
    confidence = 1 / (1 + np.std(feature_contrib))  # Sigmoid-like confidence
    
    explanation = {
        "predicted_position": float(round(base_pred, 1)),
        "confidence": float(confidence),
        "base_prediction": float(base_pred),
        "feature_contributions": [
            {"feature": name, "impact": float(impact)}
            for name, impact in top_features
        ],
        "trait_influences": trait_influences,
        "summary": format_explanation(base_pred, trait_influences, top_features)
    }
    
    return explanation


def calculate_trait_influence(
    traits: Dict,
    base_prediction: float
) -> List[Dict]:
    """
    Calculate how driver traits influence the prediction.
    
    Logic:
        - Qualifying Specialist: Predicts lower (better) qualifying results
        - Wet Weather Master: Better in rain races
        - Strong Starter: Gains positions early
        - Tire Management: Consistency helps
        - Inconsistent: Could drop positions
        - Track Expert: Better at familiar tracks
    
    Args:
        traits: Driver trait scores (0-1 for each trait)
        base_prediction: Base model prediction
    
    Returns:
        list: Trait influences, e.g.,
            [{"trait": "Qualifying Specialist", "impact": +0.5, "direction": "positive"}]
    """
    influences = []
    
    # Define trait impact direction and magnitude
    trait_impacts = {
        "qualifying_specialist": ("positive", 0.3),    # Positive in qualifying races
        "wet_weather_master": ("positive", 0.4),       # Positive in wet conditions
        "strong_starter": ("positive", 0.5),           # Gains positions
        "tire_management": ("positive", 0.2),          # More consistent
        "inconsistent": ("negative", -0.3),            # Can lose positions
        "track_expert": ("positive", 0.25),            # Better at familiar tracks
    }
    
    for trait_name, (trait_score) in traits.items():
        if trait_name not in trait_impacts:
            continue
        
        direction, max_impact = trait_impacts[trait_name]
        
        # Scale impact by trait score
        impact = max_impact * trait_score
        
        influences.append({
            "trait": trait_name.replace("_", " ").title(),
            "score": float(trait_score),
            "impact": float(impact),
            "direction": direction
        })
    
    # Sort by absolute impact
    influences.sort(key=lambda x: abs(x["impact"]), reverse=True)
    
    return influences


def format_explanation(
    base_pred: float,
    trait_influences: List[Dict],
    feature_contributions: List[Tuple]
) -> str:
    """
    Format explanation as human-readable text.
    
    Example output:
        "Predicted P2 based on: Strong Qualifier (+0.3), Tire Management (+0.2),
         Constructor Strength (-0.5). Top factor: driver_form"
    
    Args:
        base_pred: Base prediction (position)
        trait_influences: List of trait influences
        feature_contributions: List of (feature, impact) tuples
    
    Returns:
        str: Human-readable explanation
    """
    pos_traits = [t for t in trait_influences if t["direction"] == "positive"]
    neg_traits = [t for t in trait_influences if t["direction"] == "negative"]
    
    parts = [f"Predicted P{round(base_pred)}"]
    
    if pos_traits:
        pos_str = ", ".join([t["trait"] for t in pos_traits[:2]])
        parts.append(f"because: {pos_str} (+benefits)")
    
    if neg_traits:
        neg_str = ", ".join([t["trait"] for t in neg_traits[:1]])
        parts.append(f"but {neg_str} (-risks)")
    
    if feature_contributions:
        top_feature = feature_contributions[0][0]
        parts.append(f"(Top factor: {top_feature})")
    
    return " ".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test: Sample explanation structure
    sample_explanation = {
        "predicted_position": 2,
        "confidence": 0.78,
        "trait_influences": [
            {"trait": "Qualifying Specialist", "impact": 0.3, "direction": "positive"},
            {"trait": "Tire Management", "impact": 0.2, "direction": "positive"},
        ],
        "summary": "Predicted P2 because: Qualifying Specialist, Tire Management"
    }
    
    print("Sample Explanation:")
    print(sample_explanation["summary"])
