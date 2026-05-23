"""
Model Training Module

Trains XGBoost model for F1 race position prediction.

Functions:
    - train_model(): Train XGBoost on feature matrix
    - evaluate_model(): Calculate model performance metrics
    - save_model(): Save trained model to disk
    - load_model(): Load model from disk
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import pickle
import json
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

# Model configuration
MODEL_CONFIG = {
    "model_type": "xgboost",
    "hyperparameters": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "objective": "reg:squarederror",
        "random_state": 42,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "training_data": {
        "train_set": "2020-2022",
        "val_set": "2023",
        "test_set": "2024",
    }
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None
) -> xgb.XGBRegressor:
    """
    Train XGBoost model for race position prediction.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target (finishing positions)
        X_val: Validation features (optional, for early stopping)
        y_val: Validation targets (optional)
    
    Returns:
        xgb.XGBRegressor: Trained model
    """
    logger.info(f"Training XGBoost model on {X_train.shape[0]} samples...")
    
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        logger.info("Using validation set for early stopping")
    
    model = xgb.XGBRegressor(
        **MODEL_CONFIG["hyperparameters"],
        eval_metric="mae",
        verbose=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=10 if eval_set else None,
        verbose=False
    )
    
    logger.info("Model training complete")
    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """
    Evaluate model performance on test set.
    
    Metrics:
        - MSE: Mean Squared Error
        - MAE: Mean Absolute Error (avg position error)
        - R2: Coefficient of determination
        - Within X positions: % predictions within 1, 2, 3 positions
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        dict: Performance metrics
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate "within X positions" accuracy
    errors = np.abs(y_pred - y_test.values)
    within_1 = (errors <= 1).sum() / len(errors) * 100
    within_2 = (errors <= 2).sum() / len(errors) * 100
    within_3 = (errors <= 3).sum() / len(errors) * 100
    
    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "within_1_position": float(within_1),
        "within_2_positions": float(within_2),
        "within_3_positions": float(within_3),
        "test_samples": len(X_test),
    }
    
    logger.info(f"Model Evaluation: MAE={mae:.2f}, R²={r2:.3f}")
    logger.info(f"Predictions within: 1 pos={within_1:.1f}%, 2 pos={within_2:.1f}%, 3 pos={within_3:.1f}%")
    
    return metrics


def save_model(
    model: xgb.XGBRegressor,
    model_path: Path = Path("models/model.pkl"),
    metrics: Dict = None
) -> None:
    """
    Save trained model and configuration to disk.
    
    Args:
        model: Trained model
        model_path: Path to save model pickle
        metrics: Performance metrics to save in config
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved model to {model_path}")
    
    # Save model config and metrics
    config = MODEL_CONFIG.copy()
    if metrics:
        config["performance"] = metrics
    
    config_path = model_path.parent / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved config to {config_path}")


def load_model(model_path: Path = Path("models/model.pkl")) -> xgb.XGBRegressor:
    """
    Load previously trained model from disk.
    
    Args:
        model_path: Path to model pickle file
    
    Returns:
        xgb.XGBRegressor: Loaded model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded model from {model_path}")
    return model


def get_feature_importance(model: xgb.XGBRegressor, top_n: int = 20) -> Dict:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        top_n: Number of top features to return
    
    Returns:
        dict: Feature names and importance scores (sorted by importance)
    """
    importance = model.get_booster().get_score(importance_type='weight')
    
    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_importance = sorted_importance[:top_n]
    
    return dict(top_importance)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test: Print model config
    print("Model Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
