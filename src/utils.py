"""
Utilities Module

Shared utility functions for logging, data loading, configuration.

Functions:
    - setup_logging(): Configure logging for the project
    - load_config(): Load configuration from YAML/JSON
    - get_project_root(): Get project root directory
"""

import logging
from pathlib import Path
import yaml
import json
from typing import Dict, Any


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger("F1Predictor")
    
    # Create console handler with formatting
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Set log level
    logger.setLevel(getattr(logging, log_level.upper()))
    handler.setLevel(getattr(logging, log_level.upper()))
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


def get_project_root() -> Path:
    """
    Get the root directory of the project.
    
    Returns:
        Path: Project root directory
    """
    return Path(__file__).parent.parent


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    config_file = get_project_root() / config_path
    
    if not config_file.exists():
        logger = logging.getLogger(__name__)
        logger.warning(f"Config file not found: {config_file}")
        return {}
    
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    return {}


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
    
    Returns:
        Path: The directory path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    logger = setup_logging(log_level="DEBUG")
    logger.debug("Logging setup test")
    
    root = get_project_root()
    print(f"Project root: {root}")
