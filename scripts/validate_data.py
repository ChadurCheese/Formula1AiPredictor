"""
Data Initialization Module

Sets up initial data directory structure and validates F1 dataset.
Run this first before training!

Usage:
    python scripts/validate_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging

logger = setup_logging(log_level="INFO")


def validate_data_structure():
    """Check if all required CSV files exist in data/raw/."""
    
    required_files = [
        "races.csv",
        "results.csv",
        "drivers.csv",
        "constructors.csv",
        "qualifying.csv",
    ]
    
    optional_files = [
        "pit_stops.csv",
        "status.csv",
        "circuits.csv",
    ]
    
    raw_data_path = Path("data/raw")
    
    logger.info("Checking data files...")
    logger.info(f"Looking in: {raw_data_path.absolute()}")
    
    # Check required files
    missing_required = []
    for filename in required_files:
        filepath = raw_data_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ {filename} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"  ❌ {filename} - MISSING")
            missing_required.append(filename)
    
    # Check optional files
    logger.info("\nOptional files:")
    for filename in optional_files:
        filepath = raw_data_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ {filename} ({size_mb:.1f} MB)")
        else:
            logger.info(f"  ℹ️  {filename} - not found (optional)")
    
    if missing_required:
        logger.error(f"\n❌ Missing required files: {missing_required}")
        logger.info("Download from: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024")
        return False
    
    logger.info("\n✅ All required data files present!")
    return True


def main():
    """Main validation entry point."""
    logger.info("=" * 60)
    logger.info("F1 RACE PREDICTOR - DATA VALIDATION")
    logger.info("=" * 60)
    
    if validate_data_structure():
        logger.info("\nReady to run: python scripts/train_model.py")
    else:
        logger.error("\nPlease download missing files first.")


if __name__ == "__main__":
    main()
