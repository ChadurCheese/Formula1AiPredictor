"""
Quickstart script for the F1 Race Predictor.

For a new clone of this repo, this is the one command to get running:

    python run.py

What it does:
    1. Checks that the required Kaggle CSVs are present in data/raw/
       (can't be auto-downloaded - Kaggle requires an account/API key)
    2. Installs dependencies from requirements.txt (skipped if the core
       packages already import successfully, so repeat runs are fast)
    3. Trains the model if models/model.pkl doesn't exist yet
    4. Launches the Streamlit app

Flags:
    --skip-install   Don't check/install dependencies
    --retrain        Retrain the model even if models/model.pkl already exists
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

REQUIRED_DATA_FILES = [
    "races.csv", "results.csv", "drivers.csv", "constructors.csv", "qualifying.csv",
]
KAGGLE_URL = "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024"


def check_data_files() -> bool:
    """Verify the required Kaggle CSVs are present before doing anything else."""
    raw_dir = ROOT / "data" / "raw"
    missing = [f for f in REQUIRED_DATA_FILES if not (raw_dir / f).exists()]

    if missing:
        print("Missing required data files in data/raw/:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nDownload the dataset from:\n  {KAGGLE_URL}")
        print(f"and place the CSVs in: {raw_dir}")
        return False

    print("Data files found.")
    return True


def dependencies_satisfied() -> bool:
    """Quick check so repeat runs don't pay the pip install cost every time."""
    try:
        import pandas, numpy, sklearn, xgboost, shap, streamlit, plotly  # noqa: F401
        return True
    except ImportError:
        return False


def install_dependencies() -> None:
    print("Installing dependencies from requirements.txt...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )


def train_model_if_needed(force: bool = False) -> None:
    model_path = ROOT / "models" / "model.pkl"

    if model_path.exists() and not force:
        print("Trained model found (models/model.pkl) - skipping training.")
        print("  Run `python run.py --retrain` to retrain from scratch.")
        return

    print("Training model (this takes under a minute)...")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_model.py")],
        check=True,
        cwd=ROOT,
    )


def launch_app() -> None:
    print("Launching Streamlit app - it will open in your browser shortly...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "app.py")],
        cwd=ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Don't check/install dependencies")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model even if one already exists")
    args = parser.parse_args()

    print("=" * 60)
    print("F1 Race Predictor - Quickstart")
    print("=" * 60)

    if not check_data_files():
        sys.exit(1)

    if not args.skip_install and not dependencies_satisfied():
        install_dependencies()
    else:
        print("Dependencies already satisfied - skipping install.")

    train_model_if_needed(force=args.retrain)
    launch_app()


if __name__ == "__main__":
    main()
