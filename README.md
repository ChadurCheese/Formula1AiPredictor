# Formula 1 AI Race Predictor

[![Tests](https://github.com/ChadurCheese/Formula1AiPredictor/actions/workflows/tests.yml/badge.svg)](https://github.com/ChadurCheese/Formula1AiPredictor/actions/workflows/tests.yml)

An intelligent ML system that predicts F1 race results with explainable driver traits, similar to player characteristics in Football Manager games.

## 🎯 Features

- **Race Position Prediction**: Predict finishing positions using historical data
- **Driver Traits System**: FM-style player traits (Qualifying Specialist, Wet Weather Master, etc.)
- **Explainable Predictions**: Understand *why* a prediction was made with trait breakdowns
- **Historical Analysis**: Review model accuracy on past seasons
- **Streamlit UI**: Interactive web interface for exploration

## 🏗️ Project Structure

```
src/              # Core ML pipeline (data, features, model, traits)
app/              # Streamlit web UI
data/             # raw/ → processed/ → cache/
models/           # Trained model and config
notebooks/        # Exploratory analysis
tests/            # Unit tests
scripts/          # Training, validation scripts
docs/             # Architecture and methodology docs
```

## 🚀 Quick Start

The data (`data/raw/*.csv`) and a trained model (`models/model.pkl`) are already
committed to this repo, so a fresh clone works out of the box:

```bash
git clone https://github.com/ChadurCheese/Formula1AiPredictor.git
cd Formula1AiPredictor
python run.py
```

`run.py` installs dependencies (skipped if already satisfied), trains the
model if `models/model.pkl` is missing, and launches the Streamlit app in
your browser. Useful flags:

```bash
python run.py --retrain        # retrain the model even if one already exists
python run.py --skip-install   # skip the dependency check/install
```

### Manual setup (if you'd rather do it step by step)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
python scripts/train_model.py   # creates models/model.pkl, data/cache/driver_traits.json
streamlit run app/app.py        # opens at http://localhost:8501
```

### Refreshing the data

The committed CSVs are a snapshot. To pull a newer version, download from the
[Kaggle F1 Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024)
(races, results, drivers, constructors, qualifying, pit_stops, status, circuits),
replace the files in `data/raw/`, then run `python run.py --retrain`.

## 📊 Tech Stack

- **Python 3.10+**
- **ML**: XGBoost, scikit-learn
- **Data**: Pandas, NumPy
- **UI**: Streamlit, Plotly
- **Explainability**: SHAP

## 📈 Model Performance

- **Target**: Predict race finishing positions
- **Current accuracy (2024 test set)**: ~50% of predictions within 2 positions, MAE ~3.0 positions
  (honest, leakage-free numbers - see `models/model_config.json` for the exact current figures)
- **Training Data**: 2020-2022
- **Validation**: 2023
- **Test Set**: 2024

The original aspirational goal was 65%+ within 2 positions. Reaching that
would likely need data this project doesn't have access to (weather, tire
compound choices, safety car history) - see
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## 🎮 Driver Traits

Each driver is assigned 6 dynamically-calculated traits:

1. **Qualifying Specialist** - Excels in qualifying but variable race day
2. **Wet Weather Master** - Better performance in rain
3. **Strong Starter** - High overtakes in early laps
4. **Tire Management Expert** - Consistent across tire strategies
5. **Inconsistent** - High variability in finishes
6. **Track Expert** - Strong at familiar circuits

## 📚 Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design
- [Traits Methodology](docs/TRAITS_METHODOLOGY.md) - How traits are calculated
- [Data Sources](docs/DATA_SOURCES.md) - Data pipeline info
- [Implementation Setup](IMPLEMENTATION_SETUP.md) - Detailed setup guide

## 🔌 Future Extensions

- Real-time race predictions with live timing API
- Qualifying predictions
- Safety car / incident modeling
- Championship simulations
- Cloud deployment

## 📝 Development Notes

- 1-week MVP timeline
- Streamlit for rapid prototyping
- Historical data only (MVP phase)
- Designed for future live race integration

## 👤 Author

Piotr Mazur

## 📄 License

MIT
