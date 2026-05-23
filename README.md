# Formula 1 AI Race Predictor

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

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 2. Download Data
Download from [Kaggle F1 Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024):
- races.csv
- results.csv
- drivers.csv
- constructors.csv
- qualifying.csv
- pit_stops.csv
- status.csv

Save all CSVs to `data/raw/`

### 3. Train Model
```bash
python scripts/train_model.py
# Creates: models/model.pkl, data/cache/driver_traits.json
```

### 4. Run Web App
```bash
streamlit run app/app.py
# Opens at http://localhost:8501
```

## 📊 Tech Stack

- **Python 3.10+**
- **ML**: XGBoost, scikit-learn
- **Data**: Pandas, NumPy
- **UI**: Streamlit, Plotly
- **Explainability**: SHAP

## 📈 Model Performance

- **Target**: Predict race finishing positions
- **Accuracy Goal**: ±2 positions for 65%+ of predictions
- **Training Data**: 2020-2022
- **Validation**: 2023
- **Test Set**: 2024

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
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - 7-day development plan

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
