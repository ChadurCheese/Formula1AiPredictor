# F1 AI Predictor - Implementation Setup Guide

## 📁 Directory Structure

```
Formula1AiPredictor/
├── .github/
│   └── agents/
│       └── planner.agent.md
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # Data loading, feature engineering, caching
│   ├── features.py               # Feature extraction and engineering
│   ├── traits.py                 # Driver traits calculation engine
│   ├── model.py                  # Model training and prediction logic
│   ├── explainer.py              # SHAP-based explanation generation
│   ├── predict.py                # Prediction API (called by Streamlit)
│   └── utils.py                  # Utility functions (logging, helpers)
├── data/
│   ├── raw/                      # Raw F1 data (CSVs from Kaggle/Ergast)
│   │   ├── races.csv
│   │   ├── results.csv
│   │   ├── drivers.csv
│   │   ├── constructors.csv
│   │   ├── qualifying.csv
│   │   └── weather.csv           # (if available)
│   ├── processed/
│   │   └── features.pkl          # Processed features (pandas pickle)
│   └── cache/
│       ├── driver_traits.json    # Cached trait calculations
│       ├── predictions_cache.pkl # Cached predictions
│       └── model_metadata.json   # Model info (date trained, version, etc)
├── models/
│   ├── model.pkl                 # Trained XGBoost model
│   ├── model_config.json         # Model hyperparameters
│   └── feature_importance.json   # Feature importance scores
├── notebooks/
│   ├── 01_data_exploration.ipynb # Data analysis & validation
│   └── 02_model_development.ipynb # Model iteration & tuning
├── app/
│   ├── app.py                    # Main Streamlit application
│   ├── pages/
│   │   ├── 01_predictions.py     # Race predictions page
│   │   ├── 02_driver_traits.py   # Driver traits dashboard
│   │   └── 03_historical_analysis.py  # Historical race analysis
│   ├── components/
│   │   ├── prediction_card.py    # Reusable UI components
│   │   ├── trait_visualizer.py   # Trait radar/bar charts
│   │   └── comparison_table.py   # Predicted vs actual table
│   └── config.py                 # Streamlit configuration
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_traits.py
│   ├── test_model.py
│   └── test_predict.py
├── scripts/
│   ├── download_data.py          # Script to download F1 data from Ergast
│   ├── train_model.py            # Script to train model
│   ├── calculate_traits.py       # Script to calculate driver traits
│   └── validate_predictions.py   # Script to validate model accuracy
├── docs/
│   ├── ARCHITECTURE.md           # System architecture details
│   ├── TRAITS_METHODOLOGY.md     # How traits are calculated
│   ├── DATA_SOURCES.md           # Data source documentation
│   ├── REAL_TIME_INTEGRATION.md  # Future live race integration guide
│   └── SETUP.md                  # Detailed setup instructions
├── .gitignore
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup (optional)
├── README.md                     # Project overview
├── IMPLEMENTATION_SETUP.md       # This file
└── IMPLEMENTATION_PLAN.md        # Detailed 7-day plan (save planner output here)
```

---

## 🛠️ Technologies & Languages

### **Primary Language: Python 3.10+**
- **Why**: Rich ML ecosystem, fast development, Streamlit native
- **Version**: 3.10 or 3.11 (compatibility with XGBoost, pandas, scikit-learn)

### **Data Science Stack**
| Tool | Purpose | Version |
|------|---------|---------|
| **Pandas** | Data loading, feature engineering, data wrangling | 2.0+ |
| **NumPy** | Numerical computations, array operations | 1.24+ |
| **Scikit-learn** | Preprocessing, model utilities | 1.3+ |
| **XGBoost** | Gradient boosting model (or LightGBM) | 2.0+ |
| **SHAP** | Model explainability, feature importance | 0.13+ |

### **Frontend: Streamlit**
| Tool | Purpose | Version |
|------|---------|---------|
| **Streamlit** | Web UI framework (zero backend needed) | 1.28+ |
| **Plotly** | Interactive charts & visualizations | 5.17+ |
| **Altair** | Declarative visualizations (included with Streamlit) | Latest |

### **Data Source Integration**
| Tool | Purpose |
|------|---------|
| **Requests** | HTTP requests to Ergast F1 API (historical data) |
| **Kaggle API** | Download data from Kaggle (alternative source) |
| **SQLite** | Local database for caching (future enhancement) |

### **Development Tools**
| Tool | Purpose |
|------|---------|
| **Jupyter** | Exploratory data analysis notebooks |
| **Git** | Version control |
| **Pytest** | Unit testing |
| **Black** | Code formatting |
| **Pylint** | Code quality |

---

## 📋 File Breakdown by Day

### **Days 1-2: Data Preparation**

**Created Files**:
- `src/data_pipeline.py` - Load raw F1 data from Kaggle CSV or Ergast API
- `src/features.py` - Build features: driver stats, constructor, track, form
- `src/traits.py` - Calculate 6 driver traits from historical data
- `scripts/download_data.py` - Download/cache Kaggle F1 dataset
- `data/raw/` - Store downloaded CSVs
- `data/cache/driver_traits.json` - Cache trait calculations
- `notebooks/01_data_exploration.ipynb` - Data validation & exploration

**Key Functions**:
```python
# data_pipeline.py
def load_raw_data() -> dict  # Load all CSVs
def engineer_features() -> pd.DataFrame  # Create feature matrix
def cache_features() -> None  # Save to pickle

# traits.py
def calculate_driver_traits(historical_data) -> dict  # Compute 6 traits
def trait_qualifying_specialist(driver_data) -> float
def trait_wet_weather_master(driver_data) -> float
# ... 4 more trait functions
```

### **Days 3-4: Model Training & Explainability**

**Created Files**:
- `src/model.py` - Train XGBoost model on feature matrix
- `src/explainer.py` - Generate SHAP explanations & trait influence
- `src/predict.py` - Prediction API (returns position + explanation + traits)
- `scripts/train_model.py` - Script to train model
- `models/model.pkl` - Trained model weights
- `models/model_config.json` - Hyperparameters
- `notebooks/02_model_development.ipynb` - Model iteration

**Key Functions**:
```python
# model.py
def train_model(X_train, y_train) -> XGBRegressor
def evaluate_model(model, X_test, y_test) -> dict

# explainer.py
def explain_prediction(model, driver_id, race_id) -> dict
def calculate_trait_influence(base_pred, driver_traits) -> list

# predict.py
def predict_race(race_id) -> list[dict]  # Returns all driver predictions
# Returns: [{'driver': 'HAM', 'predicted_pos': 2, 'confidence': 0.78, 
#            'traits': [...], 'explanation': "..."}, ...]
```

### **Days 5-6: Streamlit UI & Traits Dashboard**

**Created Files**:
- `app/app.py` - Streamlit main entry point (multi-page setup)
- `app/pages/01_predictions.py` - Race prediction page
- `app/pages/02_driver_traits.py` - Driver trait dashboard
- `app/pages/03_historical_analysis.py` - Historical accuracy review
- `app/components/prediction_card.py` - Reusable prediction display
- `app/components/trait_visualizer.py` - Radar/bar charts for traits
- `app/config.py` - Streamlit theme, caching, layout settings

**Streamlit Pages**:
1. **Predictions Page**: Select race → show predicted vs actual positions
2. **Driver Traits Page**: Select driver → show 6 traits with radar chart
3. **Historical Analysis**: Browse past seasons, model accuracy metrics

### **Day 7: Testing & Deployment**

**Created Files**:
- `tests/test_*.py` - Unit tests for all modules
- `README.md` - Project overview and quickstart
- `docs/ARCHITECTURE.md` - System design
- `docs/TRAITS_METHODOLOGY.md` - How traits are calculated
- `docs/REAL_TIME_INTEGRATION.md` - Future live data hooks

---

## 🚀 Quick Setup Instructions

### **Step 1: Initialize Python Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Download Data**
```bash
python scripts/download_data.py
# Downloads F1 data from Ergast API or Kaggle to data/raw/
```

### **Step 4: Build Features & Train Model**
```bash
python scripts/train_model.py
# Outputs: models/model.pkl, data/cache/driver_traits.json
```

### **Step 5: Run Streamlit App**
```bash
streamlit run app/app.py
# Opens browser at http://localhost:8501
```

---

## 📦 requirements.txt

```
# Core Data Science
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0

# Model Training
xgboost==2.0.0
# OR: lightgbm==4.0.0

# Explainability
shap==0.42.1

# Frontend
streamlit==1.28.0
plotly==5.17.0

# Utilities
requests==2.31.0
python-dateutil==2.8.2
pyyaml==6.0
tqdm==4.66.1

# Development
jupyter==1.0.0
pytest==7.4.0
black==23.9.1
pylint==2.17.5

# Optional: For Kaggle data download
kaggle==1.5.13
```

---

## 🎯 Language & Format Choices

### **Python Versions**
- **3.10+**: Required for modern pandas, scikit-learn, streamlit
- **No Python 2**: End of life, incompatible with latest libraries

### **Data Formats**
| Format | Use Case | Location |
|--------|----------|----------|
| CSV | Raw input data | `data/raw/` |
| Pickle (.pkl) | Cached features, model weights | `data/cache/`, `models/` |
| JSON | Traits, metadata, config | `data/cache/`, `models/` |
| Parquet | Future: High-compression storage | (Post-MVP) |

### **Code Organization**
- **Modular**: Each function has single responsibility
- **Type hints**: `def predict_race(race_id: int) -> list[dict]`
- **Docstrings**: Google-style docstrings for all functions
- **Config files**: YAML/JSON for all hyperparameters (avoid hardcoding)

### **Testing Framework**
- **Pytest**: Unit tests in `tests/` directory
- **Test file naming**: `test_module_name.py`
- **Coverage target**: 80%+ for critical paths (model, traits, prediction)

---

## 🔧 Configuration Files

### **models/model_config.json** (XGBoost hyperparameters)
```json
{
  "model_type": "xgboost",
  "hyperparameters": {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "reg:squarederror",
    "random_state": 42
  },
  "training_data": {
    "train_set": "2020-2022",
    "val_set": "2023",
    "test_set": "2024"
  },
  "performance": {
    "mse": 2.14,
    "mae": 1.23,
    "r2": 0.68
  }
}
```

### **app/config.py** (Streamlit settings)
```python
# App configuration constants
RACES_TO_LOAD = 50  # Historical races to cache
MODEL_PATH = "models/model.pkl"
TRAITS_CACHE = "data/cache/driver_traits.json"
FEATURES_CACHE = "data/processed/features.pkl"

# UI Settings
THEME = "light"
LAYOUT = "wide"
SIDEBAR_WIDTH = 2

# Trait display
TRAIT_NAMES = {
    "qualifying_specialist": "Qualifying Specialist",
    "wet_weather_master": "Wet Weather Master",
    # ... etc
}
```

---

## 📝 File Naming Conventions

- **Modules**: `snake_case.py` (e.g., `data_pipeline.py`)
- **Classes**: `PascalCase` (e.g., `RacePredictor`)
- **Functions**: `snake_case()` (e.g., `predict_race()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_DRIVERS = 20`)
- **Data files**: `descriptive_name.csv` (e.g., `races_2024.csv`)

---

## 🔀 Git Workflow

### **Initial Commit Structure**
```
1. Initial setup (requirements.txt, .gitignore, README)
2. Data pipeline (src/data_pipeline.py, src/features.py)
3. Traits engine (src/traits.py)
4. Model training (src/model.py, src/explainer.py)
5. Prediction API (src/predict.py)
6. Streamlit app (app/app.py, pages/)
7. Tests & docs
```

### **.gitignore**
```
# Virtual environment
venv/
env/

# Data (raw data, cache, models - consider if shared)
data/raw/
data/processed/
data/cache/
models/*.pkl
models/*.joblib

# Cache/temp
__pycache__/
*.pyc
.pytest_cache/
.streamlit/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.DS_Store
```

---

## ✅ Pre-Implementation Checklist

- [ ] Python 3.10+ installed locally
- [ ] Git initialized in project root
- [ ] Virtual environment created (`venv/`)
- [ ] `requirements.txt` saved in root
- [ ] `.gitignore` configured
- [ ] Directory structure created
- [ ] README.md written (project overview)
- [ ] Day-1 data download script ready
- [ ] First commit: "Initial project setup"

---

## 📌 Key Decision Points

1. **Model Choice**: XGBoost vs LightGBM
   - **XGBoost**: More stable, better documentation (recommended for MVP)
   - **LightGBM**: Faster training, slightly less interpretable

2. **Data Source**: Kaggle vs Ergast API
   - **Kaggle**: Complete dataset, one-time download
   - **Ergast API**: Real-time, but rate-limited
   - **Recommendation**: Download Kaggle once, cache locally

3. **Feature Store**: Pickle vs Parquet vs SQLite
   - **Pickle**: Fast, simple (MVP choice)
   - **Parquet**: Compressed, queryable (future)
   - **SQLite**: For real-time queries (post-MVP)

4. **Deployment**: Streamlit Cloud vs Local
   - **Streamlit Cloud**: Free, instant (recommended)
   - **Local Uvicorn**: More control, requires server

---

## 🎯 Next Steps

1. **Review this setup** with the team
2. **Create directory structure** manually or via script
3. **Initialize Git repository**
4. **Create virtual environment**
5. **Install dependencies**
6. **Start Day 1**: Data pipeline implementation

Ready to start building? Let me know if you'd like me to generate:
- The initial data download script
- Skeleton Python files for each module
- Pytest templates for testing
