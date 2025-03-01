# Dutch Energy Price Analysis and Forecasting

This project implements a machine learning pipeline for analyzing and forecasting Dutch energy prices using data from the ENTSO-E API. The system uses XGBoost to predict energy prices up to 24 hours ahead.

## Project Structure

```
thesis-dutch-energy-analysis/
├── data/                    # Data and model artifacts
│   ├── models/             # Trained XGBoost models
│   ├── raw_prices.csv      # Raw price data from ENTSO-E
│   ├── features_*.csv      # Processed feature sets
│   └── *.png               # Generated visualizations
├── energy_price_analysis.ipynb  # Interactive analysis notebook
├── fetch_entsoe_data.py    # ENTSO-E API data fetching
├── prepare_features.py     # Feature engineering pipeline
├── train_model.py          # Model training and evaluation
├── utils.py               # Helper functions and metrics
└── requirements.txt       # Project dependencies
```

## Setup and Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your ENTSO-E API key in `.env`:
```
ENTSOE_API_KEY=your_api_key_here
```

## Usage

### 1. Data Collection
Fetch energy price data from ENTSO-E:
```bash
python fetch_entsoe_data.py
```

### 2. Feature Engineering
Process raw data and create features:
```bash
python prepare_features.py
```

### 3. Model Training
Train the XGBoost models:
```bash
python train_model.py
```

### 4. Interactive Analysis
Use the Jupyter notebook for interactive analysis:
```bash
jupyter notebook energy_price_analysis.ipynb
```

## Data Files

- `raw_prices.csv`: Original price data from ENTSO-E
- `features_scaled.csv`: Processed features (standardized)
- `features_unscaled.csv`: Processed features (original scale)
- `cv_metrics.csv`: Cross-validation results
- `test_predictions.csv`: Model predictions on test set

## Model Performance

The model achieves the following performance metrics:
- Short-term (1-6 hours): MAPE 6-17%
- Medium-term (7-12 hours): MAPE 18-24%
- Long-term (13-24 hours): MAPE 23-28%

## Features

The model uses several feature types:
1. Time-based features (hour, day, month, etc.)
2. Holiday indicators
3. Lagged price variables
4. Rolling statistics
5. Price differences and momentum

## License

This project is for academic research purposes.

## Author

Emma Arussi
