# Dutch Energy Price Analysis

This project focuses on analyzing and forecasting Day-Ahead prices of the Dutch energy grid using machine learning techniques. The current implementation uses synthetic data for testing and development purposes.

## Project Structure

```
dutch_energy_analysis/
├── data/               # Data directory for storing CSV files
├── notebooks/         # Jupyter notebooks for analysis
├── synthetic_data.py  # Module for generating synthetic data
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

- Synthetic data generation with realistic patterns
- Advanced feature engineering including:
  - Time-based features (hour, day, month)
  - Price momentum indicators
  - Volatility measures
  - Moving averages
- XGBoost-based price prediction model
- Detailed performance analysis by hour and day of week

## Usage

1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Open the notebooks in the `notebooks/` directory to see the analysis
