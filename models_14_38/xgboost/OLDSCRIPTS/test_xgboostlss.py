import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from scipy.stats import norm

import multiprocessing
import plotnine
from plotnine import *
plotnine.options.figure_size = (12, 8)

def prepare_data(data, horizon, feature_set='all'):
    """Prepare features and target for a specific horizon
    
    Args:
        data: DataFrame with features and targets
        horizon: Forecast horizon
        feature_set: One of ['all', 'price_only', 'no_weather', 'weather_only']
    """
    # Define feature patterns to exclude based on feature_set
    if feature_set == 'price_only':
        excluded_patterns = [
            'wind', 'Wind', 'WIND',
            'solar', 'Solar', 'SOLAR',
            'coal', 'Coal', 'COAL',
            'consumption', 'Consumption', 'CONSUMPTION',
            'load', 'Load', 'LOAD'
        ]
    elif feature_set == 'no_weather':
        excluded_patterns = [
            'wind', 'Wind', 'WIND',
            'solar', 'Solar', 'SOLAR',
            'temperature', 'Temperature', 'TEMPERATURE'
        ]
    elif feature_set == 'weather_only':
        included_patterns = [
            'wind', 'Wind', 'WIND',
            'solar', 'Solar', 'SOLAR',
            'temperature', 'Temperature', 'TEMPERATURE'
        ]
        # Get weather-related features plus basic time features
        feature_cols = [col for col in data.columns 
                       if not col.startswith('target_t') and
                       (any(pattern in col for pattern in included_patterns) or
                        any(t in col for t in ['hour', 'day', 'month', 'holiday']))]
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y
    else:  # 'all' features
        excluded_patterns = []
    
    # Get all columns except target columns and excluded features
    feature_cols = [col for col in data.columns 
                   if not col.startswith('target_t') and
                   not any(pattern in col for pattern in excluded_patterns)]
    
    X = data[feature_cols]
    y = data[f'target_t{horizon}']
    return X, y

# Load data
data_path = Path('data/processed/multivariate_features.csv')
data = pd.read_csv(data_path, index_col=0)
data.index = pd.to_datetime(data.index)

# Split data
train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')

train_data = data[:train_end]
test_data = data[test_start:test_end]

# Test different feature sets
horizon = 14
feature_sets = ['all', 'price_only', 'no_weather', 'weather_only']

for feature_set in feature_sets:
    print(f"\nTesting feature set: {feature_set}")
    
    # Prepare data
    X_train, y_train = prepare_data(train_data, horizon, feature_set)
    X_test, y_test = prepare_data(test_data, horizon, feature_set)
    
    print(f"Number of features: {X_train.shape[1]}")
    print("Sample features:", ', '.join(list(X_train.columns)[:5]))

    xgblss = XGBoostLSS(
        Gaussian(stabilization="None",  
                response_fn="exp",      
                loss_fn="nll"          
                )
    )
    
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Define parameter search space
    param_dict = {
        "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
        "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
        "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
        "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
        "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
        "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}],
        "booster":          ["categorical", ["gbtree"]]
    }

    # Optimize hyperparameters
    print("Optimizing hyperparameters...")
    opt_param = xgblss.hyper_opt(param_dict,
                             dtrain,
                             num_boost_round=100,
                             nfold=5,
                             early_stopping_rounds=20,
                             max_minutes=10,
                             n_trials=30,
                             silence=True,
                             seed=123,
                             hp_seed=123
                            )
    
    print("\nBest parameters:")
    for param, value in opt_param.items():
        print(f"{param}: {value}")
    

    np.random.seed(123)

    opt_params = opt_param.copy()
    n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]

    # Train Model with optimized hyperparameters
    xgblss.train(opt_params,
                dtrain,
                num_boost_round=n_rounds
                )

    # Make predictions
    print("\nMaking predictions...")
    pred = xgblss.predict(dtest)
    
    # For Gaussian distribution, first column is mean (mu) and second column is log(sigma)
    mu = pred.iloc[:, 0]  # mean predictions
    log_sigma = pred.iloc[:, 1]  # log standard deviation predictions
    sigma = np.exp(np.clip(log_sigma, -2, 3))  # allow for wider range of uncertainties
    
    # Calculate metrics
    rmse = np.sqrt(((y_test.values - mu) ** 2).mean())
    avg_std = sigma.mean()
    
    print(f"\nResults for horizon t+{horizon}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"Average predicted standard deviation: {avg_std:.2f}")
    
    # Print first few predictions with uncertainty
    print("\nFirst 5 predictions with uncertainty:")
    for i in range(5):
        print(f"Actual: {y_test.iloc[i]:.2f}, Predicted: {mu.iloc[i]:.2f} ± {2*sigma.iloc[i]:.2f}")
    
    # Calculate prediction interval coverage
    z_score = 1.28  # For 80% prediction interval
    lower_bound = pd.Series(mu - z_score * sigma, index=y_test.index)
    upper_bound = pd.Series(mu + z_score * sigma, index=y_test.index)
    coverage = ((y_test >= lower_bound) & (y_test <= upper_bound)).mean() * 100
    print(f"\n80% Prediction Interval Coverage: {coverage:.1f}%")
    
    # Save results to a different plot for each feature set
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, 'k-', label='Actual', alpha=0.7)
    plt.plot(y_test.index, mu, 'b--', label='Predicted', alpha=0.7)
    plt.fill_between(y_test.index, 
                     lower_bound, 
                     upper_bound, 
                     color='b', 
                     alpha=0.2, 
                     label='80% Prediction Interval')
    plt.title(f'XGBoostLSS Predictions (t+{horizon}h) - {feature_set} features')
    plt.xlabel('Date')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create plots directory if it doesn't exist
    os.makedirs('models_14_38/xgboost/plots/lss', exist_ok=True)
    plt.savefig(f'models_14_38/xgboost/plots/lss/lss_predictions_t{horizon}_{feature_set}.png')
    plt.close()
    
    print('\n' + '='*50)
