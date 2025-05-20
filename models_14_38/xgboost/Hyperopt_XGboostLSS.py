"""
This model is the XGBoostLSS model, which is a quantile regression model.
it trains on full train and predicts on last 3 months test set
it uses basic parameters, 
the difference with the XGboost model, is that it uses distributional regression to predict both the mean and variance of electricity prices.
how this works is that it fits a distribution to the data, and then uses the distribution to predict the mean and variance of the data.
the xgboost model only predicts the mean, and the variance is estimated by the model.

XGBoostLSS models uncertainty directly via distributional regression by learning both mean 
and variance of the target variable. Hyperparameters are optimized to 
minimize negative log-likelihood of the predicted Gaussian distribution, 
in contrast to standard XGBoost which is tuned to minimize RMSE. 
This probabilistic approach enables principled uncertainty quantification 
and calibrated prediction intervals.

"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

def load_and_prepare_data(horizon):
    """Load and prepare data for a specific horizon"""
    data_path = Path('data/processed/multivariate_features_selectedXGboost.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')
    
    # Train-test split
    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    
    train_df = df[train_start:train_end]
    test_df = df[test_start:test_end]
    
    # Get target columns and feature columns
    target_cols = [col for col in df.columns if col.startswith('target_t')]
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[f'target_t{horizon}']
    X_test = test_df[feature_cols]
    y_test = test_df[f'target_t{horizon}']
    
    return X_train, y_train, X_test, y_test

def train_and_analyze_model(horizon):
    """Train model and analyze its behavior"""
    # Load data
    X_train, y_train, X_test, y_test = load_and_prepare_data(horizon=horizon)
    
    print("Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    


    model = XGBoostLSS(Gaussian())

    print("Optimizing hyperparameters...")
    param_dict = {
        "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
        "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
        "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
        "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
        "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
        "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}]
    }


    opt_param = model.hyper_opt(
        param_dict,
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

    opt_params = opt_param.copy()
    n_rounds = opt_params.pop("opt_rounds")

    model.train(params=opt_params, dtrain=dtrain, num_boost_round=n_rounds, verbose_eval=10)
    predictions = model.predict(dtest)

    pred_mean = predictions['loc'].values
    pred_std = predictions['scale'].values
    y_test_values = y_test.values

    metrics = calculate_metrics(y_test_values, pred_mean)
    print(f"\n\U0001F4CA Evaluation for t+{horizon}h horizon:")
    print(f"Number of predictions: {len(y_test_values)}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"SMAPE: {metrics['SMAPE']:.2f}%")
    print(f"R2: {metrics['R2']:.4f}")

    z = 1.645  # 90% confidence interval
    lower = pred_mean - z * pred_std
    upper = pred_mean + z * pred_std
    coverage = np.mean((y_test_values >= lower) & (y_test_values <= upper))
    print(f"\n90% Prediction interval coverage: {coverage:.2%}")

    out_dir = Path(f'models_14_38/xgboost/plots/debug')
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': pred_mean,
        'Lower': lower,
        'Upper': upper
    }, index=y_test.index)

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df.index, plot_df['Actual'], label='Actual', alpha=0.7)
    plt.plot(plot_df.index, plot_df['Predicted'], label='Predicted', alpha=0.7)
    plt.fill_between(plot_df.index, plot_df['Lower'], plot_df['Upper'], alpha=0.2, label='90% CI')
    plt.title(f'Actual vs Predicted Prices - t+{horizon}h')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f'predictions_{horizon}.png')
    plt.close()

    residuals = y_test - pred_mean
    plt.figure(figsize=(12, 6))
    plt.scatter(pred_mean, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residuals vs Predicted Values - t+{horizon}h')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(out_dir / f'residuals_{horizon}.png')
    plt.close()

    print("\nResiduals Statistics:")
    print(pd.Series(residuals).describe())

if __name__ == "__main__":
    for horizon in [14, 24, 38]:
        train_and_analyze_model(horizon)

