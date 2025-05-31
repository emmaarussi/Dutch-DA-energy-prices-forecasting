"""
XGBoost-based probabilistic forecasting using quantile regression.
This model predicts multiple quantiles (0.1, 0.5, 0.9) to provide uncertainty estimates
for electricity price forecasts.
"""


import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics


class XGBoostQuantileForecaster:
    def __init__(self):
        self.horizons = range(14, 15)  # Only t+14 for now
        self.quantiles = [0.1, 0.5, 0.9]
        self.models = {}
        self.scaler = RobustScaler()

    def quantile_loss(self, q):
        def loss(preds, dtrain):
            labels = dtrain.get_label()
            errors = labels - preds
            grad = np.where(errors < 0, -q, 1 - q)
            hess = np.ones_like(labels) * 0.001  # Set hessian to zero for quantile regression
            return grad, hess
        return loss

    def get_hyperparameters(self, horizon):
        if horizon <= 24:
            return {
                'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 2500,
                'min_child_weight': 4, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'gamma': 0.2, 'random_state': 42
            }
        elif horizon <= 31:
            return {
                'max_depth': 6, 'learning_rate': 0.025, 'n_estimators': 2800,
                'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.7,
                'gamma': 0.25, 'random_state': 42
            }
        else:
            return {
                'max_depth': 6, 'learning_rate': 0.02, 'n_estimators': 3000,
                'min_child_weight': 6, 'subsample': 0.6, 'colsample_bytree': 0.6,
                'gamma': 0.3, 'random_state': 42
            }


    def prepare_data(self, df, horizon):
        """Prepare features and target for a specific horizon, excluding wind, coal, solar, and consumption features"""
        # Define excluded feature patterns
        excluded_patterns = [
            'wind', 'Wind', 'WIND',
            'solar', 'Solar', 'SOLAR',
            'coal', 'Coal', 'COAL',
            'consumption', 'Consumption', 'CONSUMPTION',
            'load', 'Load', 'LOAD'
        ]
        
        # Get all columns except target columns and excluded features
        feature_cols = [col for col in df.columns 
                       if not col.startswith('target_t') and 
                       not any(pattern in col for pattern in excluded_patterns)]
        
        X = df[feature_cols]
        y = df[f'target_t{horizon}']
        return X, y

    def train_and_predict(self, train_window, test_window, window_size_str):
        predictions_all = []
        scaler = RobustScaler()

        for horizon in self.horizons:
            X_train, y_train = self.prepare_data(train_window, horizon)
            X_test, y_test = self.prepare_data(test_window, horizon)

            if len(X_train) < 100 or len(X_test) == 0:
                continue

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            dtrain = xgb.DMatrix(X_train_scaled, label=y_train.clip(lower=0))
            dtest = xgb.DMatrix(X_test_scaled)
            quantile_predictions = {}

            for q in self.quantiles:
                params = self.get_hyperparameters(horizon)
                try:
                    model = xgb.train(params, dtrain, num_boost_round=params['n_estimators'], obj=self.quantile_loss(q))
                    self.models[(horizon, q)] = model
                    quantile_predictions[q] = model.predict(dtest)
                except Exception as e:
                    print(f"⚠️ Error training model for t+{horizon}h, q={q}: {e}")
                    continue

            if all(q in quantile_predictions for q in self.quantiles):
                for idx, target_time in enumerate(y_test.index):
                    predictions_all.append({'window_size': window_size_str, 'target_time': target_time,
                                            'horizon': horizon, 'actual': y_test.iloc[idx],
                                            'q10': quantile_predictions[0.1][idx],
                                            'q50': quantile_predictions[0.5][idx],
                                            'q90': quantile_predictions[0.9][idx]})

        return pd.DataFrame(predictions_all)

    def evaluate_predictions(self, results_df, horizon):
        df = results_df[results_df['horizon'] == horizon].copy()
        if df.empty:
            print(f"No results to evaluate for t+{horizon}h.")
            return

        metrics = calculate_metrics(df['actual'], df['q50'])
        print(f"\n\U0001F4CA Evaluation for t+{horizon}h horizon:")
        print(f"Number of predictions: {len(df)}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R2: {metrics['R2']:.4f}")

        coverage = np.mean((df['actual'] >= df['q10']) & (df['actual'] <= df['q90'])) * 100
        print(f"80% Prediction Interval Coverage: {coverage:.1f}%")

    def plot_predictions(self, results_df, horizon):
        df = results_df[results_df['horizon'] == horizon].copy()
        if df.empty:
            return

        df = df.sort_values('target_time')

        plt.figure(figsize=(15, 6))
        plt.fill_between(df['target_time'], df['q10'], df['q90'], alpha=0.3, label='80% Prediction Interval')
        plt.plot(df['target_time'], df['q50'], linestyle='--', label='Median Prediction')
        plt.plot(df['target_time'], df['actual'], color='black', label='Actual')

        plt.title(f'Price Predictions with Uncertainty Bands (t+{horizon}h)')
        plt.xlabel('Time')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        output_dir = os.path.join(os.path.dirname(__file__), 'plots', 'quantile')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'quantile_predictions_t{horizon}.png'))
        plt.close()



def main():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    df = pd.read_csv(features_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Define train-test split
    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    
    train_df = df[train_start:train_end]
    test_df = df[test_start:test_end]
    
    # Initialize and train model
    model = XGBoostQuantileForecaster()
    results_df = model.train_and_predict(train_df, test_df, '12m')
    
    
    # Plot and evaluate results for each horizon
    for horizon in model.horizons:
        model.plot_predictions(results_df, horizon)
        model.evaluate_predictions(results_df, horizon)

if __name__ == "__main__":
    main()
