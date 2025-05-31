"""
XGBoost model for medium to long-term energy price forecasting using only price and calendar features.
Excludes all features related to wind, coal, solar, and consumption.
t+14h horizon:
Number of predictions: 730
RMSE: 17.23
SMAPE: 28.35%
R2: 0.2899

t+24h horizon:
Number of predictions: 730
RMSE: 17.99
SMAPE: 29.97%
R2: 0.1996

t+38h horizon:
Number of predictions: 730
RMSE: 16.08
SMAPE: 24.55%
R2: 0.3571
Exit Code 0
It uses the utils.utils module for utility functions calculate_metrics, plot_feature_importance.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance

class XGBoostclean:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        
    def prepare_data(self, data, horizon):
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
        feature_cols = [col for col in data.columns 
                       if not col.startswith('target_t') and 
                       not any(pattern in col for pattern in excluded_patterns)]
        
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y
        
    def train_and_evaluate(self, train_data, test_data, horizon):
        """Train and evaluate model for a specific window and horizon"""
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, horizon)
        X_test, y_test = self.prepare_data(test_data, horizon)
        
        # Get hyperparameters and train model
        params = self.get_hyperparameters(horizon)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics using utils function
        metrics = calculate_metrics(y_test, predictions)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions
        }, index=test_data.index)

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': importance,
            'predictions': predictions_df
        }
        
    def get_hyperparameters(self, horizon):
        if horizon <= 24:
            return {
                'max_depth': 8, 'learning_rate': 0.0110, 'n_estimators': 1300,
                'min_child_weight': 3.1910, 'subsample': 0.8837, 'colsample_bytree': 0.8369,
                'gamma': 0.1393, 'random_state': 42
            }
        elif horizon <= 31:
            return {
                'max_depth': 9, 'learning_rate': 0.0693, 'n_estimators': 1400,
                'min_child_weight': 1.4524, 'subsample': 0.7879, 'colsample_bytree': 0.8769,
                'gamma': 0.1656, 'random_state': 42
            }
        else:  # for t+38h
            return {
                'max_depth': 9, 'learning_rate': 0.0179, 'n_estimators': 1100,
                'min_child_weight': 5.4722, 'subsample': 0.8081, 'colsample_bytree': 0.8531,
                'gamma': 0.2176, 'random_state': 42
            }

def main():
    print("Starting main function...")

    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    # Define forecast horizons
    horizons = [14, 24, 38]

    # Create safe target columns without leakage
    for h in horizons:
        data[f'target_t{h}'] = data['current_price'].shift(-h)

    # Drop rows that would leak future info (i.e., targets that aren't known yet)
    data.dropna(subset=[f'target_t{h}' for h in horizons], inplace=True)

    # Split into training and test data
    test_start = '2024-01-01'
    train_data = data[data.index < test_start]
    test_data = data[data.index >= test_start]

    print(f"Full data range: {data.index.min()} to {data.index.max()}")
    print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

    window_sizes = ['365D']

    for window_size in window_sizes:
        print(f"\nEvaluating with {window_size} window:")
        window_predictions = []

        for day in pd.date_range(test_data.index.min(), test_data.index.max(), freq='7D'):
            forecast_start = pd.Timestamp(day.date()).replace(hour=12, tzinfo=test_data.index.tzinfo)

            if forecast_start in test_data.index:
                window_end = forecast_start
                window_start = forecast_start - pd.Timedelta(window_size)
                train_window = data[(data.index >= window_start) & (data.index < window_end)]
                test_window = data[(data.index >= forecast_start)]

                model = XGBoostclean(horizons=horizons)

                for h in horizons:
                    result = model.train_and_evaluate(train_window, test_window, h)
                    pred_df = result['predictions'].copy()
                    pred_df['horizon'] = h
                    pred_df['forecast_start'] = forecast_start
                    pred_df = pred_df.reset_index().rename(columns={'index': 'target_time'})
                    for _, row in pred_df.iterrows():
                        window_predictions.append({
                            'window_size': window_size,
                            'forecast_start': forecast_start,
                            'target_time': row['target_time'],
                            'horizon': row['horizon'],
                            'predicted': row['predicted'],
                            'actual': row['actual']
                        })

        results_df = pd.DataFrame(window_predictions)
        print(f"\nMetrics for {window_size} window:")
        for horizon in results_df['horizon'].unique():
            subset = results_df[results_df['horizon'] == horizon]
            metrics = calculate_metrics(subset['actual'], subset['predicted'])
            print(f"  Horizon t+{int(horizon)}h:")
            for key, val in metrics.items():
                print(f"    {key}: {val:.2f}")

        # Plotting
        for h in horizons:
            h_df = results_df[results_df['horizon'] == h]
            plt.figure(figsize=(15, 6))
            plt.plot(h_df['target_time'], h_df['actual'], label='Actual', alpha=0.7)
            plt.plot(h_df['target_time'], h_df['predicted'], label='Predicted', alpha=0.7)
            plt.title(f'Actual vs Predicted Prices Over Time (XGBoost, {window_size} window, t+{h}h)')
            plt.xlabel('Date')
            plt.ylabel('Price (EUR/MWh)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            os.makedirs('models_14_38/xgboost/XGboost_training_features_hyperparameters_cv/plots', exist_ok=True)
            plt.savefig(f'models_14_38/xgboost/XGboost_training_features_hyperparameters_cv/plots/predictions_over_time_{window_size}_{h}h.png', dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    main()
