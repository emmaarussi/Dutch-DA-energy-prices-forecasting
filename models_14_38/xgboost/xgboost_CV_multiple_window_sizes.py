"""
XGBoost model for medium to long-term energy price forecasting using new feature set called multivariate_features.csv.

It uses the utils.utils module for utility functions calculate_metrics, plot_feature_importance, plot_predictions, plot_error_distribution, rolling_window_evaluation

Features:
1. Historical price data
2. Wind generation (total)
3. Solar generation
4. Consumption
5. Calendar features

The model uses rolling windows for training, shifted by 1 week, evaluating on horizons from t+14 to t+38.

"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import matplotlib.pyplot as plt
from datetime import timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

class XGBoostMultivariate:
    def __init__(self, horizons=[14, 24, 38]):
        self.horizons = horizons

    def prepare_data(self, data, horizon):
        feature_cols = [col for col in data.columns if not col.startswith('target_t') and col != 'current_price']
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y

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

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    test_start = pd.Timestamp('2024-01-01', tz='UTC')
    test_data = data[data.index >= test_start]

    print(f"Full data range: {data.index.min()} to {data.index.max()}")
    print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

    model = XGBoostMultivariate()
    window_sizes = ['30D', '90D', '180D', '365D']
    test_window_length = pd.Timedelta(days=30)


    for window_size_str in window_sizes:
        print(f"\nEvaluating with {window_size_str} window:")
        window_size = pd.Timedelta(window_size_str)
        predictions_all = []

        forecast_starts = pd.date_range(
            start=test_data.index.min(), 
            end=test_data.index.max() - test_window_length,  # <-- ensures full test window fits
            freq='7D'
        )

        for forecast_start in forecast_starts:
            forecast_start = forecast_start.replace(hour=12)
            if forecast_start not in data.index:
                continue

            window_start = forecast_start - window_size
            train_window = data[(data.index >= window_start) & (data.index < forecast_start)]
            test_window = data[(data.index >= forecast_start) & (data.index < forecast_start + test_window_length)]

            for horizon in model.horizons:
                X_train, y_train = model.prepare_data(train_window, horizon)
                X_test, y_test = model.prepare_data(test_window, horizon)

                if len(X_train) < 100 or len(X_test) == 0:
                    continue

                model_xgb = xgb.XGBRegressor(**model.get_hyperparameters(horizon))
                model_xgb.fit(X_train, y_train)
                y_pred = model_xgb.predict(X_test)

                print(f"[{forecast_start}] Horizon t+{horizon}h — Predictions: {len(y_pred)}")

                for idx, target_time in enumerate(y_test.index):
                    predictions_all.append({
                        'window_size': window_size_str,
                        'forecast_start': forecast_start,
                        'target_time': target_time,
                        'horizon': horizon,
                        'predicted': y_pred[idx],
                        'actual': y_test.iloc[idx]
                    })

        # Convert to DataFrame
        results_df = pd.DataFrame(predictions_all)

        # Plot predictions
        for horizon in model.horizons:
            horizon_df = results_df[results_df['horizon'] == horizon]

            if horizon_df.empty:
                continue

            metrics = calculate_metrics(horizon_df['actual'], horizon_df['predicted'])
            print(f"\nt+{horizon}h horizon:")
            print(f"Number of predictions: {len(horizon_df)}")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"SMAPE: {metrics['SMAPE']:.2f}%")
            print(f"R2: {metrics['R2']:.4f}")

            plt.figure(figsize=(15, 6))
            plt.plot(horizon_df['target_time'], horizon_df['actual'], label='Actual', alpha=0.7)
            plt.plot(horizon_df['target_time'], horizon_df['predicted'], label='Predicted', alpha=0.7)
            plt.title(f'XGBoost: Actual vs Predicted Prices Over Time (t+{horizon}h, {window_size_str} rolling)')
            plt.xlabel('Date')
            plt.ylabel('Price (EUR/MWh)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()


            out_dir = f'models_14_38/xgboost/plots/rolling_{window_size_str}'
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f'{out_dir}/predictions_over_time_h{horizon}.png', dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()