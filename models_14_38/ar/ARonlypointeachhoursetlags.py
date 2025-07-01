"""

Metrics for horizon t+14h:
  MAE: 12.62
  RMSE: 17.72
  SMAPE: 19.39
  WMAPE: 16.77
  R2: 0.32

Metrics for horizon t+24h:
  MAE: 14.31
  RMSE: 19.49
  SMAPE: 33.87
  WMAPE: 25.47
  R2: 0.22

Metrics for horizon t+38h:
  MAE: 14.30
  RMSE: 18.40
  SMAPE: 19.34
  WMAPE: 18.69
  R2: 0.10

✅ Completed AR  in 4.96 seconds

"""
# --- Imports
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from statsmodels.api import OLS, add_constant

# Ensure access to project root utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics


# --- Rolling AR Forecasting
class ARRollingForecaster:
    def __init__(self, data, horizons=[14, 24, 38], window_size='365D', max_lags=72):
        self.data = data.sort_index()
        self.horizons = horizons
        self.window_size = window_size
        self.max_lags = max_lags
        self.results_df = pd.DataFrame()

    def forecast_day(self, forecast_start):
        train_data = self.data[self.data.index < forecast_start]
        history = train_data['current_price']
        predictions = {}

        for h in self.horizons:
            window_end = forecast_start
            window_start = window_end - pd.Timedelta(self.window_size)
            window_data = history[(history.index >= window_start) & (history.index < window_end)].copy()
            window_data = window_data.asfreq('h')

            if len(window_data) < 500:
                print(f"Skipping horizon t+{h}: not enough data in window.")
                continue

            X_list = []
            y_list = []

            for t in window_data.index:
                target_time = t + pd.Timedelta(hours=h)
                if target_time not in history.index:
                    continue

                lag_values = []
                lags_ok = True

                for lag in range(1, 25):  # hourly lags
                    lag_time = t - pd.Timedelta(hours=lag)
                    if lag_time in history.index:
                        lag_values.append(history[lag_time])
                    else:
                        lags_ok = False
                        break

                for extra_lag in [48, 168]:  # daily/weekly lags
                    lag_time = t - pd.Timedelta(hours=extra_lag)
                    if lag_time in history.index:
                        lag_values.append(history[lag_time])
                    else:
                        lags_ok = False

                if lags_ok:
                    X_list.append(lag_values)
                    y_list.append(history[target_time])

            if len(X_list) < 30:
                print(f"Skipping horizon t+{h}: not enough training samples.")
                continue

            X = add_constant(np.array(X_list), has_constant='add')
            y = np.array(y_list)
            model = OLS(y, X).fit()

            # Forecast from current forecast_start time
            lag_values = []
            valid = True

            for lag in range(1, 25):
                lag_time = forecast_start - pd.Timedelta(hours=lag)
                if lag_time in history.index:
                    lag_values.append(history[lag_time])
                else:
                    valid = False
                    break

            for extra_lag in [48, 168]:
                lag_time = forecast_start - pd.Timedelta(hours=extra_lag)
                if lag_time in history.index:
                    lag_values.append(history[lag_time])
                else:
                    valid = False

            if not valid:
                print(f"Skipping forecast for t+{h}: missing lag values.")
                continue

            X_pred = add_constant(np.array([lag_values]), has_constant='add')
            y_pred = model.predict(X_pred)[0]
            forecast_time = forecast_start + pd.Timedelta(hours=h)
            predictions[forecast_time] = y_pred

        return predictions

    def run_forecast(self, test_start):
        test_data = self.data[self.data.index >= test_start]
        window_predictions = []

        start = test_data.index.min()
        end = test_data.index.max()
        tz = getattr(self.data.index, 'tz', None) or 'UTC'

        for timestamp in pd.date_range(start=start, end=end, freq='1D'):
            forecast_start = timestamp.tz_localize(tz) if timestamp.tzinfo is None else timestamp
            predictions = self.forecast_day(forecast_start)

            for target_time, pred_price in predictions.items():
                if target_time in test_data.index:
                    actual_price = test_data.loc[target_time, 'current_price']
                    window_predictions.append({
                        'window_size': self.window_size,
                        'forecast_start': forecast_start,
                        'target_time': target_time,
                        'horizon': (target_time - forecast_start).total_seconds() / 3600,
                        'predicted': pred_price,
                        'actual': actual_price
                    })

        self.results_df = pd.DataFrame(window_predictions)

    def evaluate_and_plot(self, save_path='models_14_38/ar/plots/plots_arsetlags_rolling'):
        os.makedirs(save_path, exist_ok=True)
        for horizon in sorted(self.results_df['horizon'].unique()):
            subset = self.results_df[self.results_df['horizon'] == horizon]
            metrics = calculate_metrics(subset['actual'], subset['predicted'])
            print(f"\nMetrics for horizon t+{int(horizon)}h:")
            for key, val in metrics.items():
                print(f"  {key}: {val:.2f}")

            # Plotting
            plt.figure(figsize=(15, 6))
            plt.plot(subset['target_time'], subset['actual'], label='Actual', alpha=0.7)
            plt.plot(subset['target_time'], subset['predicted'], label='Predicted', alpha=0.7)
            plt.title(f'Actual vs Predicted Prices (AR, {self.window_size} window, t+{int(horizon)}h)')
            plt.xlabel('Date')
            plt.ylabel('Price (EUR/MWh)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}/predictions_over_time_{self.window_size}_{int(horizon)}h.png',
                        dpi=300, bbox_inches='tight')
            plt.close()


def main():
    start_time = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)

    print(f"Full data range: {data.index.min()} to {data.index.max()}")
    test_start = pd.Timestamp('2024-01-01', tz=data.index.tz)

    forecaster = ARRollingForecaster(data, window_size='365D', horizons=[14, 24, 38])
    forecaster.run_forecast(test_start)
    forecaster.evaluate_and_plot()

    print(f"\n✅ Completed AR in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
