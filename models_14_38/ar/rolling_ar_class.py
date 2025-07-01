




"""
Rolling window AR model for medium to long-term energy price forecasting.
Uses the previous 71 hours of prices to predict future prices.
Training window rolls forward each day to maintain recent data relevance.
Train window (e.g., last 365 days)
       ↓
At forecast_start (e.g., 2024-01-03 08:00)
       ↓
Fit AR(71) on last 365 days
       ↓
Predict t+14h, t+24h, t+38h recursively
       ↓
Save predicted vs actual
       ↓
Next day (move forecast_start 1 day forward)
       ↓
Repeat

what we see here is that the model actually kind of reproduces the actual data, but with a lag in time. This is quite expectable, since AR with 71 lags, will probably overfit completely, thus for 14 hours ahead, it will think that the price will behave exactly the same.
There is also a lung box test, to test for white noiseness of the errors. here we clearly see that the errors are not white noise, but they are autocorrelated.
Even though the model lag number was selected with AIC in simple_arp_recursive.py.

Metrics for 365D window:
  Horizon t+14h:
    MAE: 11.93
    RMSE: 20.24
    SMAPE: 41.70
    WMAPE: 25.15
    R2: 0.51
  Horizon t+24h:
    MAE: 15.56
    RMSE: 25.02
    SMAPE: 23.15
    WMAPE: 22.33
    R2: -1.52
  Horizon t+38h:
    MAE: 21.01
    RMSE: 30.16
    SMAPE: 58.23
    WMAPE: 42.01
    R2: -0.07

"""
import pandas as pd
import numpy as np
import sys
import time
import os
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.ar_model import ar_select_order  # ✅ correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_predictions, rolling_window_evaluation


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from utils.utils import calculate_metrics

class ARRollingForecaster:
    def __init__(self, data, horizons=[14, 24, 38], window_size='365D', max_lags=72):
        self.data = data.sort_index()
        self.horizons = horizons
        self.window_size = window_size
        self.max_lags = max_lags
        self.results_df = pd.DataFrame()

    def forecast_day(self, forecast_start):
        window_start = forecast_start - pd.Timedelta(self.window_size)
        history = self.data[(self.data.index >= window_start) &
                            (self.data.index < forecast_start)]['current_price']
        history = history.asfreq('h')

        if len(history) < self.max_lags * 2:
            return {}

        sel = ar_select_order(history, maxlag=self.max_lags, ic='aic', old_names=False)
        selected_lags = sel.ar_lags or [1]

        print(f"\nBest model for window starting at {window_start}:")
        print(f"Selected lags via AIC: {selected_lags}")

        model = AutoReg(history, lags=selected_lags, old_names=False)
        model_fit = model.fit()
        params = model_fit.params

        # Print significant lags (p < 0.05)
        significant_lags = [lag for i, lag in enumerate(selected_lags)
                            if model_fit.pvalues[i + 1] < 0.05]
        print(f"Significant lags (p < 0.05): {significant_lags}")

        predictions = {}
        last_values = history.iloc[-max(selected_lags):].values
        max_horizon = max(self.horizons)

        for step in range(1, max_horizon + 1):
            next_pred = params.iloc[0]
            for i, lag in enumerate(selected_lags):
                next_pred += params.iloc[i + 1] * last_values[-lag]

            last_values = np.append(last_values[1:], next_pred)

            if step in self.horizons:
                target_time = forecast_start + pd.Timedelta(hours=step)
                predictions[target_time] = next_pred

        return predictions

    def run_forecast(self, test_start, step='1D', forecast_hour=12):
        train_data = self.data[self.data.index < test_start]
        test_data = self.data[self.data.index >= test_start]

        window_predictions = []

        start = test_data.index.min()
        end = test_data.index.max()

        # Force consistent timezone (same as full dataset)
        tz = getattr(self.data.index, 'tz', None) or 'UTC'

        for day in pd.date_range(start=start, end=end, freq=step):
            timestamp = pd.Timestamp(day).replace(hour=forecast_hour)

            if timestamp.tzinfo:
                forecast_start = timestamp.tz_convert(tz) if timestamp.tzinfo != tz else timestamp
            else:
                forecast_start = timestamp.tz_localize(tz)

            if forecast_start in test_data.index:
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

    def evaluate_and_plot(self, save_path='models_14_38/ar/plots/plots_ar_rolling'):
        os.makedirs(save_path, exist_ok=True)
        for horizon in self.results_df['horizon'].unique():
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

    print(f"\n✅ Completed AR  in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
