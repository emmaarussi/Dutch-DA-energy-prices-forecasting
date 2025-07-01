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
from statsmodels.graphics.tsaplots import plot_acf
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order  # 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance


# --- Rolling AR Forecasting
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

        # Clean history
        history = history.replace([np.inf, -np.inf], np.nan).dropna()
        if len(history) < self.max_lags * 2:
            return {}

        # --- Custom sparse lag selection based on BIC threshold
        try:
            initial_model = AutoReg(history, lags=self.max_lags, old_names=False)
            initial_fit = initial_model.fit()
        except Exception as e:
            print(f"[{forecast_start}] Initial AR fit failed: {e}")
            return {}

        n = len(history)
        bic_threshold = np.log(n)
        selected_lags = []

        for i, (coef, std_err) in enumerate(zip(initial_fit.params[1:], initial_fit.bse[1:])):
            t_stat = coef / std_err
            bic_contribution = t_stat**2 - bic_threshold
            if bic_contribution > 0:
                selected_lags.append(i + 1)

        if not selected_lags:
            selected_lags = [1]

        print(f"\nBest model for window starting at {window_start}:")
        print(f"Selected sparse lags via BIC penalty: {selected_lags}")

        # --- Final AR fit with selected sparse lags
        try:
            model = AutoReg(history, lags=selected_lags, old_names=False)
            model_fit = model.fit()
            params = model_fit.params
        except Exception as e:
            print(f"[{forecast_start}] Final AR refit failed: {e}")
            return {}

        # Forecast generation
        predictions = {}
        last_values = history.iloc[-max(selected_lags):].values
        max_horizon = max(self.horizons)

        for step in range(1, max_horizon + 1):
            next_pred = params.iloc[0]  # intercept
            for i, lag in enumerate(selected_lags):
                if lag <= len(last_values):
                    next_pred += params.iloc[i + 1] * last_values[-lag]

            last_values = np.append(last_values[1:], next_pred)

            if step in self.horizons:
                target_time = forecast_start + pd.Timedelta(hours=step)
                predictions[target_time] = next_pred

        return predictions


    def run_forecast(self, test_start):
        train_data = self.data[self.data.index < test_start]
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
                    residual = actual_price - pred_price
                    window_predictions.append({
                        'window_size': self.window_size,
                        'forecast_start': forecast_start,
                        'target_time': target_time,
                        'horizon': (target_time - forecast_start).total_seconds() / 3600,
                        'predicted': pred_price,
                        'actual': actual_price,
                        'residual': residual
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

            # Residuals plot
            plt.figure(figsize=(15, 6))
            plt.plot(subset['target_time'], subset['residual'], label='Residuals', alpha=0.7)
            plt.title(f'Residuals (AR, {self.window_size} window, t+{int(horizon)}h)')
            plt.xlabel('Date')
            plt.ylabel('Residuals')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}/residuals_{self.window_size}_{int(horizon)}h.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Residuals autocorrelation plot
            plot_acf(subset['residual'], lags=48)
            plt.title(f'Residuals Autocorrelation (AR, {self.window_size} window, t+{int(horizon)}h)')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}/residuals_acf_{self.window_size}_{int(horizon)}h.png',
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
    forecaster.evaluate_and_plot(save_path='models_14_38/ar/plots/ARonlypointeachhour')
    
    forecaster.results_df.to_csv('models_14_38/ar/plots/ARonlypointeachhour/full_predictions.csv', index=False)


    print(f"\n✅ Completed AR  in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
