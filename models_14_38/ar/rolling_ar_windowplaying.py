




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
import os
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.ar_model import ar_select_order  # ✅ correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_predictions, rolling_window_evaluation


def forecast_day(train_data, forecast_start, window_size='365D', horizons=[14, 24, 38], max_lags=72):
    """Make price forecasts for a single day using AR model with AIC-based lag subset selection."""
    # Get training history up to forecast_start, but only use the last window_size of data
    window_start = forecast_start - pd.Timedelta(window_size)
    history = train_data[(train_data.index >= window_start) & 
                         (train_data.index < forecast_start)]['current_price']
    history = history.asfreq('h')
    if len(history) < max_lags * 2:
        return {}

    # Use AIC to select best subset of lags
    sel = ar_select_order(history, maxlag=max_lags, ic='aic', old_names=False)
    selected_lags = sel.ar_lags
    if selected_lags is None or len(selected_lags) == 0:
        selected_lags = [1]  # fallback
    
    print(f"\nBest model for window starting at {window_start}:")
    print(f"Selected lags via AIC: {selected_lags}")

    model = AutoReg(history, lags=selected_lags, old_names=False)
    model_fit = model.fit()
    params = model_fit.params

    # Print significant lags (p < 0.05)
    significant_lags = [lag for i, lag in enumerate(selected_lags) 
                        if model_fit.pvalues[i+1] < 0.05]  # Skip intercept
    print(f"Significant lags (p < 0.05): {significant_lags}")

    # Recursive forecasting
    predictions = {}
    last_values = history.iloc[-max(selected_lags):].values  # Ensure full range
    max_horizon = max(horizons)

    for step in range(1, max_horizon + 1):
        next_pred = params.iloc[0]  # Intercept
        for i, lag in enumerate(selected_lags):
            next_pred += params.iloc[i+1] * last_values[-lag]

        # Update values
        last_values = np.append(last_values[1:], next_pred)

        if step in horizons:
            target_time = forecast_start + pd.Timedelta(hours=step)
            predictions[target_time] = next_pred

    print("Forecast start:", forecast_start)
    print("History ends:", history.index.max())
    print("Forecast targets:", list(predictions.keys())[:3])
    

    return predictions

def main():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    # Split into training and test data
    test_start = '2024-01-01'
    train_data = data[data.index < test_start]
    test_data = data[data.index >= test_start]

    print(f"Full data range: {data.index.min()} to {data.index.max()}")
    print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

    # For each day at 12:00 in test period
    all_predictions = []
    horizons = [14, 24, 38]
    window_sizes = ['365D']

    for window_size in window_sizes:
        print(f"\nEvaluating with {window_size} window:")
        window_predictions = []

        for day in pd.date_range(test_data.index.min(), test_data.index.max(), freq='7D'):
            forecast_start = pd.Timestamp(day.date()).replace(hour=12, tzinfo=test_data.index.tzinfo)
            
            if forecast_start in test_data.index:
                # Make predictions for this day
                predictions = forecast_day(data, forecast_start, window_size, horizons)
                
                # Record predictions with their actual values
                for target_time, pred_price in predictions.items():
                    if target_time in test_data.index:
                        actual_price = test_data.loc[target_time, 'current_price']
                        window_predictions.append({
                            'window_size': window_size,
                            'forecast_start': forecast_start,
                            'target_time': target_time,
                            'horizon': (target_time - forecast_start).total_seconds() / 3600,
                            'predicted': pred_price,
                            'actual': actual_price
                        })

        

        results_df = pd.DataFrame(window_predictions)
        print(f"\nMetrics for {window_size} window:")

        for horizon in results_df['horizon'].unique():
            subset = results_df[results_df['horizon'] == horizon]
            metrics = calculate_metrics(subset['actual'], subset['predicted'])
            print(f"  Horizon t+{int(horizon)}h:")
            for key, val in metrics.items():
                print(f"    {key}: {val:.2f}")

        
       # Plot each horizon separately
        for h in horizons:
            h_df = results_df[results_df['horizon'] == h]
            
            plt.figure(figsize=(15, 6))
            plt.plot(h_df['target_time'], h_df['actual'], label='Actual', alpha=0.7)
            plt.plot(h_df['target_time'], h_df['predicted'], label='Predicted', alpha=0.7)
            plt.title(f'Actual vs Predicted Prices Over Time (AR, {window_size} window, t+{h}h)')
            plt.xlabel('Date')
            plt.ylabel('Price (EUR/MWh)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save to the same kind of directory structure
            os.makedirs('models_14_38/ar/plots/plots_ar_rolling', exist_ok=True)
            plt.savefig(f'models_14_38/ar/plots/plots_ar_rolling/predictions_over_time_{window_size}_{h}h.png', dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    os.makedirs('models_14_38/ar/plots_ar_rolling', exist_ok=True)
    main()


