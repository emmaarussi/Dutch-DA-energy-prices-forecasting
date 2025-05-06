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


"""
import pandas as pd
import numpy as np
import sys
import os
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.api import OLS, add_constant
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_predictions, rolling_window_evaluation

def forecast_day(train_data, forecast_start, window_size='365D', horizons=[14, 24, 38], lags=71):
    """Make price forecasts for a single day using AR(71) model with rolling window"""
    # Get training history up to forecast_start, but only use the last window_size of data
    window_start = forecast_start - pd.Timedelta(window_size)
    history = train_data[(train_data.index >= window_start) & 
                        (train_data.index < forecast_start)]['current_price']
    history = history.asfreq('h')
    if len(history) < lags * 2:  # Need at least 2*lags points to train
        return {}
    
    # Create lagged features
    X_list = []
    y_list = []
    
    for i in range(len(history) - lags):
        x_i = history.iloc[i:i+lags].values
        y_i = history.iloc[i+lags]
        X_list.append(x_i)
        y_list.append(y_i)
    
    X = np.array(X_list)
    y = np.array(y_list)
    X = add_constant(X)

    # Fit OLS model on rolling window
    model = AutoReg(history, lags=lags, old_names=False)
    model_fit = model.fit()

    params = model_fit.params

    # Make predictions for each horizon
    predictions = {}
    last_values = history.iloc[-lags:].values
    
    # Predict recursively for all horizons
    max_horizon = max(horizons)
    all_preds = []
    
    for step in range(1, max_horizon + 1):
        # Make prediction using last lags values
        next_pred = params.iloc[0]  # intercept
        for i in range(lags):
            next_pred += params.iloc[i+1] * last_values[i]
        
        all_preds.append(next_pred)
        
        # Update last_values array: remove oldest, add new prediction
        last_values = np.roll(last_values, -1)
        last_values[-1] = next_pred
        
        if step in horizons:
            target_time = forecast_start + pd.Timedelta(hours=step)
            predictions[target_time] = next_pred
    
    # Perform residual analysis
    residuals = model_fit.resid
    
    # Perform Ljung-Box test for residuals up to lag 71
    ##lb_test = acorr_ljungbox(residuals, lags=[71], return_df=True)
    ## print(f"\nLjung-Box test for window starting at {window_start}:")
    ##print(lb_test)

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
    window_sizes = ['30D', '90D', '180D', '365D']

    for window_size in window_sizes:
        print(f"\nEvaluating with {window_size} window:")
        window_predictions = []

        for day in pd.date_range(test_data.index.min(), test_data.index.max(), freq='D'):
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

        # Calculate metrics for this window size
        results_df = pd.DataFrame(window_predictions)
        
       # Plot AR predictions just like XGBoost
        plt.figure(figsize=(15, 6))
        plt.plot(results_df['target_time'], results_df['actual'], label='Actual', alpha=0.7)
        plt.plot(results_df['target_time'], results_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (AR, {window_size} window, t+{horizons[0]}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save to the same kind of directory structure
        os.makedirs('models_14_38/ar/plots/plots_ar_rolling', exist_ok=True)
        plt.savefig(f'models_14_38/ar/plots/plots_ar_rolling/predictions_over_time_{window_size}_{horizons[0]}h.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    os.makedirs('models_14_38/ar/plots_ar_rolling', exist_ok=True)
    main()


