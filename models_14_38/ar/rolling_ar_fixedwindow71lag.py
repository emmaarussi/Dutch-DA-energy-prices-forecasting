###
"""
Simple linear model with lags 
Uses only the previous price values to predict future prices.

Train window (e.g., full train period 365 days)
       ↓
Fit AR(p) on full train period, select lags with AIC
       ↓
Predict t+14h, seperately
       ↓
Save predicted vs actual
       ↓
Next day (move forecast_start 1 day forward)
       ↓
Repeat

The model uses 14-month rolling windows for training,
evaluating on horizons from t+14
###
"""


import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance, rolling_window_evaluation


class RollingARModel:
    def __init__(self, lags=71):
        self.lags = lags
        
    def make_single_forecast(self, history, horizon=14):
        """Make a single h-step ahead forecast using data up to current point"""
        model = AutoReg(history, lags=self.lags)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)
        return forecast.iloc[-1]  # Return only the h-step ahead forecast

    def walk_forward_validation(self, data, train_end, test_end, horizon=14):
        """Perform walk-forward validation"""
        # Initialize results
        predictions = []
        actuals = []
        timestamps = []
    
        if isinstance(data, pd.DataFrame):
            data = data['price_eur_per_mwh']
        
        test_start = train_end
        current_time = test_start
        safe_test_end = test_end - pd.Timedelta(hours=horizon)
        
        total_steps = int((safe_test_end - current_time).total_seconds() / 3600)
        pbar = tqdm(total=total_steps, desc=f'Making t+{horizon}h forecasts')

        while current_time <= safe_test_end:
            history = data[data.index < current_time]
            actual_idx = current_time + pd.Timedelta(hours=horizon)

            if actual_idx not in data.index:
                current_time += pd.Timedelta(hours=1)
                pbar.update(1)
                continue

            actual = data[actual_idx]
            prediction = self.make_single_forecast(history, horizon)

            predictions.append(prediction)
            actuals.append(actual)
            timestamps.append(actual_idx)

            current_time += pd.Timedelta(hours=1)
            pbar.update(1)

            # Debugging info
            progress = len(predictions) / total_steps * 100
            if 92 <= progress <= 94:
                print(f"\nDebug at {progress:.1f}%:")
                print(f"Current time: {current_time}")
                print(f"Actual index: {actual_idx}")
                print(f"Predictions made: {len(predictions)}")

        # After while-loop: create result DataFrame
        results = pd.DataFrame({
            'timestamp': timestamps,
            'actual': actuals,
            'predicted': predictions
        })
        results.set_index('timestamp', inplace=True)

        return results

    def plot_predictions(self, results, horizon):
        """Plot actual vs predicted values for the test period"""
        plt.figure(figsize=(15, 6))
        plt.plot(results.index, results['actual'], label='Actual', color='black', linewidth=2)
        plt.plot(results.index, results['predicted'], 
                label=f'{horizon}-hour Forecast', color='#e74c3c', 
                linestyle='--', alpha=0.8)
        
        plt.title(f'Rolling AR Model: {horizon}-hour Ahead Forecasts vs Actual Values', 
                 fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (EUR/MWh)', fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt
    
    def plot_scatter(self, results, horizon):
        """Create scatter plot of predicted vs actual values"""
        plt.figure(figsize=(10, 10))
        plt.scatter(results['actual'], results['predicted'], alpha=0.5)
        
        # Add diagonal line
        min_val = min(results['actual'].min(), results['predicted'].min())
        max_val = max(results['actual'].max(), results['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Price (EUR/MWh)')
        plt.ylabel('Predicted Price (EUR/MWh)')
        plt.title(f'Predicted vs Actual Prices for t+{horizon}')
        plt.grid(True, alpha=0.3)
        
        return plt

def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/processed/multivariate_features.csv', index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.asfreq('h')
    
    # Split into train/test
    train_end = pd.Timestamp('2024-01-29', tz='UTC')
    test_end = pd.Timestamp('2024-03-01', tz='UTC')
    
    # Initialize model
    model = RollingARModel(lags=71)  # Use 71 lags (1 day)
    
    # Perform walk-forward validation
    horizon = 14  # 14-hour ahead forecasts
    results = model.walk_forward_validation(data, train_end, test_end, horizon)
    
    # Calculate metrics
    metrics = calculate_metrics(results['actual'], results['predicted'])
    
    print("\nTest Metrics:")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2: {metrics['R2']:.2f}")
    print(f"SMAPE: {metrics['SMAPE']:.2f}%")
    
    # Create plots directory
    os.makedirs('models_14_38/ar/plots/rolling_ar_fixed71lag', exist_ok=True)
    
    # Plot time series
    plt = model.plot_predictions(results, horizon)
    plt.savefig(f'models_14_38/ar/plots/rolling_ar_fixed71lag/forecast_{horizon}h.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    main()
