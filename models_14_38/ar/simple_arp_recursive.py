"""
Simple AR(p) model for medium to long-term energy price forecasting.
Uses only the previous price values to predict future prices.

Train window (e.g., full train period 365 days)
       ↓
Fit AR(p) on full train period, select lags with AIC
       ↓
Predict t+14h, t+24h, t+38h recursively
       ↓
Save predicted vs actual
       ↓
Next day (move forecast_start 1 day forward)
       ↓
Repeat

The model uses 12-month rolling windows for training,
evaluating on horizons from t+14 to t+38.

what we see here is that the forecast produces a flat line. This is because we are not updating the model parameters after each prediction.
This is a clear difference with the other ar models made is this study. Those are more responsive and adaptive to changes in the data.
What we take from this is that there is a clear gain in updating model over time, also when applying XGboost. However, updating too often (daily) is already a heavy task for a simple regression, let alone a more complex model like XGboost.
Finding the balance between updating frequency, window sizes and model complexity is key.
"""
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf


def forecast_day(train_data, forecast_start, horizons=[14, 24, 38]):
    """Forecast using AutoReg with automatic lag selection"""
    history = train_data[train_data.index < forecast_start]['price_eur_per_mwh']
    history = history.asfreq('h')

    # --- Auto lag selection ---
    best_lag = 24  # Default
    try:
        temp_model = AutoReg(history, lags=np.arange(1, 72), old_names=False)
        temp_fit = temp_model.fit()
        best_lag = max(temp_fit.model._lags)
        print(f"Selected lag at {forecast_start}: {best_lag}")
    except Exception as e:
        print(f"Auto lag selection failed, using default lag=24. Error: {e}")

    # --- Fit final model with selected lag ---
    model = AutoReg(history, lags=best_lag, old_names=False)
    model_fit = model.fit()

    # Get fitted model lags
    selected_lags = model_fit.model._lags  # ← Important: dynamic lags
    if isinstance(selected_lags, (list, np.ndarray)):
        selected_lags = np.array(selected_lags)
    else:
        selected_lags = np.array([selected_lags])  # In case it's a single lag

    # Make predictions recursively
    predictions = {}
    last_values = history.iloc[-max(selected_lags):].values  # Keep enough history

    max_horizon = max(horizons)
    all_preds = []

    for step in range(1, max_horizon + 1):
        # Predict next value
        next_pred = model_fit.params.iloc[0]  # Intercept

        for idx, lag in enumerate(selected_lags):
            if lag <= len(last_values):
                next_pred += model_fit.params.iloc[idx+1] * last_values[-lag]

        all_preds.append(next_pred)

        # Update history with prediction
        last_values = np.append(last_values, next_pred)

        if step in horizons:
            target_time = forecast_start + pd.Timedelta(hours=step)
            predictions[target_time] = next_pred

    return predictions


def main():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    # Split into training (up to Jan 2024) and test data
    train_data = data[data.index < '2024-01-01']
    test_data = data[data.index >= '2024-01-01']

    print(f"Training data: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Test data: {test_data.index.min()} to {test_data.index.max()}")

    # For each day at 12:00 in test period
    all_predictions = []
    horizons = [14, 24, 38]

    for day in pd.date_range(test_data.index.min(), test_data.index.max(), freq='D'):
        forecast_start = pd.Timestamp(day.date()).replace(hour=12, tzinfo=test_data.index.tzinfo)
        
        if forecast_start in test_data.index:
            # Make predictions for this day
            predictions = forecast_day(train_data, forecast_start, horizons)
            
            # Record predictions with their actual values
            for target_time, pred_price in predictions.items():
                if target_time in test_data.index:
                    actual_price = test_data.loc[target_time, 'price_eur_per_mwh']
                    all_predictions.append({
                        'forecast_start': forecast_start,
                        'target_time': target_time,
                        'horizon': (target_time - forecast_start).total_seconds() / 3600,
                        'predicted': pred_price,
                        'actual': actual_price
                    })

    # Convert to DataFrame and calculate metrics
    results_df = pd.DataFrame(all_predictions)
    
    # Calculate metrics for each horizon
    print("\nResults by horizon:")
    for horizon in horizons:
        horizon_results = results_df[results_df['horizon'] == horizon]
        metrics = calculate_metrics(horizon_results['actual'], horizon_results['predicted'])
        
        print(f"\nt+{horizon}h horizon:")
        print(f"Number of predictions: {len(horizon_results)}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"Mean actual price: {horizon_results['actual'].mean():.2f}")
        print(f"Mean predicted price: {horizon_results['predicted'].mean():.2f}")

        # Plot predictions vs actuals
        plt.figure(figsize=(15, 6))
        plt.plot(horizon_results['target_time'], horizon_results['actual'], label='Actual', alpha=0.7)
        plt.plot(horizon_results['target_time'], horizon_results['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'AR(p) Model: Actual vs Predicted Prices (t+{horizon}h)')
        plt.xlabel('Target Time')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs('models_14_38/ar/plots_arp_recursive', exist_ok=True)
        plt.savefig(f'models_14_38/ar/plots_arp_recursive/predictions_h{horizon}.png')
        plt.close()
            
        # Print horizon-specific stats
        print(f"\nPrediction stats for horizon t+{horizon}h:")
        print(f"Number of valid predictions: {len(horizon_results)}")
        print(f"Mean actual price: {horizon_results['actual'].mean():.2f}")
        print(f"Mean predicted price: {horizon_results['predicted'].mean():.2f}")

        #check stationarity traindata
        result = adfuller(train_data['current_price'].dropna())
        print(f"ADF Statistic train data: {result[0]}")
        print(f"p-value train data: {result[1]}")
        
        if result[1] < 0.05:
            print("Series is stationary.")
        else:
            print("Series is non-stationary.")

        #check stationarity testdata
        result = adfuller(test_data['price_eur_per_mwh'].dropna())
        print(f"ADF Statistic test data: {result[0]}")
        print(f"p-value test data: {result[1]}")
        
        if result[1] < 0.05:
            print("Series is stationary.")
        else:
            print("Series is non-stationary.")

        #check acf traindata
        plot_acf(train_data['price_eur_per_mwh'].dropna(), lags=100)
        plt.show()

        #check acf testdata
        plot_acf(test_data['price_eur_per_mwh'].dropna(), lags=100)
        plt.show()

        # Check Ljung-Box test for residuals
        lb_test = acorr_ljungbox(test_data['price_eur_per_mwh'].dropna(), lags=71)
        print(f"\nLjung-Box test results for horizon {horizon}:")
        print("H0: The residuals are independently distributed")
        print("H1: The residuals are not independently distributed")
        print("\nTest statistics:")
        print(lb_test)
        

if __name__ == "__main__":
    main()





