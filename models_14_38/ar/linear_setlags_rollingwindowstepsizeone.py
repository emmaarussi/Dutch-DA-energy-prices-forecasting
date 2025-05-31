"""
Linear model with lagged price values respective of the forecast horizon 


- Recent hours: price_lag_1h, price_lag_2h, price_lag_3h
- Daily patterns: price_lag_24h, price_lag_48h
- Weekly pattern: price_lag_168h

Rolling forecast with a window size of 90D and horizons [14, 24, 38]
predicts step wise
 
Metrics for 365D window:
  Horizon t+14h:
    MAE: 12.94
    RMSE: 18.88
    SMAPE: 46.22
    WMAPE: 27.29
    R2: 0.57
  Horizon t+24h:
    MAE: 14.08
    RMSE: 25.93
    SMAPE: 21.77
    WMAPE: 20.21
    R2: -1.70
  Horizon t+38h:
    MAE: 20.28
    RMSE: 29.14
    SMAPE: 57.52
    WMAPE: 40.56
    R2: -0.00


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_predictions
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
##from statsmodels.stats.diagnostic import acorr_ljungbox


def SimpleARModel_fixedlags(train_data, forecast_start, window_size='365D', horizons=[14, 24, 38]):
    """
    Forecast electricity prices for multiple horizons using past lags at time t to predict price at t+h.
    Avoids lookahead bias by ensuring all lags are from before t.
    """
    predictions = {}
    history = train_data['current_price']
    
    for h in horizons:
        window_end = forecast_start
        window_start = window_end - pd.Timedelta(window_size)
        window_data = history[(history.index >= window_start) & (history.index < window_end)].copy()
        window_data = window_data.asfreq('h')

        if len(window_data) < 500:
            print(f"Skipping horizon t+{h}: not enough data in window.")
            continue

        X_list = []
        y_list = []

        for t in window_data.index:
            # t is the input time; y is at t + h
            target_time = t + pd.Timedelta(hours=h)
            if target_time not in history.index:
                continue

            lags_ok = True
            lag_values = []

            for lag in range(1, 25):  # hourly lags
                lag_time = t - pd.Timedelta(hours=lag)
                if lag_time in history.index:
                    lag_values.append(history[lag_time])
                else:
                    lags_ok = False
                    break

            for extra_lag in [48, 168]:  # additional daily/weekly lags
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
                predictions = SimpleARModel_fixedlags(data, forecast_start, window_size, horizons)
                
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
            os.makedirs('models_14_38/ar/plots/plots_ar_setlags', exist_ok=True)
            plt.savefig(f'models_14_38/ar/plots/plots_ar_setlags/predictions_over_time_{window_size}_{h}h.png', dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    os.makedirs('models_14_38/ar/plots_ar_setlags', exist_ok=True)
    main()
