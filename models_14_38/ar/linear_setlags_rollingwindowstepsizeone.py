"""
Linear model with lagged price values respective of the forecast horizon 


- Recent hours: price_lag_1h, price_lag_2h, price_lag_3h
- Daily patterns: price_lag_24h, price_lag_48h
- Weekly pattern: price_lag_168h

Rolling forecast with a window size of 90D and horizons [14, 24, 38]
predicts step wise for each horizon 1 value at a time, then updates the window and continues. 
quite a slow model since it makes about 1400 predictions per horizon
 --
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


def SimpleARModel_fixedlags(train_data, forecast_start, window_size='90D', horizons=[14, 24, 38]):
    """
    Forecast electricity prices for multiple horizons using lagged features
    relative to the forecast target time (t + h).
    
    Lags used: 1–24 hours and 168 hours (1 week) before each forecast target time.
    """
    predictions = {}
    
    # Univariate series
    history = train_data['current_price']
    
    for h in horizons:
        forecast_time = forecast_start + pd.Timedelta(hours=h)
        window_start = forecast_time - pd.Timedelta(window_size)
        window_data = history[(history.index >= window_start) & (history.index < forecast_time)].copy()
        window_data = window_data.asfreq('h')

        if len(window_data) < 200:
            print(f"Skipping horizon t+{h}: not enough data in window.")
            continue

        X_list = []
        y_list = []

        for t in window_data.index:
            target_time = t + pd.Timedelta(hours=h)
            if target_time not in history.index:
                continue

            # Get lags relative to forecast target (t + h)
            lag_values = []
            valid = True
            for lag in range(1, 25):  # 1 to 24 hours before target
                lag_time = target_time - pd.Timedelta(hours=lag)
                if lag_time in history.index:
                    lag_values.append(history[lag_time])
                else:
                    valid = False
                    break

           # two day lag: 48h before target
            two_day_lag_time = target_time - pd.Timedelta(hours=48)
            if two_day_lag_time in history.index:
                lag_values.append(history[two_day_lag_time])
            else:
                valid = False
            
            
            # Weekly lag: 168h before target
            weekly_lag_time = target_time - pd.Timedelta(hours=168)
            if weekly_lag_time in history.index:
                lag_values.append(history[weekly_lag_time])
            else:
                valid = False

            if valid:
                X_list.append(lag_values)
                y_list.append(history[target_time])

        if len(X_list) < 30:
            print(f"Skipping horizon t+{h}: not enough training samples.")
            continue

        X = np.array(X_list)
        y = np.array(y_list)
        X = add_constant(X, has_constant="add")

        model = OLS(y, X).fit()

        # Create lagged input for forecast_time (i.e., t + h)
        forecast_lags = []
        valid = True
        for lag in range(1, 25):  # lags 1–24h before forecast target
            lag_time = forecast_time - pd.Timedelta(hours=lag)
            if lag_time in history.index:
                forecast_lags.append(history[lag_time])
            else:
                valid = False
                break

        # Add 48h lag
        two_day_lag_time = forecast_time - pd.Timedelta(hours=48)
        if two_day_lag_time in history.index:
            forecast_lags.append(history[two_day_lag_time])
        else:
            valid = False

        weekly_lag_time = forecast_time - pd.Timedelta(hours=168)
        if weekly_lag_time in history.index:
            forecast_lags.append(history[weekly_lag_time])
        else:
            valid = False

        if not valid:
            print(f"Skipping forecast for t+{h}: missing lag values.")
            continue

        X_pred = add_constant(np.array([forecast_lags]), has_constant="add")
        y_pred = model.predict(X_pred)[0]
        predictions[forecast_time] = y_pred


    return predictions


def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/processed/multivariate_features.csv', index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.asfreq('h')  # Lowercase 'h' to avoid FutureWarning

    # Define test window
    train_end = pd.Timestamp('2024-01-01', tz='UTC')
    test_end = pd.Timestamp('2024-03-01', tz='UTC')

    # Split for display/logging
    train_data = data[data.index < train_end]
    test_data = data[data.index >= train_end]

    print(f"Data shape: {data.shape}")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
    
    horizons = [14, 24, 38]
    all_predictions = {h: [] for h in horizons}

    # Rolling forecast starts after enough history is available
    rolling_start = train_end
    rolling_dates = pd.date_range(start=rolling_start, end=test_end - pd.Timedelta(hours=max(horizons)), freq='h')

    for forecast_start in tqdm(rolling_dates, desc="Rolling forecasts"):
        # Include all data up to the forecast_start + horizon
        relevant_data = data[data.index < forecast_start + pd.Timedelta(hours=max(horizons))]
        preds = SimpleARModel_fixedlags(relevant_data, forecast_start)

        for h in horizons:
            ts = forecast_start + pd.Timedelta(hours=h)
            if ts in preds:
                all_predictions[h].append((ts, preds[ts]))

    # Assemble prediction DataFrames
    forecast_dfs = {}
    for h in horizons:
        df = pd.DataFrame(all_predictions[h], columns=['timestamp', f'forecast_t{h}'])
        df.set_index('timestamp', inplace=True)
        forecast_dfs[h] = df

    test_pred = pd.concat(forecast_dfs.values(), axis=1)
    
    # Evaluate and plot
    print("\nTest metrics:")
    for h in horizons:
        y_true = test_data['current_price'].dropna()
        y_pred = test_pred[f'forecast_t{h}'].dropna()

        common_idx = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_idx]
        y_pred = y_pred.loc[common_idx]

        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"\nHorizon t+{h}:")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R2: {metrics['r2']:.2f}")
        print(f"SMAPE: {metrics['smape']:.2f}%")
        
        plt = plot_predictions(common_idx, y_true, y_pred, h,
                               f'Linear Model with lagged price values: {h}-Hour Ahead Forecast vs Actual Values')
        os.makedirs('models_14_38/ar/plots/linear_with_lags', exist_ok=True)
        plt.savefig(f'models_14_38/ar/plots/linear_with_lags/forecast_{h}h.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
