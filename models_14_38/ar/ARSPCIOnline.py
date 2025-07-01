

"""Running Online AR SPCI on AR forecasts...



✅ Completed AR Online SPCI in 79.42 seconds
"""



# --- Imports
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order  # 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance
from sklearn.ensemble import GradientBoostingRegressor


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

        for timestamp in pd.date_range(start=start, end=end, freq='1H'):
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


class OnlineARSPCI:
    def __init__(self, horizons=[14, 24, 38], alpha=0.1, window_size=100):
        self.horizons = horizons
        self.alpha = alpha
        self.window_size = window_size
        self.update_freq = 24  # hours
        self.max_buffer_size = 1000  # max number of residuals stored

    def fit_quantile_model(self, X, y, quantile, existing_model=None):
        model = GradientBoostingRegressor(loss="quantile", alpha=quantile)
        model.fit(X, y)
        return model

    def online_spci(self, point_forecasts_df):
        all_results = []

        for horizon in self.horizons:
            df = point_forecasts_df[point_forecasts_df['horizon'] == horizon].copy()
            df = df.dropna(subset=['actual', 'predicted'])
            df['residual'] = df['actual'] - df['predicted']

            residual_buffer = list(df['residual'].iloc[:self.window_size])
            q05_model = q50_model = q95_model = None

            preds, lowers, uppers, actuals, times = [], [], [], [], []

            for t in range(self.window_size, len(df)):
                y_pred = df['predicted'].iloc[t]
                y_t = df['actual'].iloc[t]
                target_time = df['target_time'].iloc[t]

                if len(residual_buffer) >= self.window_size and (t % self.update_freq == 0):
                    recent = residual_buffer[-self.max_buffer_size:]
                    X_resid, y_resid = [], []

                    for i in range(len(recent) - self.window_size):
                        X_resid.append(recent[i:i + self.window_size])
                        y_resid.append(recent[i + self.window_size])

                    if X_resid:
                        X_resid = pd.DataFrame(X_resid)
                        y_resid = pd.Series(y_resid)

                        q05_model = self.fit_quantile_model(X_resid, y_resid, self.alpha / 2)
                        q50_model = self.fit_quantile_model(X_resid, y_resid, 0.5)
                        q95_model = self.fit_quantile_model(X_resid, y_resid, 1 - self.alpha / 2)

                if all([q05_model, q50_model, q95_model]):
                    X_input = pd.DataFrame(
                        np.array(residual_buffer[-self.window_size:]).reshape(1, -1),
                        columns=range(self.window_size)
                    )
                    q05 = q05_model.predict(X_input)[0]
                    q50 = q50_model.predict(X_input)[0]
                    q95 = q95_model.predict(X_input)[0]

                    lower = y_pred + (q05 - q50)
                    upper = y_pred + (q95 - q50)
                else:
                    lower = y_pred - 15
                    upper = y_pred + 15

                preds.append(y_pred)
                lowers.append(lower)
                uppers.append(upper)
                actuals.append(y_t)
                times.append(target_time)

                residual_buffer.append(y_t - y_pred)
                if len(residual_buffer) > self.max_buffer_size:
                    residual_buffer.pop(0)

            result_df = pd.DataFrame({
                'target_time': times,
                'horizon': horizon,
                'actual': actuals,
                'predicted': preds,
                'lower': lowers,
                'upper': uppers
            })

            all_results.append(result_df)

        return pd.concat(all_results, ignore_index=True)









def main():
    start_time = time.time()

    # --- Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')

    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path, index_col=0)
    train_df.index = pd.to_datetime(train_df.index, utc=True)
    print(f"Training data loaded: {train_df.index.min()} to {train_df.index.max()}")

    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path, index_col=0)
    test_df.index = pd.to_datetime(test_df.index, utc=True)
    print(f"Test data loaded: {test_df.index.min()} to {test_df.index.max()}")

    # --- Merge data
    full_df = pd.concat([train_df, test_df]).sort_index()

    # --- Run AR forecasts
    forecaster = ARRollingForecaster(data=full_df, horizons=[14, 24, 38], window_size='365D')
    test_start = test_df.index.min()
    forecaster.run_forecast(test_start=test_start)
    point_forecasts = forecaster.results_df
    point_forecasts = point_forecasts[point_forecasts['target_time'].isin(test_df.index)]

    # --- Online AR SPCI
    print("\nRunning Online AR SPCI on AR forecasts...")
    model = OnlineARSPCI(window_size=100)
    results = model.online_spci(point_forecasts_df=point_forecasts)

    # --- Evaluation and Plotting
    out_dir = Path('models_14_38/ar/plots/spci_online')
    out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in [14, 24, 38]:
        df_h = results[results['horizon'] == horizon]
        coverage = ((df_h['actual'] >= df_h['lower']) & (df_h['actual'] <= df_h['upper'])).mean() * 100
        width = (df_h['upper'] - df_h['lower']).mean()

        print(f"\nHorizon t+{horizon}h — Coverage: {coverage:.1f}%, Mean Width: {width:.2f}")

        plt.figure(figsize=(12, 6))
        plt.plot(df_h['target_time'], df_h['actual'], label="Actual", alpha=0.7)
        plt.plot(df_h['target_time'], df_h['predicted'], label="Predicted", alpha=0.7)
        plt.fill_between(df_h['target_time'], df_h['lower'], df_h['upper'], alpha=0.2, label="90% CI")
        plt.title(f"AR + Online SPCI — t+{horizon}h")
        plt.xlabel("Date")
        plt.ylabel("Price (EUR/MWh)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"spci_ar_online_t{horizon}h.png")
        plt.close()

    print(f"\n✅ Completed AR SPCI in {time.time() - start_time:.2f} seconds")


    
if __name__ == "__main__":
    main()
