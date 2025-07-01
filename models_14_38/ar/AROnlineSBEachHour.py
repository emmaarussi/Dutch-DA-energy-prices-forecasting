"""

Horizon t+14h — Coverage: 81.8%, Mean Width: 76.60

Horizon t+24h — Coverage: 70.0%, Mean Width: 153.07

Horizon t+38h — Coverage: 70.0%, Mean Width: 601.97

✅ Completed AR Sieve Bootstrap in 73.74 seconds


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


class OnlineARBootstrap:
    def __init__(self, horizons=[14, 24, 38], residual_window=100):
        self.horizons = horizons
        self.residual_window = residual_window
        self.B = 100
        self.alpha = 0.1

    def fit_ar_model(self, residuals):
        max_lags = min(20, len(residuals) // 2)  # prevent overfitting on short buffers
        if max_lags < 1:
            raise ValueError("Too few residuals to fit AR model.")
        
        initial_model = AutoReg(residuals, lags=max_lags, old_names=False, period=24)
        initial_fit = initial_model.fit()
        n = len(residuals)
        bic_threshold = np.log(n)
        selected_lags = []

        for i, (coef, std_err) in enumerate(zip(initial_fit.params[1:], initial_fit.bse[1:])):
            t_stat = coef / std_err
            bic_contribution = t_stat**2 - bic_threshold
            if bic_contribution > 0:
                selected_lags.append(i + 1)
        if not selected_lags:
            selected_lags = [1]

        model = AutoReg(residuals, lags=selected_lags, old_names=False)
        fit = model.fit()
        return selected_lags, fit.params[1:], fit.resid

    def generate_sieve_bootstrap_paths(self, residuals, selected_lags, phi, innovations, h, B):
        max_lag = max(selected_lags)
        series_init = list(residuals[-max_lag:])
        samples = []

        for _ in range(B):
            resampled_innov = np.random.choice(innovations, size=h, replace=True)
            series = series_init.copy()
            path = []

            for t in range(h):
                prev = np.array([series[-lag] for lag in selected_lags])
                val = np.dot(phi, prev) + resampled_innov[t]
                series.append(val)
                path.append(val)

            samples.append(path)

        return np.array(samples)

    def compute_intervals_from_ar_predictions(self, point_forecasts_df, horizons, alpha=None, prefill_buffer=None):
        alpha = self.alpha if alpha is None else alpha
        results = []

        for horizon in horizons:
            df = point_forecasts_df[point_forecasts_df['horizon'] == horizon].copy()
            df = df.sort_values('target_time')
            residuals = df['actual'] - df['predicted']

            buffer = (prefill_buffer or []) + residuals.tolist()
            lowers, uppers = [], []

            for i in range(len(df)):
                pred = df.iloc[i]['predicted']

                resid_buffer = buffer[i : i + self.residual_window]
                if len(resid_buffer) < 20:
                    print(f"[t+{horizon}h | {i}] Skipping — buffer too small ({len(resid_buffer)})")
                    continue

                try:
                    selected_lags, phi, innovations = self.fit_ar_model(pd.Series(resid_buffer))
                    samples = self.generate_sieve_bootstrap_paths(
                        residuals=resid_buffer,
                        selected_lags=selected_lags,
                        phi=np.array(phi),
                        innovations=innovations,
                        h=int(horizon),
                        B=self.B
                    )
                    residual_forecasts = samples[:, horizon - 1]
                    lower = pred + np.percentile(residual_forecasts, 100 * alpha / 2)
                    upper = pred + np.percentile(residual_forecasts, 100 * (1 - alpha / 2))
                except Exception as e:
                    print(f"[t+{horizon}h | {i}] AR fitting failed: {e}")
                    continue

                lowers.append(lower)
                uppers.append(upper)

            df = df.iloc[len(df) - len(lowers):].copy()
            df['lower'] = lowers
            df['upper'] = uppers
            results.append(df)

        return pd.concat(results)



def main():
    start_time = time.time()

    # --- Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')

    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path, index_col=0)
    train_df.index = pd.to_datetime(train_df.index, utc=True)

    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path, index_col=0)
    test_df.index = pd.to_datetime(test_df.index, utc=True)

    # --- Merge data
    full_df = pd.concat([train_df, test_df]).sort_index()

    # --- Run AR forecasts
    forecaster = ARRollingForecaster(data=full_df, horizons=[14, 24, 38], window_size='365D')
    test_start = test_df.index.min()
    forecaster.run_forecast(test_start=test_start)
    point_forecasts = forecaster.results_df
    point_forecasts = point_forecasts[point_forecasts['target_time'].isin(test_df.index)]

    # --- Pre-fill buffer from training residuals
    prefill_residuals = forecaster.results_df[
        (forecaster.results_df['target_time'] < test_df.index.min())
    ]['actual'] - forecaster.results_df['predicted']
    prefill_buffer = prefill_residuals.dropna().tolist()[-100:]

    # --- Sieve Bootstrap
    print("\nRunning Sieve Bootstrap for AR forecasts...")
    model = OnlineARBootstrap(residual_window=100)
    results = model.compute_intervals_from_ar_predictions(
        point_forecasts_df=point_forecasts,
        horizons=[14, 24, 38],
        alpha=0.1,
        prefill_buffer=prefill_buffer
    )

    # --- Evaluation and Plotting
    out_dir = Path('models_14_38/ar/plots/sieve_bootstrap')
    out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in [14, 24, 38]:
        df_h = results[results['horizon'] == horizon]
        if df_h.empty:
            print(f"[t+{horizon}h] Skipped plotting — no intervals available.")
            continue

        coverage = ((df_h['actual'] >= df_h['lower']) & (df_h['actual'] <= df_h['upper'])).mean() * 100
        width = (df_h['upper'] - df_h['lower']).mean()

        print(f"\nHorizon t+{horizon}h — Coverage: {coverage:.1f}%, Mean Width: {width:.2f}")

        plt.figure(figsize=(12, 6))
        plt.plot(df_h['target_time'], df_h['actual'], label="Actual", alpha=0.7)
        plt.plot(df_h['target_time'], df_h['predicted'], label="Predicted", alpha=0.7)
        plt.fill_between(df_h['target_time'], df_h['lower'], df_h['upper'], alpha=0.2, label="90% CI")
        plt.title(f"AR + Sieve Bootstrap — t+{horizon}h")
        plt.xlabel("Date")
        plt.ylabel("Price (EUR/MWh)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"sieve_bootstrap_ar_t{horizon}h.png")
        plt.close()

    print(f"\n✅ Completed AR Sieve Bootstrap in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
