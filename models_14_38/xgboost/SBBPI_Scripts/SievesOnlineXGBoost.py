"""
Processing Online Sieve Bootstrap for t+14h...
Processing horizon t+14h...
Coverage: 80.3%, Mean Width: 30.49

Processing horizon t+24h...
Coverage: 79.2%, Mean Width: 31.30

Processing horizon t+38h...
Coverage: 78.5%, Mean Width: 30.92

✅ Completed Sieve calibration in 78.84 seconds




Processing horizon t+14h...
Coverage: 79.8%, Mean Width: 30.40

Processing horizon t+24h...
Coverage: 80.0%, Mean Width: 31.52

Processing horizon t+38h...
Coverage: 77.6%, Mean Width: 30.77

✅ Completed Sieve calibration in 186.91 seconds
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance
from models_14_38.xgboost.OptimizedXGboost import XGBoostOptimized as XGBoostOptimized

class OnlineXGBoostSieveBootstrap(XGBoostOptimized):
        def __init__(self, horizons=range(14, 39), window_size=200, update_freq=24, B=100, alpha=0.1):
            super().__init__(horizons)
            self.window_size = window_size
            self.update_freq = update_freq
            self.B = B
            self.alpha = alpha

        def fit_ar_model(self, residuals):
            max_lags = 71
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

        def plot_residuals_over_time(self, results_df, horizon):
            df = results_df.copy()
            df['residual'] = df['actual'] - df['point_pred']

            plt.figure(figsize=(12, 5))
            plt.plot(df.index, df['residual'], label="Residuals", alpha=0.7)
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.title(f"Residuals Over Time — t+{horizon}h")
            plt.xlabel("Date")
            plt.ylabel("Residual (Actual - Predicted)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        def plot_rolling_mean_residual(self, results_df, horizon, window=24):
            df = results_df.copy()
            df['residual'] = df['actual'] - df['point_pred']
            df['rolling_mean'] = df['residual'].rolling(window=window).mean()

            plt.figure(figsize=(12, 5))
            plt.plot(df.index, df['rolling_mean'], label=f"{window}-hour rolling mean", color='orange')
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.title(f"Rolling Mean Residuals — t+{horizon}h")
            plt.xlabel("Date")
            plt.ylabel("Residual")
            plt.legend()
            plt.tight_layout()
            plt.show()
            

        def generate_bootstrap_samples(self, residuals, selected_lags, phi, innovations, n, B):
            max_lag = max(selected_lags)
            samples = []
            for _ in range(B):
                resampled = np.random.choice(innovations, size=n, replace=True)
                series = list(residuals[-max_lag:])
                for t in range(n):
                    prev = np.array([series[-lag] for lag in selected_lags])
                    val = np.dot(phi, prev) + resampled[t]
                    series.append(val)
                samples.append(series[-n:])
            return np.array(samples)

        def online_forecast(self, X_train, y_train, X_test, y_test, horizon):
            params = self.get_hyperparameters(horizon)
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)

            # Only use the last residuals from training to initialize buffer
            train_preds = model.predict(X_train)
            train_residuals = y_train - train_preds
            buffer = list(train_residuals[-self.window_size:])  # Only the tail
            selected_lags, phi, innovations = None, None, None

            preds, lowers, uppers, actuals = [], [], [], []
            ar_success_count = 0
            default_count = 0

            for t in range(len(X_test)):
                x_t = X_test.iloc[t].values.reshape(1, -1)
                y_t = y_test.iloc[t]
                y_pred = model.predict(x_t)[0]

                # Update AR model periodically
                if len(buffer) >= self.window_size and t % self.update_freq == 0:
                    resid_series = pd.Series(buffer[-self.window_size:])
                    try:
                        selected_lags, phi, innovations = self.fit_ar_model(resid_series)
                    except Exception as e:
                        print(f"AR model fitting failed at step {t}: {e}")
                        selected_lags, phi, innovations = None, None, None

                # Generate prediction interval
                if selected_lags is not None:
                    try:
                        samples = self.generate_bootstrap_samples(
                            buffer, selected_lags, phi, innovations, 1, self.B
                        )
                        lower = y_pred + np.percentile(samples, 100 * self.alpha / 2)
                        upper = y_pred + np.percentile(samples, 100 * (1 - self.alpha / 2))
                        ar_success_count += 1
                    except Exception as e:
                        print(f"Bootstrap generation failed at step {t}: {e}")
                        lower = y_pred - 15
                        upper = y_pred + 15
                        default_count += 1
                else:
                    lower = y_pred - 15
                    upper = y_pred + 15
                    default_count += 1

                preds.append(y_pred)
                lowers.append(lower)
                uppers.append(upper)
                actuals.append(y_t)

                # ⚠️ Only update buffer with current residual — do not look ahead
                buffer.append(y_t - y_pred)
                if len(buffer) > 1000:
                    buffer.pop(0)

            print(f"[t+{horizon}h] AR-based intervals: {ar_success_count}, fallback defaults: {default_count}")

            return pd.DataFrame({
                'point_pred': preds,
                'lower': lowers,
                'upper': uppers,
                'actual': actuals
            }, index=y_test.index)




def main():
    start_time = time.time()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load training data (Jan 2023 – Mar 2024)
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path, index_col=0)
    train_df.index = pd.to_datetime(train_df.index)
    print(f"Training data loaded, date range: {train_df.index.min()} to {train_df.index.max()}")
    
    # Load test data from separate calibration set
    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path, index_col=0)
    test_df.index = pd.to_datetime(test_df.index)
    print(f"Test data loaded, date range: {test_df.index.min()} to {test_df.index.max()}")

    # Initialize model
    model = OnlineXGBoostSieveBootstrap(horizons=[14, 24, 38])

    for horizon in model.horizons:
        print(f"\nProcessing horizon t+{horizon}h...")

        X_train, y_train = model.prepare_data(train_df, horizon)
        X_test, y_test = model.prepare_data(test_df, horizon)

        results = model.online_forecast(
            X_train=pd.DataFrame(X_train, index=train_df.index[-len(y_train):]), 
            y_train=y_train,
            X_test=pd.DataFrame(X_test, index=test_df.index[-len(y_test):]),
            y_test=y_test,
            horizon=horizon
        )


        # Coverage and width
        coverage = np.mean((results['actual'] >= results['lower']) & (results['actual'] <= results['upper'])) * 100
        mean_width = np.mean(results['upper'] - results['lower'])
        metrics = calculate_metrics(y_test, results['point_pred'])
        print(metrics)

        print(f"Coverage: {coverage:.1f}%, Mean Width: {mean_width:.2f}")

        out_dir = Path('models_14_38/xgboost/plots/sieve_online')
        out_dir.mkdir(parents=True, exist_ok=True)

        # Plot
        plot_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": results['point_pred'],
            "Lower": results['lower'],
            "Upper": results['upper']
        }, index=y_test.index)

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df.index, plot_df["Actual"], label="Actual", alpha=0.7)
        plt.plot(plot_df.index, plot_df["Predicted"], label="Predicted", alpha=0.7)
        plt.fill_between(plot_df.index, plot_df["Lower"], plot_df["Upper"], alpha=0.2, label="90% CI")
        plt.title(f"Actual vs Predicted Prices - t+{horizon}h")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"predictions_{horizon}.png")
        plt.close()

        # Visualize residual drift
        model.plot_residuals_over_time(results, horizon)
        model.plot_rolling_mean_residual(results, horizon, window=48)


    print(f"\n✅ Completed Sieve calibration in {time.time() - start_time:.2f} seconds")

    
if __name__ == "__main__":
    main()

