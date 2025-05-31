"""test_df.index = pd.to_datetime(test_df.index)
Test data loaded, date range: 2024-03-08 00:00:00+01:00 to 2024-04-29 09:00:00+02:00

Processing horizon t+14h...
Coverage: 23.5%, Mean Width: 10.43

Processing horizon t+24h...
Coverage: 22.8%, Mean Width: 10.42

Processing horizon t+38h...
Coverage: 16.8%, Mean Width: 10.82

✅ Completed Sieve calibration in 54.25 seconds

Training data loaded, date range: 2023-01-08 00:00:00+01:00 to 2024-02-28 09:00:00+01:00

Loading test data from: /Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/data/processed/multivariate_features_testset_selectedXGboost.csv
Test data loaded, date range: 2024-03-08 00:00:00+01:00 to 2024-04-29 09:00:00+02:00

Processing horizon t+14h...

AR Model Summary for Residuals:
Number of lags: 71

Significant lags (p < 0.05):
Lag 1: coefficient = 0.3011 (p = 0.0000)
Lag 2: coefficient = 0.0354 (p = 0.0007)
Lag 4: coefficient = 0.0382 (p = 0.0003)
Lag 6: coefficient = 0.0217 (p = 0.0391)
Lag 7: coefficient = 0.0306 (p = 0.0036)
Lag 14: coefficient = -0.0706 (p = 0.0000)
Lag 15: coefficient = 0.0217 (p = 0.0398)
Lag 16: coefficient = -0.0265 (p = 0.0119)
Lag 17: coefficient = 0.0222 (p = 0.0349)
Lag 24: coefficient = -0.1119 (p = 0.0000)
Lag 25: coefficient = 0.0288 (p = 0.0066)
Lag 39: coefficient = -0.0291 (p = 0.0060)
Lag 43: coefficient = 0.0224 (p = 0.0346)
Lag 47: coefficient = -0.0214 (p = 0.0433)

Total significant lags: 14
Most important lags (by absolute coefficient value):
Lag 1: coefficient = 0.3011
Lag 24: coefficient = -0.1119
Lag 14: coefficient = -0.0706
Lag 4: coefficient = 0.0382
Lag 2: coefficient = 0.0354
Coverage: 23.9%, Mean Width: 10.44

Processing horizon t+24h...

AR Model Summary for Residuals:
Number of lags: 71

Significant lags (p < 0.05):
Lag 1: coefficient = 0.2259 (p = 0.0000)
Lag 2: coefficient = 0.0387 (p = 0.0002)
Lag 4: coefficient = 0.0369 (p = 0.0003)
Lag 7: coefficient = 0.0362 (p = 0.0004)
Lag 8: coefficient = 0.0325 (p = 0.0016)
Lag 9: coefficient = 0.0356 (p = 0.0006)
Lag 17: coefficient = 0.0204 (p = 0.0487)
Lag 20: coefficient = 0.0239 (p = 0.0208)
Lag 24: coefficient = -0.1611 (p = 0.0000)
Lag 25: coefficient = 0.0384 (p = 0.0002)
Lag 43: coefficient = 0.0221 (p = 0.0344)
Lag 45: coefficient = 0.0230 (p = 0.0271)
Lag 48: coefficient = -0.0556 (p = 0.0000)
Lag 49: coefficient = 0.0205 (p = 0.0470)

Total significant lags: 14
Most important lags (by absolute coefficient value):
Lag 1: coefficient = 0.2259
Lag 24: coefficient = -0.1611
Lag 48: coefficient = -0.0556
Lag 2: coefficient = 0.0387
Lag 25: coefficient = 0.0384
Coverage: 21.9%, Mean Width: 10.42

Processing horizon t+38h...

AR Model Summary for Residuals:
Number of lags: 71

Significant lags (p < 0.05):
Lag 1: coefficient = 0.3018 (p = 0.0000)
Lag 2: coefficient = 0.0227 (p = 0.0303)
Lag 4: coefficient = 0.0345 (p = 0.0010)
Lag 6: coefficient = 0.0303 (p = 0.0038)
Lag 23: coefficient = 0.0222 (p = 0.0341)
Lag 24: coefficient = 0.0712 (p = 0.0000)
Lag 35: coefficient = 0.0217 (p = 0.0384)
Lag 45: coefficient = 0.0219 (p = 0.0363)
Lag 47: coefficient = -0.0223 (p = 0.0334)
Lag 48: coefficient = -0.0725 (p = 0.0000)
Lag 49: coefficient = 0.0304 (p = 0.0037)

Total significant lags: 11
Most important lags (by absolute coefficient value):
Lag 1: coefficient = 0.3018
Lag 48: coefficient = -0.0725
Lag 24: coefficient = 0.0712
Lag 4: coefficient = 0.0345
Lag 49: coefficient = 0.0304
Coverage: 16.4%, Mean Width: 10.83

✅ Completed Sieve calibration in 89.98 seconds


Loading training data from: /Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/data/processed/multivariate_features_selectedXGboost.csv
Training data loaded, date range: 2023-01-08 00:00:00+01:00 to 2024-02-28 09:00:00+01:00

Loading test data from: /Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/data/processed/multivariate_features_testset_selectedXGboost.csv
Test data loaded, date range: 2024-03-08 00:00:00+01:00 to 2024-04-29 09:00:00+02:00

Processing horizon t+14h...
Selected lags via BIC: [1, 2, 4, 14, 24]

AR Model Summary for Residuals:
Number of lags: 5

Lag coefficients:
Lag 1: coefficient = 0.3006 (p = 0.0000)
Lag 2: coefficient = 0.0380 (p = 0.0001)
Lag 4: coefficient = 0.0443 (p = 0.0000)
Lag 14: coefficient = -0.0630 (p = 0.0000)
Lag 24: coefficient = -0.0987 (p = 0.0000)

Most important lags (by absolute coefficient value):
Lag 1: coefficient = 0.3006
Lag 24: coefficient = -0.0987
Lag 14: coefficient = -0.0630
Lag 4: coefficient = 0.0443
Lag 2: coefficient = 0.0380
Coverage: 24.9%, Mean Width: 10.48

Processing horizon t+24h...
Selected lags via BIC: [1, 2, 4, 7, 8, 9, 24, 25, 48]

AR Model Summary for Residuals:
Number of lags: 9

Lag coefficients:
Lag 1: coefficient = 0.2249 (p = 0.0000)
Lag 2: coefficient = 0.0411 (p = 0.0000)
Lag 4: coefficient = 0.0413 (p = 0.0000)
Lag 7: coefficient = 0.0403 (p = 0.0000)
Lag 8: coefficient = 0.0336 (p = 0.0009)
Lag 9: coefficient = 0.0361 (p = 0.0003)
Lag 24: coefficient = -0.1562 (p = 0.0000)
Lag 25: coefficient = 0.0348 (p = 0.0005)
Lag 48: coefficient = -0.0492 (p = 0.0000)

Most important lags (by absolute coefficient value):
Lag 1: coefficient = 0.2249
Lag 24: coefficient = -0.1562
Lag 48: coefficient = -0.0492
Lag 4: coefficient = 0.0413
Lag 2: coefficient = 0.0411
Coverage: 22.4%, Mean Width: 10.38

Processing horizon t+38h...
Selected lags via BIC: [1, 4, 24, 48]

AR Model Summary for Residuals:
Number of lags: 4

Lag coefficients:
Lag 1: coefficient = 0.3108 (p = 0.0000)
Lag 4: coefficient = 0.0429 (p = 0.0000)
Lag 24: coefficient = 0.0687 (p = 0.0000)
Lag 48: coefficient = -0.0689 (p = 0.0000)

Most important lags (by absolute coefficient value):
Lag 1: coefficient = 0.3108
Lag 48: coefficient = -0.0689
Lag 24: coefficient = 0.0687
Lag 4: coefficient = 0.0429
Coverage: 17.3%, Mean Width: 10.77

✅ Completed Sieve calibration in 83.54 seconds




"""

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance
from models_14_38.xgboost.OptimizedXGboost import XGBoostOptimized as XGBoostOptimized

class XGBoostSieveBootstrap(XGBoostOptimized):
    def __init__(self, horizons=range(14, 39)):
        super().__init__(horizons)
        self.bootstrap_forecasts = []

    def generate_sieve_bootstrap_residuals(self, residuals, B=100, ar_order=None):
        # Initial fit with all lags
        max_lags = 71 if ar_order is None else ar_order
        initial_model = AutoReg(residuals, lags=max_lags, old_names=False, period=24)
        initial_fit = initial_model.fit()
        
        # Calculate BIC for each lag to determine significance
        n = len(residuals)
        bic_threshold = np.log(n)  # BIC penalty term
        selected_lags = []
        
        # For each lag, check if its contribution justifies its inclusion
        for i, (coef, std_err) in enumerate(zip(initial_fit.params[1:], initial_fit.bse[1:])):
            t_stat = coef / std_err
            bic_contribution = t_stat**2 - bic_threshold
            if bic_contribution > 0:
                selected_lags.append(i + 1)
        
        if not selected_lags:
            selected_lags = [1]  # Default to lag 1 if no lags are significant
        
        # Sort lags for consistent ordering
        selected_lags.sort()
        max_lag = max(selected_lags)
        
        # Refit model with only significant lags
        model = AutoReg(residuals, lags=selected_lags, old_names=False)
        fit = model.fit()
        print(f"Selected lags via BIC: {selected_lags}")
        
        # Get residuals and coefficients
        innovations = fit.resid
        phi = fit.params[1:]  # Skip intercept
        
        # Print model summary
        print(f"\nAR Model Summary for Residuals:")
        print(f"Number of lags: {len(selected_lags)}")
        print("\nLag coefficients:")
        for lag, (coef, pval) in zip(selected_lags, zip(phi, fit.pvalues[1:])):
            print(f"Lag {lag}: coefficient = {coef:.4f} (p = {pval:.4f})")
        
        print("\nMost important lags (by absolute coefficient value):")
        sorted_lags = sorted(zip(selected_lags, phi), key=lambda x: abs(x[1]), reverse=True)[:5]
        for lag, coef in sorted_lags:
            print(f"Lag {lag}: coefficient = {coef:.4f}")
        
        # For bootstrap, we need to track max_lag for the initial conditions
        n = len(residuals)
        bootstrap_residuals = []
        for b in range(B):
            # Randomly sample innovations with replacement
            resampled = np.random.choice(innovations, size=n, replace=True)
            # Generate bootstrap sample
            series = list(residuals[:max_lag])  # Initial conditions
            for t in range(max_lag, n):
                # Get previous values for all selected lags
                prev_values = np.array([series[t - lag] for lag in selected_lags])
                # Apply AR coefficients
                val = np.dot(phi, prev_values) + resampled[t]
                series.append(val)
            bootstrap_residuals.append(series)
        
        return np.array(bootstrap_residuals)

        # Print significant lags and their coefficients
        print(f"\nAR Model Summary for Residuals:")
        print(f"Number of lags: {ar_order}")
        print("\nSignificant lags (p < 0.05):")
        pvalues = fit.pvalues[1:]  # Skip intercept
        significant_lags = []
        for lag, (coef, pval) in enumerate(zip(phi, pvalues), 1):
            if pval < 0.05:
                significant_lags.append(lag)
                print(f"Lag {lag}: coefficient = {coef:.4f} (p = {pval:.4f})")
        
        print(f"\nTotal significant lags: {len(significant_lags)}")
        print("Most important lags (by absolute coefficient value):")
        sorted_lags = sorted([(lag, coef) for lag, coef in enumerate(phi, 1)], 
                            key=lambda x: abs(x[1]), reverse=True)[:5]
        for lag, coef in sorted_lags:
            print(f"Lag {lag}: coefficient = {coef:.4f}")


        n = len(residuals)
        bootstrap_residuals = []
        for _ in range(B):
            resampled = np.random.choice(innovations, size=n, replace=True)
            series = list(residuals[:ar_order])
            for t in range(ar_order, n):
                val = np.dot(phi, series[-ar_order:]) + resampled[t]
                series.append(val)
            bootstrap_residuals.append(np.array(series[-n:]))
        return np.array(bootstrap_residuals)

    def compute_prediction_intervals(self, X_train, y_train, X_test, y_test, alpha=0.1, B=100, ar_order=5):
        self.current_horizon = self.horizons[0]
        params = self.get_hyperparameters(self.current_horizon)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train

        boot_resid = self.generate_sieve_bootstrap_residuals(residuals, B=B, ar_order=ar_order)

        # Generate B test forecasts
        n_test = len(y_test)
        forecasts = np.zeros((B, n_test))
        for b in range(B):
            resid_sample = np.random.choice(boot_resid[b], size=n_test, replace=True)
            forecasts[b] = y_pred_test + resid_sample

        lower = np.percentile(forecasts, 100 * (alpha / 2), axis=0)
        upper = np.percentile(forecasts, 100 * (1 - alpha / 2), axis=0)

        return pd.DataFrame({
            'predicted': y_pred_test,
            'lower': lower,
            'upper': upper,
            'actual': y_test
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
    model = XGBoostSieveBootstrap(horizons=[14, 24, 38])

    for horizon in model.horizons:
        print(f"\nProcessing horizon t+{horizon}h...")

        X_train, y_train = model.prepare_data(train_df, horizon)
        X_test, y_test = model.prepare_data(test_df, horizon)

        results = model.compute_prediction_intervals(
            X_train.values, y_train.values,
            X_test.values, y_test,
            alpha=0.1,
            B=500,
            ar_order=None  # Let BIC select optimal lags
        )

        # Coverage and width
        coverage = np.mean((results['actual'] >= results['lower']) & (results['actual'] <= results['upper'])) * 100
        mean_width = np.mean(results['upper'] - results['lower'])

        print(f"Coverage: {coverage:.1f}%, Mean Width: {mean_width:.2f}")

        # Plot
        out_dir = 'models_14_38/xgboost/plots/sieve_calibrated'
        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 6))
        plt.fill_between(results.index, results['lower'], results['upper'], alpha=0.3, label='90% PI')
        plt.plot(results.index, results['predicted'], linestyle='--', label='Predicted')
        plt.plot(results.index, results['actual'], color='black', label='Actual')
        plt.title(f'Sieve Bootstrap Forecast (t+{horizon}h)')
        plt.xlabel('Date'); plt.ylabel('Price (EUR/MWh)')
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(f'{out_dir}/sieve_forecast_h{horizon}.png')
        plt.close()

    print(f"\n✅ Completed Sieve calibration in {time.time() - start_time:.2f} seconds")

    
if __name__ == "__main__":
    main()

