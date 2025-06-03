import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance
from models_14_38.xgboost.OptimizedXGboost import XGBoostOptimized as XGBoostOptimized


class XGBoostENBPI(XGBoostOptimized):
    def __init__(self, horizons=range(14, 39), residual_window=100):
        super().__init__(horizons)
        self.ensemble_fitted_models = []
        self.ensemble_online_resid = np.array([])
        self.residual_window = residual_window

    def generate_bootstrap_samples(self, n, m, B):
        samples_idx = np.zeros((B, m), dtype=int)
        for b in range(B):
            sample_idx = np.random.choice(n, m)
            samples_idx[b, :] = sample_idx
        return samples_idx

    def fit_bootstrap_models_with_OOB(self, X_train, y_train, X_test, B=20):
        n = len(X_train)
        n1 = len(X_test)
        boot_samples_idx = self.generate_bootstrap_samples(n, n, B)
        boot_predictions = np.zeros((B, n + n1))
        in_boot_sample = np.zeros((B, n), dtype=bool)
        oob_residuals = []

        for b in range(B):
            params = self.get_hyperparameters(self.current_horizon)
            model = xgb.XGBRegressor(**params)
            sample_idx = boot_samples_idx[b]
            model.fit(X_train[sample_idx], y_train[sample_idx])
            self.ensemble_fitted_models.append(model)

            preds = model.predict(np.vstack((X_train, X_test)))
            boot_predictions[b] = preds
            in_boot_sample[b, sample_idx] = True

        for i in range(n):
            oob_models = ~in_boot_sample[:, i]
            if np.sum(oob_models) > 0:
                oob_pred = np.mean(boot_predictions[oob_models, i])
                resid = abs(y_train[i] - oob_pred)
            else:
                resid = abs(y_train[i])
            oob_residuals.append(resid)

        return boot_predictions, np.array(oob_residuals)

    def compute_prediction_intervals(self, X_train, y_train, X_test, y_test, test_index, alpha=0.1, B=100):
        self.current_horizon = self.horizons[0]

        boot_predictions, initial_residuals = self.fit_bootstrap_models_with_OOB(X_train, y_train, X_test, B)

        n = len(X_train)
        test_preds = np.median(boot_predictions[:, n:], axis=0)

        residual_buffer = list(initial_residuals[-self.residual_window:])

        lower_bounds = []
        upper_bounds = []

        for i in range(len(test_preds)):
            q = np.percentile(residual_buffer, (1 - alpha) * 100)
            lower_bounds.append(test_preds[i] - q)
            upper_bounds.append(test_preds[i] + q)

            residual = abs(y_test[i] - test_preds[i])
            residual_buffer.append(residual)
            if len(residual_buffer) > self.residual_window:
                residual_buffer.pop(0)

        return pd.DataFrame({
            'predicted': test_preds,
            'lower': lower_bounds,
            'upper': upper_bounds,
            'actual': y_test
        }, index=test_index)

def main():
    start_time = time.time()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path, index_col=0)
    train_df.index = pd.to_datetime(train_df.index)
    print(f"Training data loaded, date range: {train_df.index.min()} to {train_df.index.max()}")

    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path, index_col=0)
    test_df.index = pd.to_datetime(test_df.index)
    print(f"Test data loaded, date range: {test_df.index.min()} to {test_df.index.max()}")

    model = XGBoostENBPI(horizons=[14, 24, 38])

    for horizon in model.horizons:
        print(f"\nProcessing horizon t+{horizon}h...")

        X_train, y_train = model.prepare_data(train_df, horizon)
        X_test, y_test = model.prepare_data(test_df, horizon)

        results = model.compute_prediction_intervals(
            X_train.values, y_train.values,
            X_test.values, y_test.values, test_df.index,
            alpha=0.1, B=20
        )

        coverage = np.mean((results['actual'] >= results['lower']) & (results['actual'] <= results['upper'])) * 100
        mean_width = np.mean(results['upper'] - results['lower'])

        print(f"Coverage: {coverage:.1f}%, Mean Width: {mean_width:.2f}")

        out_dir = 'models_14_38/xgboost/plots/enbpi_calibrated'
        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 6))
        plt.fill_between(results.index, results['lower'], results['upper'], alpha=0.3, label='90% PI')
        plt.plot(results.index, results['predicted'], linestyle='--', label='Predicted')
        plt.plot(results.index, results['actual'], color='black', label='Actual')
        plt.title(f'ENBPI Forecast (t+{horizon}h)')
        plt.xlabel('Date'); plt.ylabel('Price (EUR/MWh)')
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(f'{out_dir}/enbpi_forecast_h{horizon}.png')
        plt.close()

    print(f"\n✅ Completed ENBPI calibration in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()