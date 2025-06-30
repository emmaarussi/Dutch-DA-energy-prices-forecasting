"""
non strided app
Processing horizon t+14h...
Coverage: 87.3%, Mean Width: 82.02

Processing horizon t+24h...
Coverage: 86.5%, Mean Width: 95.06

Processing horizon t+38h...
Coverage: 85.9%, Mean Width: 98.95

✅ Completed ENBPI calibration in 1141.95 seconds


with stride

Processing horizon t+14h...
Coverage: 79.5%, Mean Width: 63.56

Processing horizon t+24h...
Coverage: 79.7%, Mean Width: 78.76

Processing horizon t+38h...
Coverage: 78.4%, Mean Width: 77.25

✅ Completed ENBPI calibration in 1247.96 seconds

"""


import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance
from models_14_38.xgboost.OptimizedXGboost import XGBoostOptimized as XGBoostOptimized


class XGBoostENBPIonline(XGBoostOptimized):
    def __init__(self, horizons=range(14, 39), residual_window=100):
        super().__init__(horizons)
        self.ensemble_fitted_models = []
        self.ensemble_online_resid = np.array([])
        self.residual_window = residual_window
        self.update_freq = 24
        self.max_buffer_size = 1000
        self.n_rounds_update = 10
        self.ensemble_boot_predictions = None
        self.ensemble_in_boot_sample = None

    def generate_bootstrap_samples(self, n, m, B):
        '''
            Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
        '''
        samples_idx = np.zeros((B, m), dtype=int)
        for b in range(B):
            sample_idx = np.random.choice(n, m)
            samples_idx[b, :] = sample_idx
        return samples_idx

    @staticmethod
    def strided_app(a, L, S):
        """
        Generate strided windows from array a with window size L and stride S
        Returns: 2D array of shape (n_windows, L)
        """
        a = np.asarray(a)
        if a.ndim != 1 or a.size < L:
            return np.array([]).reshape(0, L)
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


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

        self.ensemble_boot_predictions = boot_predictions
        self.ensemble_in_boot_sample = in_boot_sample

        return boot_predictions, np.array(oob_residuals)

    def compute_prediction_intervals(self, X_train, y_train, X_test, y_test, test_index, alpha=0.1, B=100):
        self.current_horizon = self.horizons[0]

        boot_predictions, initial_residuals = self.fit_bootstrap_models_with_OOB(X_train, y_train, X_test, B)

        n = len(X_train)
        test_preds = np.mean(boot_predictions[:, n:], axis=0)

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

    def predict_online_with_residual_update(self, X_test_row, y_test_val, alpha=0.1):
        n = self.ensemble_in_boot_sample.shape[1]
        B = self.ensemble_boot_predictions.shape[0]

        # Predict each model
        row_preds = np.array([
            model.predict(X_test_row.reshape(1, -1)).item()
            for model in self.ensemble_fitted_models
        ])

        test_pred = np.mean(row_preds)

        # Compute prediction interval
        resid_buffer = self.ensemble_online_resid[-self.residual_window:]

        # Apply strided windowing for robust quantile calculation
        resid_windows = self.strided_app(resid_buffer, L=10, S=1)  # e.g., 10-length windows

        if len(resid_windows) > 0:
            q_per_window = np.percentile(resid_windows, (1 - alpha) * 100, axis=1)
            q = np.mean(q_per_window)  # or np.median(q_per_window)
        else:
            q = np.percentile(resid_buffer, (1 - alpha) * 100)  # fallback

        lower = test_pred - q
        upper = test_pred + q

        # Update residual
        resid = abs(y_test_val - test_pred)
        self.ensemble_online_resid = np.append(self.ensemble_online_resid, resid)

        return test_pred, lower, upper


    def compute_prediction_intervals_online(self, X_train, y_train, X_test, y_test, test_index, alpha=0.1, B=20):
        self.current_horizon = self.horizons[0]
        _, initial_residuals = self.fit_bootstrap_models_with_OOB(X_train, y_train, X_test, B)

        self.ensemble_online_resid = initial_residuals[-self.residual_window:]

        preds, lowers, uppers = [], [], []

        for i in range(len(X_test)):
            pred, lower, upper = self.predict_online_with_residual_update(
                X_test[i], y_test[i], alpha=alpha
            )
            preds.append(pred)
            lowers.append(lower)
            uppers.append(upper)

        return pd.DataFrame({
            'predicted': preds,
            'lower': lowers,
            'upper': uppers,
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

    model = XGBoostENBPIonline(horizons=[14, 24, 38])

    for horizon in model.horizons:
        print(f"\nProcessing horizon t+{horizon}h...")

        X_train, y_train = model.prepare_data(train_df, horizon)
        X_test, y_test = model.prepare_data(test_df, horizon)

        results = model.compute_prediction_intervals_online(
            X_train.values, y_train.values,
            X_test.values, y_test.values, test_df.index,
            alpha=0.1, B=20
        )

        coverage = np.mean((results['actual'] >= results['lower']) & (results['actual'] <= results['upper'])) * 100
        mean_width = np.mean(results['upper'] - results['lower'])

        print(f"Coverage: {coverage:.1f}%, Mean Width: {mean_width:.2f}")

        out_dir = Path('models_14_38/xgboost/plots/enbpi_calibrated')
        out_dir.mkdir(parents=True, exist_ok=True)


        # Plot
        plot_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": results['predicted'],
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

    print(f"\n✅ Completed ENBPI calibration in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()