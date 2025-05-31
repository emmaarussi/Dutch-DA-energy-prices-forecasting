

"""

first try of ENBPI

Processing horizon t+14h...
Coverage: 33.9%, Mean Width: 22.34

Processing horizon t+24h...
Coverage: 31.5%, Mean Width: 21.41

Processing horizon t+38h...

second try

Processing horizon t+14h...
Coverage: 69.2%, Mean Width: 46.43

Processing horizon t+24h...
Coverage: 66.3%, Mean Width: 45.66

Processing horizon t+38h...
Coverage: 64.8%, Mean Width: 45.29

✅ Completed ENBPI calibration in 1096.97 seconds


XGBoost model with ENBPI (Ensemble Neural Bootstrap Prediction Interval) for uncertainty quantification.
Combines the XGBoost model from OptimizedXGboost.py with prediction intervals from ENBPIXGboost.py.
"""
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
    def __init__(self, horizons=range(14, 39)):
        super().__init__(horizons)
        self.ensemble_fitted_models = []  # Store bootstrap models
        self.ensemble_online_resid = np.array([])  # Store residuals


    def generate_bootstrap_samples(self, n, m, B):
        '''
            Return: B-by-m matrix, where row b gives the indices for b-th bootstrap sample
        '''
        samples_idx = np.zeros((B, m), dtype=int)
        for b in range(B):
            sample_idx = np.random.choice(n, m)
            samples_idx[b, :] = sample_idx
        return(samples_idx)
    
    @staticmethod
    def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size-L)//S)+1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))
    
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

        # Compute OOB residuals
        for i in range(n):
            oob_models = ~in_boot_sample[:, i]
            if np.sum(oob_models) > 0:
                oob_pred = np.mean(boot_predictions[oob_models, i])
                resid = abs(y_train[i] - oob_pred)
            else:
                resid = abs(y_train[i])  # fallback
            oob_residuals.append(resid)

        return boot_predictions, np.array(oob_residuals)

    
    def compute_prediction_intervals(self, X_train, y_train, X_test, y_test, test_index, alpha=0.1, B=100, stride=1):
        """Compute prediction intervals using ENBPI approach"""
        self.current_horizon = self.horizons[0]  # Set current horizon
        
        # Train bootstrap models and get predictions
        boot_predictions, self.ensemble_online_resid = self.fit_bootstrap_models_with_OOB(
            X_train, y_train, X_test, B
        )

        n = len(X_train)
        
        # Calculate prediction intervals for test set
        test_preds = np.median(boot_predictions[:, n:], axis=0)
        
        # Calculate interval width using strided residuals
        width = np.percentile(self.strided_app(
            self.ensemble_online_resid, n, stride), 
            [alpha/2 * 100, (1-alpha/2) * 100], 
            axis=-1
        )
        
        q = np.percentile(self.ensemble_online_resid, (1 - alpha) * 100)

        lower = test_preds - q
        upper = test_preds + q

        
        return pd.DataFrame({
            'predicted': test_preds,
            'lower': lower,
            'upper': upper,
            'actual': y_test
        }, index=test_index)

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
    model = XGBoostENBPI(horizons=[14, 24, 38])

    for horizon in model.horizons:
        print(f"\nProcessing horizon t+{horizon}h...")

        X_train, y_train = model.prepare_data(train_df, horizon)
        X_test, y_test = model.prepare_data(test_df, horizon)

        results = model.compute_prediction_intervals(
            X_train.values, y_train.values,
            X_test.values, y_test.values, test_df.index,
            alpha=0.1, B=20, stride=1
        )

        # Coverage and width
        coverage = np.mean((results['actual'] >= results['lower']) & (results['actual'] <= results['upper'])) * 100
        mean_width = np.mean(results['upper'] - results['lower'])

        print(f"Coverage: {coverage:.1f}%, Mean Width: {mean_width:.2f}")

        # Plot
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




