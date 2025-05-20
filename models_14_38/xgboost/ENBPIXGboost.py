

"""
XGBoost model with ENBPI (Ensemble Neural Bootstrap Prediction Interval) for uncertainty quantification.
Combines the XGBoost model from xgboost_clean_full_features.py with prediction intervals from ENBPIXGboost.py.
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
from XGboostCV.xgboost_clean_full_features import XGBoostclean

class XGBoostENBPI(XGBoostclean):
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
    
    def fit_bootstrap_models(self, X_train, y_train, X_test, y_test, B=20):
        """Train B bootstrap models and calculate predictions"""
        n = len(X_train)
        n1 = len(X_test)
        boot_samples_idx = self.generate_bootstrap_samples(n, n, B)
        boot_predictions = np.zeros((B, (n+n1)), dtype=float)
        
        for b in range(B):
            # Get hyperparameters for current horizon
            params = self.get_hyperparameters(self.current_horizon)
            model = xgb.XGBRegressor(**params)
            
            # Train on bootstrap sample
            model.fit(X_train[boot_samples_idx[b]], y_train[boot_samples_idx[b]])
            
            # Store model
            self.ensemble_fitted_models.append(model)
            
            # Make predictions on both train and test data
            boot_predictions[b] = model.predict(np.vstack((X_train, X_test)))
            
        return boot_predictions
    
    def compute_prediction_intervals(self, X_train, y_train, X_test, y_test, alpha=0.1, B=100, stride=1):
        """Compute prediction intervals using ENBPI approach"""
        self.current_horizon = self.horizons[0]  # Set current horizon
        
        # Train bootstrap models and get predictions
        boot_predictions = self.fit_bootstrap_models(X_train, y_train, X_test, y_test, B)
        
        # Calculate residuals
        n = len(X_train)
        train_preds = np.mean(boot_predictions[:, :n], axis=0)
        self.ensemble_online_resid = y_train - train_preds
        
        # Calculate prediction intervals for test set
        test_preds = np.mean(boot_predictions[:, n:], axis=0)
        
        # Calculate interval width using strided residuals
        width = np.percentile(self.strided_app(
            self.ensemble_online_resid, n, stride), 
            [alpha/2 * 100, (1-alpha/2) * 100], 
            axis=-1
        )
        
        lower_quantile = np.percentile(self.ensemble_online_resid, alpha * 100)
        upper_quantile = np.percentile(self.ensemble_online_resid, (1 - alpha) * 100)

        # Create prediction intervals
        lower = test_preds + lower_quantile
        upper = test_preds + upper_quantile
        
        return pd.DataFrame({
            'predicted': test_preds,
            'lower': lower,
            'upper': upper,
            'actual': test_df[f'target_t{self.current_horizon}'].values
        }, index=test_df.index)

def main():

    start_time = time.time()  # ⏱ Start timer
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_nooutliers.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Train-test split
    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    
    train_df = data[train_start:train_end]
    test_df = data[test_start:test_end]
    
    # Initialize model for only horizon 14
    model = XGBoostENBPI(horizons=[14])
    
    # Train and evaluate with prediction intervals
    for horizon in model.horizons:
        print(f"\nProcessing horizon t+{horizon}h...")
        
        # Prepare data
        X_train, y_train = model.prepare_data(train_df, horizon)
        X_test, y_test = model.prepare_data(test_df, horizon)
        
        # Compute prediction intervals
        results = model.compute_prediction_intervals(
            X_train.values, y_train.values,
            X_test.values, y_test.values,
            alpha=0.1,  # 90% prediction intervals
            B=20,      # Number of bootstrap samples
            stride=1
        )
        
        # Calculate coverage and mean width
        coverage = np.mean((results['actual'] >= results['lower']) & 
                          (results['actual'] <= results['upper'])) * 100
        mean_width = np.mean(results['upper'] - results['lower'])
        
        print(f"90% Prediction Interval Coverage: {coverage:.1f}%")
        print(f"Mean Interval Width: {mean_width:.2f}")
        
        # Plot results
        plt.figure(figsize=(15, 6))
        plt.fill_between(results.index, 
                        results['lower'], 
                        results['upper'], 
                        alpha=0.3, 
                        label='90% Prediction Interval')
        plt.plot(results.index, results['predicted'], 
                label='Predicted', linestyle='--')
        plt.plot(results.index, y_test, 
                label='Actual', color='black')
        plt.title(f'Price Predictions with Uncertainty Bands (t+{horizon}h)')
        plt.xlabel('Time')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        out_dir = 'models_14_38/xgboost/plots/enbpi'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/predictions_with_intervals_h{horizon}.png')
        plt.close()
 
    end_time = time.time()  # ⏱ Stop timer
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main()




