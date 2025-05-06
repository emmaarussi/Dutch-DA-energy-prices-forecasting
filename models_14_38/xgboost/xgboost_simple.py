"""
Simple XGBoost model for energy price forecasting with:
- Single train-test split
- Manual hyperparameters
- No cross-validation
- No hyperparameter optimization
- No retraining
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance

class SimpleXGBoost:
    def __init__(self):
        self.horizons = range(14, 39)  # t+14 to t+38 hours
        self.models = {}  # Store one model per horizon
        
    
    def prepare_data(self, df, horizon):
        """Prepare features and target for a specific horizon."""
        # Drop target columns except for the specific horizon
        target_cols = [col for col in df.columns if col.startswith('target_t')]
        feature_cols = [col for col in df.columns if col not in target_cols]
        
        # Select features and target
        X = df[feature_cols]
        y = df[f'target_t{horizon}']
        
        return X, y
    
    def train_and_predict(self, train_df, test_df):
        """Train models and make predictions for all horizons."""
        results = {}
        
        for horizon in self.horizons:
            print(f"\nTraining model for t+{horizon}h horizon...")
            
            # Prepare data
            X_train, y_train = self.prepare_data(train_df, horizon)
            X_test, y_test = self.prepare_data(test_df, horizon)
            
            # Train model
            model = xgb.XGBRegressor(**self.get_hyperparameters())
            model.fit(X_train, y_train)
            
            # Store model
            self.models[horizon] = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            
            # Store results
            results[horizon] = {
                'metrics': metrics,
                'predictions': pd.DataFrame({
                    'actual': y_test,
                    'predicted': y_pred
                }, index=test_df.index),
                'feature_importance': pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            }
            
            # Print metrics
            print(f"\nMetrics for t+{horizon}h horizon:")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"SMAPE: {metrics['SMAPE']:.2f}%")
            print(f"R2: {metrics['R2']:.4f}")
            
        return results


    def get_hyperparameters(self):
        """Simple fixed hyperparameters for all horizons."""
        return {
            'max_depth': 6,
            'learning_rate': 0.03,
            'n_estimators': 2500,
            'min_child_weight': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.2,
            'random_state': 42
        }

def main():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    df = pd.read_csv(features_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')
    
    # Define train-test split
    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    
    train_df = df[train_start:train_end]
    test_df = df[test_start:test_end]
    
    # Train and evaluate model
    model = SimpleXGBoost()
    results = model.train_and_predict(train_df, test_df)
    
    # Create output directory
    out_dir = 'models_14_38/xgboost/plots/simple'
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot results for key horizons
    key_horizons = [14, 24, 38]
    for horizon in key_horizons:
        predictions_df = results[horizon]['predictions']
        plt.figure(figsize=(15, 6))
        plt.plot(predictions_df.index, predictions_df['actual'], label='Actual', alpha=0.7)
        plt.plot(predictions_df.index, predictions_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (t+{horizon}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{out_dir}/predictions_h{horizon}.png')
        plt.close()

        
        # Plot feature importance
        plot_feature_importance(
            results[horizon]['feature_importance'],
            top_n=20,
            title=f'Feature Importance (t+{horizon}h)',
            filename=f'{out_dir}/feature_importance_h{horizon}.png'
        )
        plt.close()

if __name__ == "__main__":
    main()
