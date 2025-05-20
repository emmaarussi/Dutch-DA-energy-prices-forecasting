"""
XGBoost model optimized for features and hyperparameters. now used on full period of train and test data

This model will be used to construct CI later with different methods.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance

class XGBoostOptimized:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        
    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon"""
        # Define excluded feature patterns
        #excluded_patterns = [
            #'forecast'
        #]
        
        # Get all columns except target columns and excluded features
        feature_cols = [col for col in data.columns 
                       if not col.startswith('target_t')]#and 
                       #not any(pattern in col for pattern in excluded_patterns)]
        
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y
        
    def train_and_evaluate(self, train_data, test_data, horizon):
        """Train and evaluate model for a specific window and horizon"""
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, horizon)
        X_test, y_test = self.prepare_data(test_data, horizon)
        
        # Get hyperparameters and train model
        params = self.get_hyperparameters(horizon)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics using utils function
        metrics = calculate_metrics(y_test, predictions)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions
        }, index=test_data.index)

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': importance,
            'predictions': predictions_df
        }
        
    def get_hyperparameters(self, horizon):
        if horizon <= 24:
            return {
                'max_depth': 8, 'learning_rate': 0.0110, 'n_estimators': 1300,
                'min_child_weight': 3.1910, 'subsample': 0.8837, 'colsample_bytree': 0.8369,
                'gamma': 0.1393, 'random_state': 42
            }
        elif horizon <= 31:
            return {
                'max_depth': 9, 'learning_rate': 0.0693, 'n_estimators': 1400,
                'min_child_weight': 1.4524, 'subsample': 0.7879, 'colsample_bytree': 0.8769,
                'gamma': 0.1656, 'random_state': 42
            }
        else:  # for t+38h
            return {
                'max_depth': 9, 'learning_rate': 0.0179, 'n_estimators': 1100,
                'min_child_weight': 5.4722, 'subsample': 0.8081, 'colsample_bytree': 0.8531,
                'gamma': 0.2176, 'random_state': 42
            }


    
def plot_model_fit(y_true, y_pred, index, horizon, output_dir):
    residuals = y_true - y_pred

    # Actual vs Predicted over time
    plt.figure(figsize=(15, 6))
    plt.plot(index, y_true, label='Actual', color='black', alpha=0.7)
    plt.plot(index, y_pred, label='Predicted', color='blue', linestyle='--', alpha=0.7)
    plt.title(f"XGBoost Fit on Full Data (t+{horizon}h)")
    plt.xlabel("Time")
    plt.ylabel("Price (EUR/MWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fit_full_data_h{horizon}.png")
    plt.close()

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Prediction Scatter (t+{horizon}h)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_full_data_h{horizon}.png")
    plt.close()

    # Residual plot
    plt.figure(figsize=(15, 4))
    plt.plot(index, residuals, label="Residuals", alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residuals Over Time (t+{horizon}h)")
    plt.xlabel("Time")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_full_data_h{horizon}.png")
    plt.close()


def main():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Use full dataset (train + test)
    full_data = data.copy()
    
    # Initialize model
    model = XGBoostOptimized(horizons=[14, 24, 38]) 
    
    # Train on full dataset
    results = {}
    for horizon in model.horizons:
        print(f"\nTraining on full dataset for horizon t+{horizon}h...")
        results[horizon] = model.train_and_evaluate(full_data, full_data, horizon)

    # Plot model fit for each horizon
    fit_plot_dir = 'models_14_38/xgboost/plots/fit_full'
    os.makedirs(fit_plot_dir, exist_ok=True)

    for horizon in model.horizons:
        preds_df = results[horizon]['predictions']
        plot_model_fit(
            y_true=preds_df['actual'].values,
            y_pred=preds_df['predicted'].values,
            index=preds_df.index,
            horizon=horizon,
            output_dir=fit_plot_dir
        )

    # Save final models
    import joblib
    model_dir = 'models_14_38/xgboost/final_models'
    os.makedirs(model_dir, exist_ok=True)
    for horizon in model.horizons:
        joblib.dump(results[horizon]['model'], f"{model_dir}/xgboost_model_h{horizon}.pkl")

    print('\nAll models trained and saved. Ready for use in SBPI and EnbPI.')
