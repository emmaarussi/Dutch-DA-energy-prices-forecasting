"""
XGBoostLSS implementation for energy price forecasting.
This model uses distributional regression to predict both the mean and variance of electricity prices.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import shap
from xgboostlss import XGBoostLSS
from xgboostlss.distributions import Normal

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

class XGBoostLSSForecaster:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        self.models = {}  # One model per horizon
        self.scalers = {}  # One scaler per horizon
        self.feature_importance = {}
        
    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon."""
        # Define excluded feature patterns
        excluded_patterns = [
            'forecast',  # Exclude forecast features for now
            'target_t'   # Exclude other target horizons
        ]
        
        # Get all columns except target columns and excluded features
        feature_cols = [col for col in data.columns 
                       if not col.startswith('target_t') and
                       not any(pattern in col for pattern in excluded_patterns)]
        
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y
    
    def get_hyperparameters(self, horizon):
        """Get hyperparameters based on forecast horizon."""
        if horizon <= 24:
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
        elif horizon <= 31:
            return {
                'max_depth': 6,
                'learning_rate': 0.025,
                'n_estimators': 2800,
                'min_child_weight': 5,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 0.25,
                'random_state': 42
            }
        else:
            return {
                'max_depth': 5,
                'learning_rate': 0.02,
                'n_estimators': 3000,
                'min_child_weight': 6,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'gamma': 0.3,
                'random_state': 42
            }

    def train_and_evaluate(self, train_data, test_data, horizon):
        """Train LSS model and evaluate its performance."""
        print(f"\nTraining models for t+{horizon}h horizon...")
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, horizon)
        X_test, y_test = self.prepare_data(test_data, horizon)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[horizon] = scaler
        
        # Convert to dataframes with column names (for SHAP)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Initialize and train LSS model
        params = self.get_hyperparameters(horizon)
        model = XGBoostLSS(
            distribution=Normal(),
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            random_state=params['random_state']
        )
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        self.models[horizon] = model
        
        # Make predictions
        dist_pred_test = model.predict(X_test_scaled)
        y_pred_test = dist_pred_test.loc  # Mean predictions
        y_std_test = dist_pred_test.scale  # Standard deviation predictions
        
        # Calculate confidence intervals
        lower_bound = y_pred_test - 1.96 * y_std_test
        upper_bound = y_pred_test + 1.96 * y_std_test
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred_test)
        coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound)) * 100
        
        # Calculate SHAP values for feature importance
        explainer = shap.TreeExplainer(model.distribution.loc_model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Store feature importance based on SHAP values
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        self.feature_importance[horizon] = importance_df
        
        # Print results
        print(f"\n📊 Evaluation for t+{horizon}h horizon:")
        print(f"Number of predictions: {len(y_test)}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R2: {metrics['R2']:.4f}")
        print(f"95% Prediction Interval Coverage: {coverage:.1f}%")
        
        # Create plots directory
        out_dir = 'models_14_38/xgboost/plots/lss'
        os.makedirs(out_dir, exist_ok=True)
        
        # Plot predictions with uncertainty
        plt.figure(figsize=(15, 6))
        plt.plot(test_data.index, y_test, label='Actual', alpha=0.7)
        plt.plot(test_data.index, y_pred_test, label='Predicted (Mean)', alpha=0.7)
        plt.fill_between(test_data.index, lower_bound, upper_bound,
                        alpha=0.2, label='95% Prediction Interval')
        plt.title(f'Price Predictions with Uncertainty - t+{horizon}h')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/predictions_h{horizon}.png')
        plt.close()
        
        return {
            'predictions': pd.DataFrame({
                'actual': y_test,
                'predicted_mean': y_pred_test,
                'predicted_std': y_std_test,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }),
            'metrics': metrics,
            'coverage': coverage
        }

    def plot_feature_importance(self, horizon, top_n=20):
        """Plot feature importance based on SHAP values."""
        importance_df = self.feature_importance.get(horizon)
        if importance_df is None or importance_df.empty:
            print(f"No importance data for horizon t+{horizon}.")
            return
            
        plt.figure(figsize=(12, 8))
        importance_df = importance_df.nlargest(top_n, 'importance')
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'Feature Importance (SHAP values) - t+{horizon}h')
        plt.xlabel('Mean |SHAP value|')
        plt.ylabel('Feature')
        
        out_dir = 'models_14_38/xgboost/plots/lss'
        os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/feature_importance_h{horizon}.png')
        plt.close()

def main():
    """Train and evaluate XGBoostLSS models."""
    # Load data
    print("Loading data...")
    data_dir = Path('data/features')
    train_df = pd.read_csv(data_dir / 'train_features.csv', index_col=0, parse_dates=True)
    test_df = pd.read_csv(data_dir / 'test_features.csv', index_col=0, parse_dates=True)
    
    # Initialize model
    model = XGBoostLSSForecaster(horizons=[14, 24, 38])  # Train for key horizons
    results = {}
    
    # Train and evaluate for each horizon
    for horizon in model.horizons:
        results[horizon] = model.train_and_evaluate(train_df, test_df, horizon)
        model.plot_feature_importance(horizon)
    
    # Save predictions
    predictions_dir = Path('predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    all_predictions = []
    for horizon in results:
        horizon_preds = results[horizon]['predictions']
        horizon_preds['horizon'] = horizon
        all_predictions.append(horizon_preds)
    
    pd.concat(all_predictions).to_csv(predictions_dir / 'xgboostlss_predictions.csv')

if __name__ == '__main__':
    main()
