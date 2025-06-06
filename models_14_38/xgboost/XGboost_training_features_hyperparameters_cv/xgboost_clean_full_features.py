"""
XGBoost model using full feature set called multivariate_features.csv.

It uses the utils.utils module for utility functions calculate_metrics, plot_feature_importance

Features:
1. Historical price data
2. Wind generation (total)
3. Solar generation
4. Consumption
5. Calendar features

No cross validation.


"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance

class XGBoostclean:
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
        """Get horizon-specific hyperparameters"""
        if horizon <= 24:  # Short-term
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
        elif horizon <= 31:  # Medium-term
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
        else:  # Long-term
            return {
                'max_depth': 6,
                'learning_rate': 0.02,
                'n_estimators': 3000,
                'min_child_weight': 6,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'gamma': 0.3,
                'random_state': 42
            }

def main():
    # Create output directories
    plots_dir = 'models_14_38/xgboost/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load multivariate features
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Initialize model
    model = XGBoostclean(horizons=[14, 24, 38]) 

    # Define train-test split
    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    
    train_df = data[train_start:train_end]
    test_df = data[test_start:test_end]
    
    # Train and evaluate
    results = {}
    for horizon in model.horizons:
        print(f"\nTraining and evaluating horizon t+{horizon}h...")
        results[horizon] = model.train_and_evaluate(train_df, test_df, horizon)
    
    # Plot predictions
    for horizon in model.horizons:
        result = results[horizon]
        predictions_df = result['predictions']
        metrics = result['metrics']
        print(f"\nt+{horizon}h horizon:")
        print(f"Number of predictions: {len(predictions_df)}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R2: {metrics['R2']:.4f}")

    # Create output directory
    out_dir = 'models_14_38/xgboost/plots/cleanfull'
    os.makedirs(out_dir, exist_ok=True)

    # Plot feature importance for each horizon
    for horizon in model.horizons:
        importance_df = results[horizon]['feature_importance']
        plot_feature_importance(
            importance_df,
            top_n=20,
            title=f'Feature Importance (t+{horizon}h)',
            filename=f'{out_dir}/feature_importance_h{horizon}.png'
        )
        plt.close()
        
    # Plot predictions over time
    for horizon in model.horizons:
        predictions_df = results[horizon]['predictions']
        plt.figure(figsize=(15, 6))
        plt.plot(predictions_df.index, predictions_df['actual'], label='Actual', alpha=0.7)
        plt.plot(predictions_df.index, predictions_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (t+{horizon}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/predictions_over_time_h{horizon}.png')
        plt.close()


if __name__ == "__main__":
    main()
