"""
XGBoost model for medium to long-term energy price forecasting using new feature set called multivariate_features.csv.

It uses the utils.utils module for utility functions calculate_metrics, plot_feature_importance, plot_predictions, plot_error_distribution, rolling_window_evaluation

Features:
1. Historical price data
2. Wind generation (total)
3. Solar generation
4. Consumption
5. Calendar features

The model uses 12-month rolling windows for training, shifted by 1 week, evaluating on horizons from t+14 to t+38.


"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance, rolling_window_evaluation

class XGBoostMultivariate:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        
    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon"""
        # Get all columns except target columns
        feature_cols = [col for col in data.columns if not col.startswith('target_t')]
        X = data[feature_cols]
        y = data[f'target_t{horizon}.1']
        return X, y
        
    def train_and_evaluate_window(self, train_data, test_data, horizon):
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
        
        return model, metrics, importance
        
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
    model = XGBoostMultivariate(horizons=[14, 24, 38])  # Key horizons
    
    # Use the rolling_window_evaluation from utils
    final_metrics, all_importance, all_predictions = rolling_window_evaluation(
        model, 
        data,
        window_size='365D',  # 1 year training window
        step_size='7D',      # Weekly steps
        test_size='7D'       # Weekly test size
    )
    
    # Print metrics
    print("\nFinal Metrics:")
    for horizon in final_metrics:
        print(f"\nHorizon t+{horizon}h:")
        for metric, value in final_metrics[horizon].items():
            print(f"{metric}: {value:.4f}")

    # Plot feature importance for each horizon
    for horizon in [14, 24, 38]:
        importance_df = pd.concat(all_importance[horizon]).groupby('feature').mean().reset_index()
        plot_feature_importance(
            importance_df,
            top_n=20,
            title=f'Feature Importance (t+{horizon}h)',
            filename=f'models_14_38/xgboost/plots/feature_importance_h{horizon}.png'
        )
        plt.close()
        
        # Plot predictions over time
        predictions_df = all_predictions[horizon]
        plt.figure(figsize=(15, 6))
        plt.plot(predictions_df.index, predictions_df['actual'], label='Actual', alpha=0.7)
        plt.plot(predictions_df.index, predictions_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (t+{horizon}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'models_14_38/xgboost/plots/predictions_over_time_h{horizon}.png')
        plt.close()

if __name__ == "__main__":
    main()

