import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import json
import os
from utils import calculate_metrics, plot_feature_importance, plot_predictions, plot_error_distribution, TimeSeriesCV

class EnergyPriceForecaster:
    def __init__(self, forecast_horizon=24):
        self.forecast_horizon = forecast_horizon
        self.models = {}  # One model per horizon
        self.feature_importance = {}
        self.metrics = {}
        
    def prepare_xy(self, data, horizon):
        """Prepare X and y for a specific horizon."""
        # Remove all target columns except the one we're training for
        feature_cols = [col for col in data.columns if not col.startswith('target_')]
        target_col = f'target_t{horizon}'
        
        X = data[feature_cols]
        y = data[target_col]
        
        return X, y
    
    def train(self, train_data, val_data=None, params=None):
        """Train models for all horizons."""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': ['rmse', 'mae'],
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'verbosity': 0
            }
        
        print(f"\nTraining models for {self.forecast_horizon} horizons...")
        
        for h in range(1, self.forecast_horizon + 1):
            print(f"\nTraining model for t+{h} horizon...")
            
            # Prepare data for this horizon
            X_train, y_train = self.prepare_xy(train_data, h)
            
            if val_data is not None:
                X_val, y_val = self.prepare_xy(val_data, h)
                eval_set = [(X_val, y_val)]
            else:
                # If no validation data, use a portion of training data
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, shuffle=False
                )
                eval_set = [(X_val, y_val)]
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Store model
            self.models[h] = model
            
            # Calculate feature importance
            importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            })
            self.feature_importance[h] = importance
            
            # Calculate metrics
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            self.metrics[h] = {
                'train': calculate_metrics(y_train, train_pred),
                'val': calculate_metrics(y_val, val_pred)
            }
            
            print(f"Horizon t+{h} validation metrics:")
            print(f"RMSE: {self.metrics[h]['val']['RMSE']:.2f}")
            print(f"MAPE: {self.metrics[h]['val']['MAPE']:.2f}%")
    
    def predict(self, X):
        """Make predictions for all horizons."""
        predictions = pd.DataFrame(index=X.index)
        
        for h in range(1, self.forecast_horizon + 1):
            # Get feature columns for this horizon
            feature_cols = [col for col in X.columns if not col.startswith('target_')]
            
            # Make predictions
            pred = self.models[h].predict(X[feature_cols])
            predictions[f'pred_t{h}'] = pred
        
        return predictions
    
    def save_model(self, path='data/models'):
        """Save the trained models and metadata."""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for h, model in self.models.items():
            model_path = os.path.join(path, f'model_h{h}.json')
            model.save_model(model_path)
        
        # Save feature importance
        importance_path = os.path.join(path, 'feature_importance.json')
        importance_dict = {
            str(h): imp.to_dict('records') 
            for h, imp in self.feature_importance.items()
        }
        with open(importance_path, 'w') as f:
            json.dump(importance_dict, f)
        
        # Save metrics
        metrics_path = os.path.join(path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)
    
    def load_model(self, path='data/models'):
        """Load trained models and metadata."""
        # Load models
        for h in range(1, self.forecast_horizon + 1):
            model_path = os.path.join(path, f'model_h{h}.json')
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            self.models[h] = model
        
        # Load feature importance
        importance_path = os.path.join(path, 'feature_importance.json')
        with open(importance_path, 'r') as f:
            importance_dict = json.load(f)
            self.feature_importance = {
                int(h): pd.DataFrame(imp) 
                for h, imp in importance_dict.items()
            }
        
        # Load metrics
        metrics_path = os.path.join(path, 'metrics.json')
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)

def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/features_scaled.csv', parse_dates=['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Set up time series cross-validation
    train_start = data.index.min()
    test_start = data.index.max() - pd.Timedelta(days=30)  # Use last 30 days as test set
    cv = TimeSeriesCV(
        train_start=train_start,
        test_start=test_start,
        end=data.index.max(),
        validation_window=7,  # 7 days validation window
        step_size=1  # Move forward 1 day at a time
    )
    
    # Initialize forecaster
    forecaster = EnergyPriceForecaster(forecast_horizon=24)
    
    # Train and evaluate models using time series CV
    cv_metrics = []
    for fold, dates in enumerate(cv.split()):
        print(f"\nFold {fold + 1}")
        print(f"Training period: {dates['train_start']} to {dates['train_end']}")
        print(f"Validation period: {dates['val_start']} to {dates['val_end']}")
        
        # Split data
        train_data = data[dates['train_start']:dates['train_end']]
        val_data = data[dates['val_start']:dates['val_end']]
        
        # Train models
        forecaster.train(train_data, val_data)
        
        # Store metrics
        for h, metrics in forecaster.metrics.items():
            cv_metrics.append({
                'fold': fold + 1,
                'horizon': h,
                'train_start': dates['train_start'],
                'train_end': dates['train_end'],
                'val_start': dates['val_start'],
                'val_end': dates['val_end'],
                **metrics['val']
            })
    
    # Save CV metrics
    cv_metrics_df = pd.DataFrame(cv_metrics)
    cv_metrics_df.to_csv('data/cv_metrics.csv', index=False)
    
    # Train final model on all data except test set
    print("\nTraining final model...")
    train_data = data[:test_start]
    test_data = data[test_start:]
    
    forecaster.train(train_data, test_data)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    predictions = forecaster.predict(test_data)
    predictions.to_csv('data/test_predictions.csv')
    
    # Plot results for first horizon
    h = 1
    y_true = test_data[f'target_t{h}']
    y_pred = predictions[f'pred_t{h}']
    
    # Plot predictions
    plt = plot_predictions(y_true, y_pred, title=f'Actual vs Predicted (t+{h} horizon)')
    plt.savefig('data/test_predictions.png')
    plt.close()
    
    # Plot error distribution
    plt = plot_error_distribution(y_true, y_pred, title=f'Error Distribution (t+{h} horizon)')
    plt.savefig('data/error_distribution.png')
    plt.close()
    
    # Plot feature importance
    plot_feature_importance(
        forecaster.feature_importance[h],
        title=f'Feature Importance (t+{h} horizon)'
    )
    
    # Save model
    print("\nSaving model...")
    forecaster.save_model()
    
    print("\nTraining complete! Results have been saved to the data directory.")

if __name__ == "__main__":
    main()
