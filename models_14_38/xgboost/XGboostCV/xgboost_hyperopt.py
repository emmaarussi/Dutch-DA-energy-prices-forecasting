"""
XGBoost model with hyperopt optimization for medium to long-term energy price forecasting.
Uses hyperopt to find optimal hyperparameters for each forecast horizon.
RMSE optimization
this model is already considerably slower due to the hyperparameter optimization and the rolling window approach,
which retrains the model for each window and each horizon on a weekly basis.
Also the window size is quite big, so that adds time too for training

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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance, rolling_window_evaluation

class XGBoostHyperopt:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        self.best_params = {}  # Store best parameters for each horizon
        
    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon"""
        feature_cols = [col for col in data.columns if not col.startswith('target_t')]
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y
        
    def objective(self, params, train_x, train_y, valid_x, valid_y):
        """Objective function for hyperopt optimization"""
        model = xgb.XGBRegressor(
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            n_estimators=int(params['n_estimators']),
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            random_state=42
        )
        
        model.fit(train_x, train_y)
        predictions = model.predict(valid_x)
        metrics = calculate_metrics(valid_y, predictions)
        
        # minimize RMSE
        return {'loss': metrics['RMSE'], 'status': STATUS_OK}
        
    def optimize_hyperparameters(self, train_data, valid_data, horizon, max_evals=100):
        """Find optimal hyperparameters using hyperopt"""
        X_train, y_train = self.prepare_data(train_data, horizon)
        X_valid, y_valid = self.prepare_data(valid_data, horizon)
        
        space = {
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'n_estimators': hp.quniform('n_estimators', 1000, 4000, 100),
            'min_child_weight': hp.uniform('min_child_weight', 1, 10),
            'subsample': hp.uniform('subsample', 0.5, 0.9),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.9),
            'gamma': hp.uniform('gamma', 0.1, 0.5),
        }
        
        trials = Trials()
        best = fmin(
            fn=lambda params: self.objective(params, X_train, y_train, X_valid, y_valid),
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        
        # Store best parameters
        self.best_params[horizon] = {
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'n_estimators': int(best['n_estimators']),
            'min_child_weight': best['min_child_weight'],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'gamma': best['gamma'],
            'random_state': 42
        }
        
        return self.best_params[horizon]
        
    def get_hyperparameters(self, horizon):
        """Get optimized hyperparameters for a specific horizon"""
        if horizon in self.best_params:
            return self.best_params[horizon]
        else:
            # Default parameters if optimization hasn't been run
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
            
    def train_and_evaluate_window(self, train_data, test_data, horizon):
        """Train and evaluate model for a specific window and horizon"""
        # First optimize hyperparameters if not already done
        if horizon not in self.best_params:
            # Use 80% of train_data for training and 20% for validation during optimization
            train_size = int(len(train_data) * 0.8)
            opt_train = train_data.iloc[:train_size]
            opt_valid = train_data.iloc[train_size:]
            self.optimize_hyperparameters(opt_train, opt_valid, horizon)
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, horizon)
        X_test, y_test = self.prepare_data(test_data, horizon)
        
        # Get optimized hyperparameters and train model
        params = self.get_hyperparameters(horizon)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, predictions)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, metrics, importance

def main():
    # Load multivariate features
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index() # Ensure tz-aware

    # Define train-test split
    train_start = pd.Timestamp('2023-01-08', tz='UTC')
    train_end = pd.Timestamp('2024-01-29', tz='UTC')
    test_start = pd.Timestamp('2024-01-29', tz='UTC')
    test_end = pd.Timestamp('2024-03-01', tz='UTC')

    train_df = data[train_start:train_end]
    test_df = data[test_start:test_end]
    train_data = train_df  # <-- define this for the next step

    # Initialize model
    model = XGBoostHyperopt(horizons=[14, 24, 38])  # Choose horizons of interest

    # Extract best hyperparameters for each horizon
    train_size = int(len(train_data) * 0.8)
    opt_train = train_data.iloc[:train_size]
    opt_valid = train_data.iloc[train_size:]

    for horizon in model.horizons:
        print(f"\n🔍 Optimizing hyperparameters for t+{horizon}h...")
        best = model.optimize_hyperparameters(opt_train, opt_valid, horizon, max_evals=30)

        print(f"\n✅ Best hyperparameters for t+{horizon}h:")
        for param, value in best.items():
            print(f"{param}: {value:.4f}" if isinstance(value, float) else f"{param}: {value}")

    # Optional: Evaluate model using the tuned params on the full train-test split
    results = {}
    for horizon in model.horizons:
        print(f"\nTraining and evaluating horizon t+{horizon}h...")
        results[horizon] = model.train_and_evaluate(train_df, test_df, horizon)

if __name__ == "__main__":
    main()

"""

def main():
    # Create output directories
    plots_dir = 'models_14_38/xgboost/plots_hyperopt'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load multivariate features
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Initialize model
    model = XGBoostHyperopt(horizons=[14, 24, 38])  # Key horizons
    
    # Use the rolling_window_evaluation from utils
    final_metrics, all_importance, all_predictions = rolling_window_evaluation(
        model, 
        data,
        window_size='365D',  # 1 year training window
        step_size='7D',      # Weekly steps
        test_size='7D'       # Weekly test size
    )
    
    # Print metrics and optimal parameters
    print("\nFinal Metrics and Optimal Parameters:")
    for horizon in final_metrics:
        print(f"\nHorizon t+{horizon}h:")
        print("\nMetrics:")
        for metric, value in final_metrics[horizon].items():
            print(f"{metric}: {value:.4f}")
        print("\nOptimal Parameters:")
        for param, value in model.best_params[horizon].items():
            print(f"{param}: {value:.4f}" if isinstance(value, float) else f"{param}: {value}")

    # Plot feature importance and predictions over time for each horizon
    for horizon in [14, 24, 38]:
        # Feature importance plot
        importance_df = pd.concat(all_importance[horizon]).groupby('feature').mean().reset_index()
        plot_feature_importance(
            importance_df,
            top_n=20,
            title=f'Feature Importance (t+{horizon}h)',
            filename=f'models_14_38/xgboost/plots_hyperopt/feature_importance_h{horizon}.png'
        )
        plt.close()
        
        # Predictions over time plot
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
        plt.savefig(f'models_14_38/xgboost/plots_hyperopt/predictions_over_time_h{horizon}.png')
        plt.close()

if __name__ == "__main__":
    main()
"""