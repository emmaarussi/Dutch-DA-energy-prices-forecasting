"""
XGBoost model optimized for features and hyperparameters. 

This model will be used to construct CI later with different methods.

⏱ Forecasting t+14h...

📊 Evaluation for t+14h:
RMSE: 20.12
SMAPE: 28.82%
R²: 0.2930

⏱ Forecasting t+24h...

📊 Evaluation for t+24h:
RMSE: 19.24
SMAPE: 25.04%
R²: 0.3476

⏱ Forecasting t+38h...

📊 Evaluation for t+38h:
RMSE: 22.54
SMAPE: 31.05%
R²: 0.0827

"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import time
from pathlib import Path
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


    def plot(self, y_test, pred_mean, horizon, out_dir):
        # Plot
        plot_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": pred_mean,
        }, index=y_test.index)

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df.index, plot_df["Actual"], label="Actual", alpha=0.7)
        plt.plot(plot_df.index, plot_df["Predicted"], label="Predicted", alpha=0.7)
        plt.title(f"Actual vs Predicted Prices - t+{horizon}h")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"predictions_{horizon}.png")
        plt.close()


    



def main():
    start_time = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load training data
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    print(f"📁 Loading features from: {features_path}")
    data = pd.read_csv(features_path, index_col=0, parse_dates=True)

    data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)  # Ensure time zone
    data = data.sort_index()

    # Train-test split
    test_start = pd.Timestamp("2024-01-01")
    train_df = data[data.index < test_start]
    test_df = data[data.index >= test_start]

    model = XGBoostOptimized()
    results_dir = Path("models_14_38/xgboost/plots/nocvoptimizedpointforecast")
    results_dir.mkdir(parents=True, exist_ok=True)

    for horizon in [14, 24, 38]:
        print(f"\n⏱ Forecasting t+{horizon}h...")

        # Train model and get results
        results = model.train_and_evaluate(train_df, test_df, horizon)
        pred_mean = results['predictions']['predicted']
        y_test = results['predictions']['actual']

        # Plot actual vs predicted
        model.plot(y_test, pred_mean, horizon, results_dir)

        # Print metrics
        print(f"\n📊 Evaluation for t+{horizon}h:")
        print(f"RMSE: {results['metrics']['RMSE']:.2f}")
        print(f"SMAPE: {results['metrics']['SMAPE']:.2f}%")
        print(f"R²: {results['metrics']['R2']:.4f}")


    print(f"\n✅ Done in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()