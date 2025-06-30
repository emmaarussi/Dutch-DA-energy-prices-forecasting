"""
XGBoost model optimized for features and hyperparameters. 

This model will be used to construct CI later with different methods.

📊 Final Metrics on FULL TEST SET:

t+14h:
  MAE: 14.17
  RMSE: 18.63
  SMAPE: 23.95
  WMAPE: 18.61
  R2: 0.41

t+24h:
  MAE: 14.19
  RMSE: 18.50
  SMAPE: 23.02
  WMAPE: 18.60
  R2: 0.42

t+38h:
  MAE: 17.33
  RMSE: 22.75
  SMAPE: 27.94
  WMAPE: 22.67
  R2: 0.09

✅ Completed full rolling evaluation in 353.73 seconds


📊 Final Metrics on FULL TEST SET:

t+14h:
  MAE: 14.17
  RMSE: 18.63
  SMAPE: 23.95
  WMAPE: 18.61
  R2: 0.41

t+24h:
  MAE: 14.19
  RMSE: 18.50
  SMAPE: 23.02
  WMAPE: 18.60
  R2: 0.42

t+38h:
  MAE: 17.33
  RMSE: 22.75
  SMAPE: 27.94
  WMAPE: 22.67
  R2: 0.09

✅ Completed full rolling evaluation in 1677.69 seconds



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
        excluded_patterns = [
            'wind', 'Wind', 'WIND',
            'solar', 'Solar', 'SOLAR',
            'coal', 'Coal', 'COAL',
            'consumption', 'Consumption', 'CONSUMPTION',
            'load', 'Load', 'LOAD'
        ]
        
        # Get all columns except target columns and excluded features
        # Get all columns except target columns and excluded features
        feature_cols = [col for col in data.columns 
                       if not col.startswith('target_t') and 
                       not any(pattern in col for pattern in excluded_patterns)]
        
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y
        
    def train_and_predict(self, train_data, test_data, horizon):
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

    def evaluate_and_plot(self, y_test, pred_mean, horizon, out_dir, fold_id=None):
        """Evaluate model and save forecast plot with prediction intervals."""
        
        residuals = y_test.values - pred_mean
        metrics = calculate_metrics(y_test.values, pred_mean)

        print(f"\n📊 Evaluation for t+{horizon}h:")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R²: {metrics['R2']:.4f}")
        print(pd.Series(residuals).describe())

        # Create predictions DataFrame
        plot_df = pd.DataFrame({
            'actual': y_test,
            'predicted': pred_mean,
        }, index=y_test.index)

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df.index, plot_df['actual'], label='Actual', alpha=0.7)
        plt.plot(plot_df.index, plot_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices - t+{horizon}h')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f'predictions_{horizon}_fold{fold_id}.png')
        plt.close()

        # Return metrics and residuals for drift analysis
        return {
            "RMSE": metrics["RMSE"],
            "SMAPE": metrics["SMAPE"],
            "R2": metrics["R2"],
            'residuals': residuals,
            'predictions': plot_df,
            'metrics': metrics,
        }

    
def main():
    print("Starting script...")
    start_time = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load full dataset
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)
    data = data.sort_index()

    print(f"📁 Data loaded from {features_path}")
    print(f"🗓 Full data range: {data.index.min()} to {data.index.max()}")

    model = XGBoostOptimized()
    horizons = [14, 24, 38]
    out_dir = Path("models_14_38/xgboost/plots/optimizednoexogenousretrain")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rolling window config
    window_size = pd.Timedelta(days=365)
    test_window = pd.Timedelta(days=7)
    step_size = pd.Timedelta(days=7)

    rolling_start = pd.to_datetime("2024-01-01")
    rolling_end = data.index.max() - test_window
    current_test_start = rolling_start
    fold_id = 0

    all_preds = []       # Collect all predictions here
    all_metrics = {h: [] for h in horizons}

    while current_test_start + test_window <= rolling_end:
        train_end = current_test_start
        train_start = train_end - window_size
        test_end = current_test_start + test_window

        print(f"\n📆 Rolling window: Train {train_start.date()} to {train_end.date()} | Test {current_test_start.date()} to {test_end.date()}")

        for horizon in horizons:
            print(f"\n⏱ Forecasting t+{horizon}h...")

            train_slice = data.loc[train_start:train_end]
            test_slice = data.loc[current_test_start:test_end]

            if len(train_slice) == 0 or len(test_slice) == 0:
                print(f"⚠️ Skipping window due to empty data.")
                continue

            # Train model and get predictions
            results = model.train_and_predict(train_slice, test_slice, horizon)
            pred_df = results['predictions'].copy()
            pred_df['horizon'] = horizon
            pred_df['forecast_start'] = current_test_start
            pred_df = pred_df.reset_index().rename(columns={'index': 'target_time'})
            all_preds.append(pred_df)

            # Save per-fold metrics
            all_metrics[horizon].append({
                "fold": fold_id,
                "RMSE": results['metrics']["RMSE"],
                "SMAPE": results['metrics']["SMAPE"],
                "R2": results['metrics']["R2"]
            })

            # Plot prediction line
            model.evaluate_and_plot(
                y_test=results['predictions']['actual'],
                pred_mean=results['predictions']['predicted'].values,
                horizon=horizon,
                out_dir=out_dir,
                fold_id=fold_id
            )

            # Plot top 20 feature importance
            importance_df = results['feature_importance'].nlargest(20, 'importance')

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(importance_df['feature'], importance_df['importance'])
            ax.set_title(f'Top 20 Feature Importance - t+{horizon}h Fold {fold_id}')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(out_dir / f'feature_importance_top20_{horizon}_fold{fold_id}.png')
            plt.close()

        current_test_start += step_size
        fold_id += 1

    # Save full predictions
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    all_preds_df.to_csv(out_dir / 'full_predictions.csv', index=False)

    # Save and report per-fold metrics
    for h in horizons:
        df = pd.DataFrame(all_metrics[h])
        df['horizon'] = h
        df.to_csv(out_dir / f'metrics_h{h}.csv', index=False)

    print("\n📊 Final Metrics on FULL TEST SET:")
    for h in horizons:
        h_df = all_preds_df[all_preds_df['horizon'] == h]
        metrics = calculate_metrics(h_df['actual'], h_df['predicted'])
        print(f"\nt+{h}h:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}")

    print(f"\n✅ Completed full rolling evaluation in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()


