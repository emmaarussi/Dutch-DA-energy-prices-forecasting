"""
XGBoost model optimized for features and hyperparameters. 

This model will be used to construct CI later with different methods.

📊 Final Metrics on FULL TEST SET no retrain!!:





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
    out_dir = Path("models_14_38/xgboost/plots/optimized_noretrain")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Train once
    train_cutoff = pd.to_datetime("2024-01-01")
    train_data = data[data.index < train_cutoff]
    test_data = data[data.index >= train_cutoff]

    print(f"\n📚 Training XGBoost models ONCE on data up to {train_cutoff}...")
    trained_models = {}

    for horizon in horizons:
        trained_models[horizon] = model.train_and_predict(train_data, test_data, horizon)['model']

    # --- Step 2: Rolling evaluation without retraining
    window_size = pd.Timedelta(days=365)
    test_window = pd.Timedelta(days=7)
    step_size = pd.Timedelta(days=7)

    current_test_start = pd.to_datetime("2024-01-01")
    rolling_end = data.index.max() - test_window
    fold_id = 0

    all_preds = []
    all_metrics = {h: [] for h in horizons}

    while current_test_start + test_window <= rolling_end:
        test_start = current_test_start
        test_end = current_test_start + test_window
        test_slice = data.loc[test_start:test_end]

        print(f"\n📆 Fold {fold_id} — Test: {test_start.date()} to {test_end.date()}")

        for horizon in horizons:
            print(f"\n⏱ Predicting t+{horizon}h with fixed model...")

            # Prepare test data
            X_test, y_test = model.prepare_data(test_slice, horizon)

            # Use pre-trained model
            xgb_model = trained_models[horizon]
            predictions = xgb_model.predict(X_test)

            # Create prediction DataFrame
            predictions_df = pd.DataFrame({
                'actual': y_test,
                'predicted': predictions
            }, index=y_test.index)

            pred_df = predictions_df.copy()
            pred_df['horizon'] = horizon
            pred_df['forecast_start'] = test_start
            pred_df = pred_df.reset_index().rename(columns={'index': 'target_time'})
            all_preds.append(pred_df)

            # Metrics
            metrics = calculate_metrics(y_test, predictions)
            all_metrics[horizon].append({
                "fold": fold_id,
                "RMSE": metrics["RMSE"],
                "SMAPE": metrics["SMAPE"],
                "R2": metrics["R2"]
            })

            # Plot
            model.evaluate_and_plot(
                y_test=y_test,
                pred_mean=predictions,
                horizon=horizon,
                out_dir=out_dir,
                fold_id=fold_id
            )

        current_test_start += step_size
        fold_id += 1

    # --- Save results
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    all_preds_df.to_csv(out_dir / 'full_predictions.csv', index=False)

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

    print(f"\n✅ Completed full evaluation in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()


