"""
Simple linear model combining:
1. Significant price lags, here we are looking at lags of the price that are either the closes one available form the horizon distance, or lags that are exactly 24, 48 or 168 away from the predicted horizon.

2. Calendar features (from dummy model)

Features:
- Price lags: Selected significant lags for each horizon
- Hour of day (sine and cosine encoded)
- Day of week (sine and cosine encoded)
- Month of year (sine and cosine encoded)
- Calendar effects (is_weekend, is_holiday)
- Time of day effects (is_morning, is_evening)

Metrics for 365D window:
  Horizon t+14h:
    MAE: 13.11
    RMSE: 16.93
    SMAPE: 23.14
    WMAPE: 18.99
    R2: 0.40
  Horizon t+24h:
    MAE: 14.13
    RMSE: 18.06
    SMAPE: 24.16
    WMAPE: 20.54
    R2: 0.31
  Horizon t+38h:
    MAE: 15.70
    RMSE: 20.03
    SMAPE: 26.00
    WMAPE: 22.83
    R2: 0.13

Why do this you might ask, well, perhaps it simply works, and some literature from 2005 said its worth trying
so here we are. We are not even AICing this, its just based on a hunch.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV



class RFECVLinearLagsAndDummies:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        self.models = {}  # One model per horizon
        self.scalers = {}  # One scaler per horizon
        self.feature_importance = {}
        
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
        # Prepare features and target for training data
        X_train, y_train = self.prepare_data(train_data, horizon)
        
        # Standardize features for stability
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers[horizon] = scaler  # Save the scaler for this horizon
        
        # Step 1: Feature Selection with RFECV (OLS as estimator)
        print(f"Running RFECV for feature selection on t+{horizon}h...")
        ols = LinearRegression()
        rfecv = RFECV(estimator=ols, step=1, cv=5, scoring='neg_mean_squared_error')
        rfecv.fit(X_train_scaled, y_train)
        
        # Get selected features
        selected_features = X_train.columns[rfecv.support_]
        print(f"Selected Features for t+{horizon}h:", list(selected_features))
        
        # If no features are selected, use all features (avoids crash)
        if len(selected_features) == 0:
            print(f"No features selected for t+{horizon}h, using all features.")
            selected_features = X_train.columns
        
        # Step 2: Fit OLS using the selected features
        X_train_selected = pd.DataFrame(X_train_scaled, columns=X_train.columns)[selected_features]
        model = LinearRegression()
        model.fit(X_train_selected, y_train)
        
        # Store the model and selected features
        self.models[horizon] = {
            'model': model,
            'features': selected_features,
            'scaler': scaler
        }

        # Make predictions on training data
        y_pred_train = model.predict(X_train_selected)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        
        # Prepare features and target for test data
        X_test, y_test = self.prepare_data(test_data, horizon)
        X_test_scaled = scaler.transform(X_test)
        X_test_selected = pd.DataFrame(X_test_scaled, columns=X_train.columns)[selected_features]
        
        # Make predictions on test data
        y_pred_test = model.predict(X_test_selected)
        
        # Calculate test metrics
        test_metrics = calculate_metrics(y_test, y_pred_test)
        
        # Store feature importance (based on coefficient magnitude)
        importance_df = pd.DataFrame({
            'feature': list(selected_features),
            'importance': abs(model.coef_)
        }).sort_values('importance', ascending=False)
        self.feature_importance[horizon] = importance_df

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': pd.DataFrame({'actual': y_test, 'predicted': y_pred_test}, index=X_test.index)
        }

    def plot_feature_importance(self, horizon, top_n=20, filename=None):
        """Plot feature importance for a specific horizon."""
        importance_df = self.feature_importance.get(horizon)
        if importance_df is None or importance_df.empty:
            print(f"No importance data for horizon t+{horizon}.")
            return

        plt.figure(figsize=(12, 8))
        importance_df = importance_df.nlargest(top_n, 'importance')
        plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
        plt.title(f'Feature Importance (t+{horizon}h)')
        plt.xlabel('Absolute Coefficient Value')
        plt.ylabel('Feature')
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def predict(self, test_data):
        """Make predictions for all horizons."""
        predictions = pd.DataFrame(index=test_data.index)
        for h in self.horizons:
            X_test, _ = self.prepare_data(test_data, h)
            model_info = self.models[h]
            predictions[f'pred_t{h}'] = model_info['model'].predict(X_test)
        return predictions

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
    # Load data
    start_time = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    

    # Load full dataset
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)
    data = data.sort_index()

    test_start = '2024-01-01'
    train_data = data[data.index < test_start]
    test_data = data[data.index >= test_start]

    print(f"Full data range: {data.index.min()} to {data.index.max()}")
    print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

    model = RFECVLinearLagsAndDummies()
    horizons = [14, 24, 38]
    out_dir = Path("models_14_38/Linear_with_lags/RFECV_linear_Exogenous/plots/optimizedretrain")
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
                "RMSE": results['test_metrics']["RMSE"],
                "SMAPE": results['test_metrics']["SMAPE"],
                "R2": results['test_metrics']["R2"]
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
            importance_df = model.feature_importance[horizon].nlargest(20, 'importance')


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

    