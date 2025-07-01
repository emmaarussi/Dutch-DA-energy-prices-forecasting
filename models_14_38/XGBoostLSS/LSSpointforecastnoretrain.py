
"""
Richer loss function: XGBoostLSS trains on the full negative log-likelihood of the distribution — this can lead to more calibrated and smoother predictions, not just minimizing MSE.

Joint learning: It jointly learns location (mean), scale (std), and shape (if applicable). That allows the model to better balance bias and variance.

Robustness: The Gaussian likelihood penalizes large deviations more heavily than squared error alone. In volatile regimes, this can help.

no retraining here

📈 Final Metrics (Fixed Model):
  Horizon t+14h:
    MAE: 12.02
    RMSE: 16.54
    SMAPE: 20.07
    WMAPE: 16.17
    R2: 0.53
  Horizon t+24h:
    MAE: 15.45
    RMSE: 20.83
    SMAPE: 23.99
    WMAPE: 20.74
    R2: 0.25
  Horizon t+38h:
    MAE: 15.78
    RMSE: 21.27
    SMAPE: 24.07
    WMAPE: 21.11
    R2: 0.20


"""


import numpy as np
import pandas as pd
import xgboost as xgb
import time
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian

# Custom metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics


class XGBOOSTLSSCV:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons

    def get_xgboostlss_model(self):
        """Initialize XGBoostLSS model with Gaussian distribution."""
        return XGBoostLSS(Gaussian())
    
    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon"""
        # Get all columns except target columns
        feature_cols = [col for col in data.columns 
                       if not col.startswith('target_t')]
        
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y

    def train_and_evaluate(self, train_data, test_data, horizon):
        """Train and evaluate model for a specific window and horizon"""
        # Prepare data
        X_train, y_train = self.prepare_data(train_data, horizon)
        X_test, y_test = self.prepare_data(test_data, horizon)

        model = self.get_xgboostlss_model()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        param_dict = {
            "eta":              ["float", {"low": 1e-5, "high": 1, "log": True}],
            "max_depth":        ["int",   {"low": 1, "high": 10, "log": False}],
            "gamma":            ["float", {"low": 1e-8, "high": 40, "log": True}],
            "subsample":        ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "min_child_weight": ["float", {"low": 1e-8, "high": 500, "log": True}]
        }

        print("🔧 Optimizing hyperparameters...")
        opt_param = model.hyper_opt(
            param_dict,
            dtrain,
            num_boost_round=100,
            nfold=5,
            early_stopping_rounds=20,
            max_minutes=10,
            n_trials=30,
            silence=True,
            seed=123,
            hp_seed=123
        )

        n_rounds = opt_param.pop("opt_rounds")
        model.train(params=opt_param, dtrain=dtrain, num_boost_round=n_rounds, verbose_eval=10)

        preds = model.predict(dtest)
        pred_mean = preds['loc'].values
        pred_std = preds['scale'].values
        # std = sqrt(variance)

        z = 1.645  # for 90% CI
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std

        metrics = calculate_metrics(y_test.values, pred_mean)
        coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
        mean_width = np.mean(upper - lower)

        print(f"\n📊 Evaluation for t+{horizon}h:")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R²: {metrics['R2']:.4f}")
        print(f"90% Prediction Interval Coverage: {coverage:.2%}")
        print(f"Mean Interval Width: {mean_width:.2f}")

        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': pred_mean,
            'lower': lower,
            'upper': upper
        }, index=test_data.index)

        # Return metrics and residuals for drift analysis
        return {
            "RMSE": metrics["RMSE"],
            "SMAPE": metrics["SMAPE"],
            "R2": metrics["R2"],
            "coverage": coverage,
            "interval_width": mean_width,
            'predictions': predictions_df,
            'metrics': metrics,
            'coverage': coverage,
            'interval_width': mean_width
        }

    def predict_only(self, model, test_data, horizon):
        X_test, y_test = self.prepare_data(test_data, horizon)
        dtest = xgb.DMatrix(X_test)

        preds = model.predict(dtest)
        pred_mean = preds['loc'].values
        pred_std = preds['scale'].values

        z = 1.645
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std

        metrics = calculate_metrics(y_test.values, pred_mean)
        coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
        mean_width = np.mean(upper - lower)

        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': pred_mean,
            'lower': lower,
            'upper': upper
        }, index=test_data.index)

        return {
            "RMSE": metrics["RMSE"],
            "SMAPE": metrics["SMAPE"],
            "R2": metrics["R2"],
            "coverage": coverage,
            "interval_width": mean_width,
            'predictions': predictions_df
        }





def main():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)
    data = data.sort_index()

    # Define forecast horizons
    horizons = [14, 24, 38]

    # Create safe target columns without leakage
    for h in horizons:
        data[f'target_t{h}'] = data['current_price'].shift(-h)

    # Drop rows with unknown targets
    data.dropna(subset=[f'target_t{h}' for h in horizons], inplace=True)

    # Define fixed training window (1 year before test starts)
    test_start = pd.Timestamp("2024-01-01")
    train_window = data[data.index < test_start]

    # Initialize model handler
    model_handler = XGBOOSTLSSCV(horizons=horizons)

    # Define hyperparameter search space (same as before)
    param_dict = {
        "eta":              ["float", {"low": 1e-5, "high": 1, "log": True}],
        "max_depth":        ["int",   {"low": 1, "high": 10, "log": False}],
        "gamma":            ["float", {"low": 1e-8, "high": 40, "log": True}],
        "subsample":        ["float", {"low": 0.2, "high": 1.0, "log": False}],
        "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
        "min_child_weight": ["float", {"low": 1e-8, "high": 500, "log": True}]
    }

    # Train once for each horizon
    trained_models = {}
    for h in horizons:
        print(f"\n🎯 Training fixed model for horizon t+{h}h")
        X_train, y_train = model_handler.prepare_data(train_window, h)
        model = model_handler.get_xgboostlss_model()
        dtrain = xgb.DMatrix(X_train, label=y_train)

        opt_param = model.hyper_opt(
            param_dict, dtrain,
            num_boost_round=100, nfold=5,
            early_stopping_rounds=20,
            max_minutes=10, n_trials=30,
            silence=True, seed=123, hp_seed=123
        )
        n_rounds = opt_param.pop("opt_rounds")
        model.train(params=opt_param, dtrain=dtrain, num_boost_round=n_rounds, verbose_eval=10)
        trained_models[h] = model

    # Rolling evaluation over 7-day windows
    window_predictions = []
    for forecast_start in pd.date_range(test_start, data.index.max(), freq='7D'):
        print(f"\n📅 Evaluating forecast starting: {forecast_start.date()}")
        window_length = pd.Timedelta(days=7)
        test_window = data[(data.index >= forecast_start) & (data.index < forecast_start + window_length)]

        for h in horizons:
            model = trained_models[h]
            result = model_handler.predict_only(model, test_window, h)

            pred_df = result['predictions'].copy()
            pred_df['horizon'] = h
            pred_df['forecast_start'] = forecast_start
            pred_df = pred_df.reset_index().rename(columns={'index': 'target_time'})

            for _, row in pred_df.iterrows():
                window_predictions.append({
                    'window_size': 'fixed',
                    'forecast_start': forecast_start,
                    'target_time': row['target_time'],
                    'horizon': row['horizon'],
                    'predicted': row['predicted'],
                    'actual': row['actual']
                })

    # Create final result DataFrame
    results_df = pd.DataFrame(window_predictions)

    # Summarize metrics
    print("\n📈 Final Metrics (Fixed Model):")
    for h in horizons:
        subset = results_df[results_df['horizon'] == h]
        metrics = calculate_metrics(subset['actual'], subset['predicted'])
        print(f"  Horizon t+{h}h:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.2f}")


    # Save plots
    output_dir = Path("models_14_38/xgboost/plots/lsspointforecast_noretrain")
    output_dir.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        h_df = results_df[results_df['horizon'] == h]
        plt.figure(figsize=(15, 6))
        plt.plot(h_df['target_time'], h_df['actual'], label='Actual', alpha=0.7)
        plt.plot(h_df['target_time'], h_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (Fixed Model, t+{h}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"predictions_over_time_fixed_{h}h.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Save raw results
    results_df.to_csv(output_dir / "fixed_model_predictions.csv", index=False)

    print("\n✅ Evaluation complete.")



if __name__ == "__main__":
    main()