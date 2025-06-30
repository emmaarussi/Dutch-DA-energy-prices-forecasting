
"""

this is with retraining!!!


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

    def train_and_predict(self, X_train, y_train, X_test):
        """Optimize hyperparameters, train model, and make predictions."""
        model = self.get_xgboostlss_model()  # create one model instance
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

        predictions = model.predict(dtest)  # now using trained model
        return predictions



    def evaluate_and_plot(self, y_test, pred_mean, pred_std, horizon, out_dir, fold_id=None):
        """Evaluate model and save forecast plot with prediction intervals."""
        z = 1.645  # for 90% CI
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std

        residuals = y_test.values - pred_mean
        metrics = calculate_metrics(y_test.values, pred_mean)
        coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
        mean_width = np.mean(upper - lower)

        print(f"\n📊 Evaluation for t+{horizon}h:")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R²: {metrics['R2']:.4f}")
        print(f"90% Prediction Interval Coverage: {coverage:.2%}")
        print(f"Mean Interval Width: {mean_width:.2f}")
        print("\nResidual Stats:")
        print(pd.Series(residuals).describe())

        # Plot
        plot_df = pd.DataFrame({
            "Actual": y_test,
            "Predicted": pred_mean,
            "Lower": lower,
            "Upper": upper
        }, index=y_test.index)

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df.index, plot_df["Actual"], label="Actual", alpha=0.7)
        plt.plot(plot_df.index, plot_df["Predicted"], label="Predicted", alpha=0.7)
        plt.fill_between(plot_df.index, plot_df["Lower"], plot_df["Upper"], alpha=0.2, label="90% CI")
        plt.title(f"Actual vs Predicted Prices - t+{horizon}h")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"predictions_{horizon}_fold{fold_id}.png")
        plt.close()

        # Return metrics and residuals for drift analysis
        return {
            "RMSE": metrics["RMSE"],
            "SMAPE": metrics["SMAPE"],
            "R2": metrics["R2"],
            "coverage": coverage,
            "interval_width": mean_width,
            "residuals": residuals
        }



def main():
    print("Starting script...")
    start_time = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load data
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')
    print(f"📁 Loading data...")
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

    # Ensure tz-naive datetime index
    train_df.index = pd.to_datetime(train_df.index, utc=True).tz_convert(None)
    test_df.index = pd.to_datetime(test_df.index, utc=True).tz_convert(None)

    print(f"🗓 Train range: {train_df.index.min()} to {train_df.index.max()}")
    print(f"🗓 Test range: {test_df.index.min()} to {test_df.index.max()}")

    model = XGBOOSTLSSCV()
    horizons = [14, 24, 38]
    all_metrics = {h: [] for h in horizons}

    # Rolling window config
    window_size = pd.Timedelta(days=365)
    test_window = pd.Timedelta(days=7)
    step_size = pd.Timedelta(days=7)

    rolling_start = pd.to_datetime("2024-03-08")
    rolling_end = test_df.index.max() - test_window
    current_test_start = rolling_start
    fold_id = 0


    while current_test_start + test_window <= rolling_end:
        train_end = current_test_start
        train_start = train_end - window_size
        test_end = current_test_start + test_window

        print(f"\n📆 Rolling window: Train {train_start.date()} to {train_end.date()} | Test {current_test_start.date()} to {test_end.date()}")

        for horizon in horizons:
            print(f"\n⏱ Forecasting t+{horizon}h...")

            # Slice data
            train_slice = train_df.loc[train_start:train_end]
            test_slice = test_df.loc[current_test_start:test_end]

            if len(train_slice) == 0 or len(test_slice) == 0:
                print(f"⚠️ Skipping window due to empty data.")
                continue

            X_train, y_train = model.prepare_data(train_slice, horizon)
            X_test, y_test = model.prepare_data(test_slice, horizon)

            predictions = model.train_and_predict(X_train, y_train, X_test)
            pred_mean = predictions['loc'].values
            pred_std = predictions['scale'].values

            out_dir = Path(f"models_14_38/xgboost/plots/lssretrain")
            result = model.evaluate_and_plot(y_test, pred_mean, pred_std, horizon, out_dir, fold_id)

            all_metrics[horizon].append({
                "RMSE": result["RMSE"],
                "RMSE_std": np.std(result["residuals"]),
                "coverage": result["coverage"],
                "interval_width": result["interval_width"]
            })

        current_test_start += step_size
        fold_id += 1

    # Save and plot residual drift
    results_dir = Path("models_14_38/xgboost/plots/lssretrain")
    results_dir.mkdir(parents=True, exist_ok=True)

    for h in horizons:
        metric_df = pd.DataFrame(all_metrics[h])
        metric_df['window'] = range(len(metric_df))

        plt.figure(figsize=(8, 4))
        plt.plot(metric_df['window'], metric_df['RMSE'], marker='o', label='RMSE')
        plt.fill_between(metric_df['window'],
                         metric_df['RMSE'] - metric_df['RMSE_std'],
                         metric_df['RMSE'] + metric_df['RMSE_std'],
                         alpha=0.2, label='±1 std')
        plt.title(f'Residual Drift (Retrained Model) - Horizon t+{h}h')
        plt.xlabel('Evaluation Window')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f"residual_drift_h{h}.png")
        plt.close()

        metric_df.to_csv(results_dir / f"drift_metrics_h{h}.csv", index=False)

    print("\n📊 Final Average Metrics Across All Rolling Windows:")
    for h in horizons:
        metric_df = pd.DataFrame(all_metrics[h])
        mean_rmse = metric_df["RMSE"].mean()
        std_rmse = metric_df["RMSE"].std()
        mean_coverage = metric_df["coverage"].mean()
        mean_width = metric_df["interval_width"].mean()

        # If SMAPE was included in evaluate_and_plot and returned:
        if "SMAPE" in metric_df.columns:
            mean_smape = metric_df["SMAPE"].mean()
            print(f"\nt+{h}h:")
            print(f"  RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
            print(f"  SMAPE: {mean_smape:.2f}%")
            print(f"  Coverage: {mean_coverage:.2f}")
            print(f"  Interval Width: {mean_width:.2f}")
        else:
            print(f"\nt+{h}h:")
            print(f"  RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
            print(f"  Coverage: {mean_coverage:.2f}")
            print(f"  Interval Width: {mean_width:.2f}")


    print(f"\n✅ Completed rolling window LSS evaluation in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()