
"""
📆 Evaluation: 2024-03-08 to 2024-03-15

⏱ Forecasting t+14h...
RMSE: 22.95 | Coverage: 53.85%

⏱ Forecasting t+24h...
RMSE: 22.96 | Coverage: 69.23%

⏱ Forecasting t+38h...
RMSE: 19.32 | Coverage: 66.86%

📆 Evaluation: 2024-03-15 to 2024-03-22

⏱ Forecasting t+14h...
RMSE: 15.13 | Coverage: 75.15%

⏱ Forecasting t+24h...
RMSE: 14.94 | Coverage: 71.60%

⏱ Forecasting t+38h...
RMSE: 20.34 | Coverage: 65.68%

📆 Evaluation: 2024-03-22 to 2024-03-29

⏱ Forecasting t+14h...
RMSE: 31.33 | Coverage: 49.70%

⏱ Forecasting t+24h...
RMSE: 31.90 | Coverage: 53.85%

⏱ Forecasting t+38h...
RMSE: 29.17 | Coverage: 52.07%

📆 Evaluation: 2024-03-29 to 2024-04-05

⏱ Forecasting t+14h...
RMSE: 28.87 | Coverage: 46.75%

⏱ Forecasting t+24h...
RMSE: 29.31 | Coverage: 43.20%

⏱ Forecasting t+38h...
RMSE: 38.55 | Coverage: 49.70%

📆 Evaluation: 2024-04-05 to 2024-04-12

⏱ Forecasting t+14h...
RMSE: 45.63 | Coverage: 32.54%

⏱ Forecasting t+24h...
RMSE: 45.64 | Coverage: 35.50%

⏱ Forecasting t+38h...
RMSE: 43.25 | Coverage: 37.28%

📆 Evaluation: 2024-04-12 to 2024-04-19

⏱ Forecasting t+14h...
RMSE: 43.34 | Coverage: 18.34%

⏱ Forecasting t+24h...
RMSE: 45.11 | Coverage: 19.53%

⏱ Forecasting t+38h...
RMSE: 40.89 | Coverage: 27.81%

✅ Finished in 349.31 seconds
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


class XGBOOSTLSSFixed:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        self.models = {}

    def get_xgboostlss_model(self):
        return XGBoostLSS(Gaussian())

    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon"""
        # Get all columns except target columns
        feature_cols = [col for col in data.columns 
                       if not col.startswith('target_t')]
        
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y

    def train_models(self, train_df):
        for horizon in self.horizons:
            print(f"\n🎯 Training model for t+{horizon}h")
            X_train, y_train = self.prepare_data(train_df, horizon)
            model = self.get_xgboostlss_model()
            dtrain = xgb.DMatrix(X_train, label=y_train)

            param_dict = {
                "eta":              ["float", {"low": 1e-5, "high": 1, "log": True}],
                "max_depth":        ["int",   {"low": 1, "high": 10, "log": False}],
                "gamma":            ["float", {"low": 1e-8, "high": 40, "log": True}],
                "subsample":        ["float", {"low": 0.2, "high": 1.0, "log": False}],
                "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
                "min_child_weight": ["float", {"low": 1e-8, "high": 500, "log": True}]
            }

            opt_param = model.hyper_opt(
                param_dict, dtrain,
                num_boost_round=100, nfold=5,
                early_stopping_rounds=20,
                max_minutes=10, n_trials=30,
                silence=True, seed=123, hp_seed=123
            )
            n_rounds = opt_param.pop("opt_rounds")
            model.train(params=opt_param, dtrain=dtrain, num_boost_round=n_rounds, verbose_eval=10)
            self.models[horizon] = model

    def predict_and_evaluate(self, test_df, horizons, start, end, out_dir):
        results = {h: [] for h in horizons}
        for horizon in horizons:
            print(f"\n⏱ Forecasting t+{horizon}h...")
            test_slice = test_df.loc[start:end]
            if len(test_slice) == 0:
                print("⚠️ Empty test slice.")
                continue

            X_test, y_test = self.prepare_data(test_slice, horizon)
            dtest = xgb.DMatrix(X_test)
            predictions = self.models[horizon].predict(dtest)
            pred_mean = predictions['loc'].values
            pred_std = predictions['scale'].values

            z = 1.645
            lower = pred_mean - z * pred_std
            upper = pred_mean + z * pred_std
            residuals = y_test.values - pred_mean
            metrics = calculate_metrics(y_test.values, pred_mean)
            coverage = np.mean((y_test.values >= lower) & (y_test.values <= upper))
            mean_width = np.mean(upper - lower)

            print(f"RMSE: {metrics['RMSE']:.2f} | Coverage: {coverage:.2%}")

            results[horizon].append({
                "RMSE": metrics["RMSE"],
                "RMSE_std": np.std(residuals),
                "coverage": coverage,
                "interval_width": mean_width
            })

            plt.figure(figsize=(10, 6))
            plt.plot(y_test.index, y_test.values, label='Actual')
            plt.plot(y_test.index, pred_mean, label='Predicted', color='orange')
            plt.fill_between(y_test.index, lower, upper, color='orange', alpha=0.2, label='90% CI')
            plt.title(f'Forecast for t+{horizon}h')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / f"forecast_h{horizon}.png")
            plt.close()

            
        return results


def main():
    start_time = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')

    print("📁 Loading data...")
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

    train_df.index = pd.to_datetime(train_df.index, utc=True).tz_convert(None)
    test_df.index = pd.to_datetime(test_df.index, utc=True).tz_convert(None)

    model = XGBOOSTLSSFixed(horizons=[14, 24, 38])
    model.train_models(train_df)

    window_size = pd.Timedelta(days=7)
    step_size = pd.Timedelta(days=7)
    start_date = pd.to_datetime("2024-03-08")
    end_date = test_df.index.max() - window_size
    results = {h: [] for h in model.horizons}

    current_start = start_date
    while current_start + window_size <= end_date:
        current_end = current_start + window_size
        print(f"\n📆 Evaluation: {current_start.date()} to {current_end.date()}")
        weekly_results = model.predict_and_evaluate(test_df, model.horizons, current_start, current_end, Path("outputs"))
        for h in model.horizons:
            if weekly_results[h]:
                results[h].append(weekly_results[h][0])
        current_start += step_size

    output_dir = Path("models_14_38/xgboost/plots/lss_noretrain")
    output_dir.mkdir(parents=True, exist_ok=True)

    for h in model.horizons:
        metric_df = pd.DataFrame(results[h])
        metric_df['window'] = range(len(metric_df))

        plt.figure(figsize=(8, 4))
        plt.plot(metric_df['window'], metric_df['RMSE'], marker='o', label='RMSE')
        plt.fill_between(metric_df['window'],
                         metric_df['RMSE'] - metric_df['RMSE_std'],
                         metric_df['RMSE'] + metric_df['RMSE_std'],
                         alpha=0.2, label='±1 std')
        plt.title(f'Residual Drift (Fixed Model) - Horizon t+{h}h')
        plt.xlabel('Evaluation Window')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"residual_drift_h{h}.png")
        plt.close()
        metric_df.to_csv(output_dir / f"drift_metrics_h{h}.csv", index=False)

    print(f"\n✅ Finished in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
