"""

✅ Best parameters:
eta: 0.07778298112954925
max_depth: 3
gamma: 7.120580346313052e-06
subsample: 0.7313459963706144
colsample_bytree: 0.7855880753079574
min_child_weight: 1.6838372382474876e-06
booster: gbtree
opt_rounds: 100

📊 Evaluation for t+14h:
RMSE: 6.61
SMAPE: 9.48%
R²: 0.9237
90% Prediction Interval Coverage: 92.43%
Mean Interval Width: 20.61

Residual Stats:
count    1401.000000
mean       -0.345639
std         6.603284
min       -48.981281
25%        -2.773571
50%        -0.095081
75%         2.376959
max        48.194931
dtype: float64


📊 Evaluation for t+24h:
RMSE: 5.46
SMAPE: 8.04%
R²: 0.9475
90% Prediction Interval Coverage: 85.87%
Mean Interval Width: 20.75

Residual Stats:
count    1401.000000
mean        0.282686
std         5.454135
min       -47.099632
25%        -2.185441
50%         0.175083
75%         2.639620
max        46.214020
dtype: float64



✅ Best parameters:
eta: 0.10573142939655886
max_depth: 2
gamma: 0.0021303995569248143
subsample: 0.8023637332827539
colsample_bytree: 0.37324608259375136
min_child_weight: 0.0692594672457401
booster: gbtree
opt_rounds: 100

📊 Evaluation for t+38h:
RMSE: 9.56
SMAPE: 12.42%
R²: 0.8351
90% Prediction Interval Coverage: 90.01%
Mean Interval Width: 30.15

Residual Stats:
count    1401.000000
mean        1.090025
std         9.498174
min       -55.307639
25%        -2.894054
50%         1.285010
75%         5.419681
max        60.263840
dtype: float64

✅ Completed LSS calibration in 432.11 seconds

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

        print("\n✅ Best parameters:")
        for param, value in opt_param.items():
            print(f"{param}: {value}")

        n_rounds = opt_param.pop("opt_rounds")
        model.train(params=opt_param, dtrain=dtrain, num_boost_round=n_rounds, verbose_eval=10)

        predictions = model.predict(dtest)  # now using trained model
        return predictions



    def evaluate_and_plot(self, y_test, pred_mean, pred_std, horizon, out_dir):
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
        plt.savefig(out_dir / f"predictions_{horizon}.png")
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
    start_time = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load training data
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    print(f"📁 Loading features from: {features_path}")
    data = pd.read_csv(features_path, index_col=0, parse_dates=True)


    data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)  # Ensure time zone
    data = data.sort_index()

    test_start = pd.Timestamp("2024-01-01")
    train_df = data[data.index < test_start]
    test_df = data[data.index >= test_start]

    

    model = XGBOOSTLSSCV()

    for horizon in [14, 24, 38]:
        print(f"\n⏱ Forecasting t+{horizon}h...")
        X_train, y_train = model.prepare_data(train_df, horizon)
        X_test, y_test = model.prepare_data(test_df, horizon)

        predictions = model.train_and_predict(X_train, y_train, X_test)
        pred_mean = predictions['loc'].values
        pred_std = predictions['scale'].values

        out_dir = Path(f"models_14_38/xgboost/plots/nocvlsspointforecast")
        model.evaluate_and_plot(y_test, pred_mean, pred_std, horizon, out_dir)
        
        # Print metrics
        print(f"\n📊 Evaluation for t+{horizon}h:")
        print(f"RMSE: {results['metrics']['RMSE']:.2f}")
        print(f"SMAPE: {results['metrics']['SMAPE']:.2f}%")
        print(f"R²: {results['metrics']['R2']:.4f}")


    print(f"\n✅ Completed LSS calibration in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()