
"""During initial experimentation, a function-based implementation of the
 XGBoostLSS model yielded unusually optimistic results. Upon further 
 inspection, it became evident that the same model instance was being 
 reused across multiple forecast horizons and validation windows. This 
 likely led to unintentional leakage of learned parameters, particularly
  from prior training iterations, resulting in artificially low errors and 
  overly confident prediction intervals. To address this,
   I adopted a class-based approach where a new XGBoostLSS model is
    instantiated for each training run. This ensures that each forecast
     is independently calibrated and evaluated, preserving the integrity 
     of the rolling window validation procedure. Although this approach 
     may yield slightly higher errors, it reflects a more honest estimate 
     of out-of-sample performance and avoids contamination from model state 
     reuse—especially critical in time series forecasting, where temporal
      dependencies can make leakage subtle but impactful. 


emmaarussi@MacBook-Air-M1-C02DQQP6Q6L4-R13-EA thesis-dutch-energy-analysis % /Users/emmaarussi/CascadeProjects/thesis-dut
ch-energy-analysis/venv_py310/bin/python /Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/models_14_38/XGBo
ostLSS/LSSwithClassnoCV.py
📁 Loading training data from: /Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/data/processed/multivariate_features_selectedXGboost.csv
📁 Loading test data from: /Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/data/processed/multivariate_features_testset_selectedXGboost.csv
🗓 Train range: 2023-01-08 00:00:00+01:00 to 2024-02-28 09:00:00+01:00
🗓 Test range: 2024-03-08 00:00:00+01:00 to 2024-04-29 09:00:00+02:00

⏱ Forecasting t+14h...
🔧 Optimizing hyperparameters...
Best trial: 0. Best value: 9238.57:   3%|█                              | 1/30 [00:04<02:13,  4.61s/it, 4.61/600 seconds]invalid value encountered in subtract
Best trial: 26. Best value: 8998.67: 100%|███████████████████████████| 30/30 [01:56<00:00,  3.89s/it, 116.59/600 seconds]

Hyper-Parameter Optimization successfully finished.
  Number of finished trials:  30
  Best trial:
    Value: 8998.6656252
    Params: 
    eta: 0.06702697037996934
    max_depth: 4
    gamma: 5.06230039397917e-07
    subsample: 0.5233070632162992
    colsample_bytree: 0.4712292347404474
    min_child_weight: 6.514620495837513e-06
    booster: gbtree
    opt_rounds: 100

📊 Evaluation for t+14h:
RMSE: 32.86
SMAPE: 45.70%
R²: 0.1847
90% Prediction Interval Coverage: 46.38%
Mean Interval Width: 31.05

Residual Stats:
count    1257.000000
mean      -13.359141
std        30.029899
min      -181.499013
25%       -26.001070
50%        -7.278556
75%         4.612552
max        89.648885
dtype: float64

⏱ Forecasting t+24h...
🔧 Optimizing hyperparameters...
Best trial: 0. Best value: 9312.41:   3%|█                              | 1/30 [00:04<02:18,  4.78s/it, 4.78/600 seconds]invalid value encountered in subtract
Best trial: 22. Best value: 8947.99: 100%|███████████████████████████| 30/30 [02:18<00:00,  4.63s/it, 138.90/600 seconds]

Hyper-Parameter Optimization successfully finished.
  Number of finished trials:  30
  Best trial:
    Value: 8947.985351399999
    Params: 
    eta: 0.06938345392704058
    max_depth: 5
    gamma: 2.2260369031530367e-05
    subsample: 0.8944117734941391
    colsample_bytree: 0.6156657499191931
    min_child_weight: 0.028283896827269908
    booster: gbtree
    opt_rounds: 99

📊 Evaluation for t+24h:
RMSE: 33.24
SMAPE: 45.94%
R²: 0.1641
90% Prediction Interval Coverage: 47.97%
Mean Interval Width: 30.66

Residual Stats:
count    1257.000000
mean      -13.520982
std        30.381502
min      -182.653153
25%       -27.670684
50%        -7.406902
75%         4.449059
max        93.662060
dtype: float64

⏱ Forecasting t+38h...
🔧 Optimizing hyperparameters...
Best trial: 0. Best value: 9426.22:   3%|██▏                                                               | 1/30 [00:04<02:11,  4.52s/it, 4.52/600 seconds]invalid value encountered in subtract
Best trial: 14. Best value: 9291.21: 100%|██████████████████████████████████████████████████████████████| 30/30 [01:43<00:00,  3.47s/it, 103.97/600 seconds]

Hyper-Parameter Optimization successfully finished.
  Number of finished trials:  30
  Best trial:
    Value: 9291.2085938
    Params: 
    eta: 0.21187774825110306
    max_depth: 2
    gamma: 3.0309494748047132e-06
    subsample: 0.37942196252470767
    colsample_bytree: 0.29884797196512025
    min_child_weight: 5.758720313577769e-05
    booster: gbtree
    opt_rounds: 99

📊 Evaluation for t+38h:
RMSE: 32.93
SMAPE: 46.34%
R²: 0.1748
90% Prediction Interval Coverage: 49.96%
Mean Interval Width: 33.89

Residual Stats:
count    1257.000000
mean      -12.911314
std        30.306005
min      -184.012910
25%       -27.204563
50%        -7.706218
75%         6.984030
max        85.217286
dtype: float64

✅ Completed LSS calibration in 364.60 seconds


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
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    print(f"📁 Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)

    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')
    print(f"📁 Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

    print(f"🗓 Train range: {train_df.index.min()} to {train_df.index.max()}")
    print(f"🗓 Test range: {test_df.index.min()} to {test_df.index.max()}")

    model = XGBOOSTLSSCV()

    for horizon in [14, 24, 38]:
        print(f"\n⏱ Forecasting t+{horizon}h...")
        X_train, y_train = model.prepare_data(train_df, horizon)
        X_test, y_test = model.prepare_data(test_df, horizon)

        predictions = model.train_and_predict(X_train, y_train, X_test)
        pred_mean = predictions['loc'].values
        pred_std = predictions['scale'].values

        out_dir = Path(f"models_14_38/xgboost/plots/nocvlss")
        model.evaluate_and_plot(y_test, pred_mean, pred_std, horizon, out_dir)


    print(f"\n✅ Completed LSS calibration in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()