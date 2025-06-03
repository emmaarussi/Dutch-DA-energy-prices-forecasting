
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
    def __init__(self):
        pass
    
    def get_xgboostlss_model(self):
        """Initialize XGBoostLSS model with Gaussian distribution."""
        return XGBoostLSS(Gaussian())
    
    def prepare_data(self, df, horizon):
        """Split features and target for a specific forecast horizon."""
        target_col = f'target_t{horizon}'  
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y

    def train_and_predict(self, X_train, y_train, X_test):
        """Optimize hyperparameters, train model, and make predictions."""
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
        opt_param = self.hyper_opt(
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
        self.train(params=opt_param, dtrain=dtrain, num_boost_round=n_rounds, verbose_eval=10)
        predictions = self.predict(dtest)

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

        out_dir = Path(f"models_14_38/xgboost/plots/lss")
        evaluate_and_plot(y_test, pred_mean, pred_std, horizon, out_dir)

    print(f"\n✅ Completed LSS calibration in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()