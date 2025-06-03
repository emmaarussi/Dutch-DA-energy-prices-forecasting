"""
Processing SPCI for t+14h...
t+14h | Coverage: 43.7% | Width: 22.51

Processing SPCI for t+24h...
t+24h | Coverage: 43.0% | Width: 22.81

Processing SPCI for t+38h...
t+38h | Coverage: 52.3% | Width: 35.32

"""

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models_14_38.xgboost.OptimizedXGboost import XGBoostOptimized
from xgboost import XGBRegressor


class XGBoostSPCI(XGBoostOptimized):
    def __init__(self, horizons=[14, 24, 38], alpha=0.1, window_size=100):
        super().__init__(horizons)
        self.alpha = alpha
        self.lower_q = alpha / 2
        self.upper_q = 1 - (alpha / 2)
        self.window_size = window_size

    def fit_quantile_model(self, X, y, quantile, horizon):
        params = self.get_hyperparameters(horizon)
        model = XGBRegressor(
            **params,
            objective=lambda y_true, y_pred: self.quantile_loss(y_true, y_pred, quantile),
        )
        model.fit(X, y)
        return model

    def get_point_model(self, horizon):
        params = self.get_hyperparameters(horizon)
        return XGBRegressor(**params)

    @staticmethod
    def quantile_loss(y_true, y_pred, q):
        e = y_true - y_pred
        grad = np.where(e < 0, -q, 1 - q)
        hess = np.ones_like(y_true)  # constant hessian
        return grad, hess

    def compute_spci_intervals(self, train_df, test_df, horizon):
        X_train, y_train = self.prepare_data(train_df, horizon)
        point_model = self.get_point_model(horizon)
        point_model.fit(X_train, y_train)

        X_test, y_test = self.prepare_data(test_df, horizon)
        point_preds = point_model.predict(X_test)
        y_train_preds = point_model.predict(X_train)

        residuals = y_train.values - y_train_preds
        residuals = pd.Series(residuals, index=train_df.index[-len(residuals):])
        residuals.name = f"target_t{horizon}"

        T = len(residuals)
        w = self.window_size
        if T <= w:
            raise ValueError("Not enough residuals for conditional modeling.")

        X_resid, y_resid = [], []
        for t in range(T - w):
            X_resid.append(residuals.iloc[t:t+w].values)
            y_resid.append(residuals.iloc[t+w])
        X_resid = pd.DataFrame(X_resid)
        y_resid = pd.Series(y_resid, name=f"target_t{horizon}")

        X_test_resid = pd.DataFrame(residuals.iloc[-w:].values.reshape(1, -1), columns=X_resid.columns)

        # Fit 3 quantile models
        q05_model = self.fit_quantile_model(X_resid, y_resid, self.lower_q, horizon)
        q50_model = self.fit_quantile_model(X_resid, y_resid, 0.5, horizon)
        q95_model = self.fit_quantile_model(X_resid, y_resid, self.upper_q, horizon)

        # Predict conditional residuals
        q05 = q05_model.predict(X_test_resid)[0]
        q50 = q50_model.predict(X_test_resid)[0]
        q95 = q95_model.predict(X_test_resid)[0]

        lower = point_preds + (q05 - q50)
        upper = point_preds + (q95 - q50)

        return pd.DataFrame({
            'actual': y_test.values,
            'point_pred': point_preds,
            'lower': lower,
            'upper': upper
        }, index=test_df.index)


    def evaluate_and_plot(self, df, horizon):
        coverage = np.mean((df['actual'] >= df['lower']) & (df['actual'] <= df['upper'])) * 100
        width = np.mean(df['upper'] - df['lower'])

        print(f"t+{horizon}h | Coverage: {coverage:.1f}% | Width: {width:.2f}")

        out_dir = f'models_14_38/xgboost/plots/spci'
        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(15, 6))
        plt.fill_between(df.index, df['lower'], df['upper'], alpha=0.3, label='90% PI')
        plt.plot(df.index, df['point_pred'], linestyle='--', label='Predicted')
        plt.plot(df.index, df['actual'], color='black', label='Actual')
        plt.title(f'SPCI Forecast (t+{horizon}h)')
        plt.xlabel('Date'); plt.ylabel('Price (EUR/MWh)')
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(f'{out_dir}/spci_forecast_h{horizon}.png')
        plt.close()

        return coverage, width
     


def main():
    start_time = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')

    train_df = pd.read_csv(train_path, index_col=0)
    train_df.index = pd.to_datetime(train_df.index)

    test_df = pd.read_csv(test_path, index_col=0)
    test_df.index = pd.to_datetime(test_df.index)

    model = XGBoostSPCI(horizons=[14, 24, 38], alpha=0.1, window_size=100)

    for h in model.horizons:
        print(f"\nProcessing SPCI for t+{h}h...")
        result_df = model.compute_spci_intervals(train_df, test_df, h)
        if not result_df.empty:
            model.evaluate_and_plot(result_df, h)

    print(f"\n✅ Completed SPCI calibration in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()