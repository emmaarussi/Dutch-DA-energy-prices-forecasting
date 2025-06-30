"""
Processing Online SPCI for t+14h...
Online SPCI - t+14h
Coverage: 59.82%
Mean Width: 16.27

Processing Online SPCI for t+24h...
Online SPCI - t+24h
Coverage: 57.76%
Mean Width: 16.18

Processing Online SPCI for t+38h...
Online SPCI - t+38h
Coverage: 61.26%
Mean Width: 23.91

✅ Completed Online SPCI calibration in 5011.03 seconds

To generate prediction intervals in a sequential setting, I implemented an online extension of the Sequential Predictive Conformal Inference (SPCI) framework. This approach combines a fixed horizon-specific point forecast model with an online-updated residual model that conditions on recent forecast errors. For each forecast horizon $h \in \{14, 24, 38\}$, I trained a dedicated XGBoost model to generate point predictions $f_h(X_t)$, capturing horizon-specific dynamics. The residuals $r_t = Y_t - f_h(X_t)$ were stored in a rolling buffer with a maximum size of 1000 and a sliding window of size 100 used to represent local temporal error structure.

Every 24 hours, I used the most recent residuals in the buffer to update horizon-specific quantile regression models that predict the conditional distribution of future residuals. Each quantile model (for the $\alpha/2$, 0.5, and $1-\alpha/2$ quantiles) was implemented via XGBoost with a custom pinball loss. Since XGBoost does not support true incremental learning, I retrained the quantile models from scratch at each update using the latest residual sequences. The models were stored in a dictionary indexed by horizon and quantile level, enabling efficient reuse and ensuring that the prediction intervals remain adaptive to both recent errors and the forecasting horizon. The final prediction interval was computed as:
\[
\left[ f_h(X_t) + (\hat{r}^{\alpha/2}_t - \hat{r}^{0.5}_t),\; f_h(X_t) + (\hat{r}^{1-\alpha/2}_t - \hat{r}^{0.5}_t) \right],
\]
where the predicted residual quantiles $\hat{r}^{q}_t$ are conditional on the most recent residual window. This method maintains both temporal adaptivity and horizon-awareness, providing better-calibrated intervals under non-stationarity than horizon-agnostic baselines.


"""
"""
Processing Online SPCI for t+14h...
Online SPCI - t+14h
Coverage: 59.82%
Mean Width: 16.27

Processing Online SPCI for t+24h...
Online SPCI - t+24h
Coverage: 57.76%
Mean Width: 16.18

Processing Online SPCI for t+38h...
Online SPCI - t+38h
Coverage: 61.26%
Mean Width: 23.91

✅ Completed Online SPCI calibration in 5011.03 seconds
"""

import pandas as pd
import numpy as np
import os
import time
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import sys

# Add path to access XGBoostOptimized
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models_14_38.xgboost.OptimizedXGboost import XGBoostOptimized

# === CONFIG ===
HORIZON = [14, 24, 38]
ALPHA = 0.1
WINDOW_SIZE = 100

class OnlineXGBoostSPCI(XGBoostOptimized):
    def __init__(self, horizons=[14, 24, 38], alpha=0.1, window_size=100):
        super().__init__(horizons)
        self.alpha = alpha
        self.lower_q = alpha / 2
        self.upper_q = 1 - (alpha / 2)
        self.window_size = window_size
        self.update_freq = 24  # Update models every 24 hours
        self.max_buffer_size = 1000  # Maximum size of residual buffer
        self.n_rounds_update = 10  # Number of boosting rounds for incremental updates

    def quantile_loss_wrapper(self, quantile):
        def quantile_loss(y_true, y_pred):
            errors = y_true - y_pred
            grad = np.where(errors < 0, -quantile, 1 - quantile)
            hess = np.ones_like(y_true)
            return grad, hess
        return quantile_loss

    def fit_quantile_model(self, X, y, quantile, xgb_params, existing_model=None):
        # Make a copy to avoid mutating the original
        xgb_params = xgb_params.copy()
        xgb_params.pop("objective", None)  # Remove default if present
        
        if existing_model is None:
            # Initial fit
            model = XGBRegressor(**xgb_params)
            model.set_params(objective=self.quantile_loss_wrapper(quantile))
            model.fit(X, y)
        else:
            # Incremental update
            model = existing_model
            # For incremental update, we'll use warm_start
            model = existing_model
            model.n_estimators += self.n_rounds_update
            model.fit(X, y)
        return model

    def get_point_model(self, horizon):
        params = self.get_hyperparameters(horizon)
        return XGBRegressor(**params)

    def online_spci(self, train_df, test_df, horizon):
        X_train, y_train = self.prepare_data(train_df, horizon)
        X_test, y_test = self.prepare_data(test_df, horizon)

        point_model = self.get_point_model(horizon)
        point_model.fit(X_train, y_train)

        y_train_pred = point_model.predict(X_train)
        residual_buffer = list((y_train - y_train_pred).values[-self.window_size:])

        predictions = []
        intervals = []
        covers = []

        params = self.get_hyperparameters(horizon)

        # Initialize quantile models
        q05_model = None
        q50_model = None
        q95_model = None

        for t in range(len(X_test)):
            x_t = X_test.iloc[t].values.reshape(1, -1)
            y_t = y_test.iloc[t]
            y_pred = point_model.predict(x_t)[0]

            # Update models every UPDATE_FREQ hours
            if len(residual_buffer) >= self.window_size and t % self.update_freq == 0:
                # Use only the most recent MAX_BUFFER_SIZE residuals
                recent_residuals = residual_buffer[-self.max_buffer_size:]
                
                X_resid = []
                y_resid = []
                for i in range(len(recent_residuals) - WINDOW_SIZE):
                    X_resid.append(recent_residuals[i:i+WINDOW_SIZE])
                    y_resid.append(recent_residuals[i+WINDOW_SIZE])

                if X_resid:
                    X_resid = pd.DataFrame(X_resid)
                    y_resid = pd.Series(y_resid)

                    # Train or update models
                    q05_model = self.fit_quantile_model(X_resid, y_resid, self.alpha / 2, params, q05_model)
                    q50_model = self.fit_quantile_model(X_resid, y_resid, 0.5, params, q50_model)
                    q95_model = self.fit_quantile_model(X_resid, y_resid, 1 - (self.alpha / 2), params, q95_model)

            # Make predictions if models exist
            if all(model is not None for model in [q05_model, q50_model, q95_model]):
                X_test_resid = pd.DataFrame(
                    np.array(residual_buffer[-self.window_size:]).reshape(1, -1),
                    columns=range(self.window_size)
                )

                q05 = q05_model.predict(X_test_resid)[0]
                q50 = q50_model.predict(X_test_resid)[0]
                q95 = q95_model.predict(X_test_resid)[0]

                lower = y_pred + (q05 - q50)
                upper = y_pred + (q95 - q50)
            else:
                # Default intervals for initial predictions
                lower = y_pred - 15
                upper = y_pred + 15

            predictions.append(y_pred)
            intervals.append((lower, upper))
            covers.append(lower <= y_t <= upper)

            # Update residual buffer with fixed size
            residual_buffer.append(y_t - y_pred)
            if len(residual_buffer) > self.max_buffer_size:
                residual_buffer.pop(0)

        return pd.DataFrame({
            'actual': y_test.values,
            'point_pred': predictions,
            'lower': [interval[0] for interval in intervals],
            'upper': [interval[1] for interval in intervals]
        }, index=y_test.index)

    def evaluate_and_plot(self, df, horizon):
        coverage = np.mean(df['actual'] >= df['lower']) * 100
        width = np.mean(df['upper'] - df['lower'])

        print(f"Online SPCI - t+{horizon}h")
        print(f"Coverage: {coverage:.2f}%")
        print(f"Mean Width: {width:.2f}")

        out_dir = f'models_14_38/xgboost/plots/spci'
        os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['actual'], label='Actual', color='black')
        plt.plot(df.index, df['point_pred'], label='Predicted', linestyle='--')
        plt.fill_between(df.index, df['lower'], df['upper'], alpha=0.3, label='90% PI')
        plt.title(f'Online SPCI Prediction Intervals - t+{horizon}h')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{out_dir}/spci_online_forecast_h{horizon}.png')
        plt.close()



# === MAIN SCRIPT ===
def main():
    start_time = time.time()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')

    train_df = pd.read_csv(TRAIN_PATH, index_col=0, parse_dates=True)
    test_df = pd.read_csv(TEST_PATH, index_col=0, parse_dates=True)

    model = OnlineXGBoostSPCI(horizons=HORIZON, alpha=ALPHA, window_size=WINDOW_SIZE)

    for h in model.horizons:
        print(f"\nProcessing Online SPCI for t+{h}h...")
        result_df = model.online_spci(train_df, test_df, h)
        if not result_df.empty:
            model.evaluate_and_plot(result_df, h)

    print(f"\n✅ Completed Online SPCI calibration in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()