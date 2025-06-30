"""
Simple linear model combining:
1. Significant price lags, here we are looking at lags of the price that are either the closes one available form the horizon distance, or lags that are exactly 24, 48 or 168 away from the predicted horizon.

2. Calendar features (from dummy model)
does not retrain each day for point forecast
Horizon t+14h — Coverage: 54.4%, Mean Width: 77.65

Horizon t+24h — Coverage: 72.0%, Mean Width: 96.88

Horizon t+38h — Coverage: 62.7%, Mean Width: 82.27

✅ Completed RFECV Linear + Online Bootstrap in 61.78 seconds

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from sklearn.feature_selection import RFECV




class RFECVLinearLagsAndDummies:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        self.models = {}  # One model per horizon
        self.scalers = {}  # One scaler per horizon
        self.feature_importance = {}
        
    def __create_lag_features(self, df):
        """Use price lag features from the data."""
        lag_features = df.filter(like='price_eur_per_mwh_lag')
        return lag_features

    def create_time_features(self, df):
        """Use existing time features from the data."""
        # Essential time features and calendar effects
        time_features = [
            'hour_sin', 'hour_cos',      # Hour of day
            'day_of_week_sin', 'day_of_week_cos',  # Day of week
            'month_sin', 'month_cos',    # Month of year
            'is_weekend', 'is_holiday',  # Calendar effects
        ]
        
        features = df[time_features].copy()
        return features

    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon."""
        # Get time features
        time_features = self.create_time_features(data)
        
        # Get significant price lags for this horizon
        lag_features = self.__create_lag_features(data)
        
        # Combine features
        X = pd.concat([time_features, lag_features], axis=1)
        
        # Target variable
        y = data[f'target_t{horizon}']
        
        return X, y
        
    def train_and_evaluate(self, train_data, test_data, horizon):
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
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': pd.DataFrame({'actual': y_test, 'predicted': y_pred_test}, index=X_test.index)
        }


    def predict(self, test_data):
        """Make predictions for all horizons."""
        predictions = pd.DataFrame(index=test_data.index)
        for h in self.horizons:
            X_test, _ = self.prepare_data(test_data, h)
            model_info = self.models[h]
            predictions[f'pred_t{h}'] = model_info['model'].predict(X_test)
        return predictions


    def run_rolling_forecast(self, data, test_start, window_size='365D', step_size='7D'):
        """Run rolling window forecasts and return a full prediction DataFrame."""
        results = []
        horizons = self.horizons
        data = data.sort_index()

        for forecast_start in pd.date_range(start=test_start, end=data.index.max(), freq=step_size):
            forecast_start = pd.Timestamp(forecast_start).replace(hour=12, tzinfo=data.index.tz)

            if forecast_start not in data.index:
                continue

            window_start = forecast_start - pd.Timedelta(window_size)
            train_window = data[(data.index >= window_start) & (data.index < forecast_start)]
            test_window = data[(data.index >= forecast_start)]

            for h in horizons:
                result = self.train_and_evaluate(train_window, test_window, h)
                pred_df = result['predictions'].copy()
                pred_df['horizon'] = h
                pred_df['forecast_start'] = forecast_start
                pred_df = pred_df.reset_index().rename(columns={'index': 'target_time'})
                
                for _, row in pred_df.iterrows():
                    results.append({
                        'window_size': window_size,
                        'forecast_start': forecast_start,
                        'target_time': row['target_time'],
                        'horizon': row['horizon'],
                        'predicted': row['predicted'],
                        'actual': row['actual']
                    })

        return pd.DataFrame(results)



class OnlineLinearBootstrap:
    def __init__(self, horizons=[14, 24, 38], residual_window=100):
        self.horizons = horizons
        self.residual_window = residual_window
        self.B = 100
        self.alpha = 0.1

    def fit_ar_model(self, residuals):
        max_lags = min(20, len(residuals) // 2)  # prevent overfitting on short buffers
        if max_lags < 1:
            raise ValueError("Too few residuals to fit AR model.")
        
        initial_model = AutoReg(residuals, lags=max_lags, old_names=False, period=24)
        initial_fit = initial_model.fit()
        n = len(residuals)
        bic_threshold = np.log(n)
        selected_lags = []

        for i, (coef, std_err) in enumerate(zip(initial_fit.params[1:], initial_fit.bse[1:])):
            t_stat = coef / std_err
            bic_contribution = t_stat**2 - bic_threshold
            if bic_contribution > 0:
                selected_lags.append(i + 1)
        if not selected_lags:
            selected_lags = [1]

        model = AutoReg(residuals, lags=selected_lags, old_names=False)
        fit = model.fit()
        return selected_lags, fit.params[1:], fit.resid

    def generate_sieve_bootstrap_paths(self, residuals, selected_lags, phi, innovations, h, B):
        max_lag = max(selected_lags)
        series_init = list(residuals[-max_lag:])
        samples = []

        for _ in range(B):
            resampled_innov = np.random.choice(innovations, size=h, replace=True)
            series = series_init.copy()
            path = []

            for t in range(h):
                prev = np.array([series[-lag] for lag in selected_lags])
                val = np.dot(phi, prev) + resampled_innov[t]
                series.append(val)
                path.append(val)

            samples.append(path)

        return np.array(samples)

    def compute_intervals_from_ar_predictions(self, point_forecasts_df, horizons, alpha=None, prefill_buffer=None):
        alpha = self.alpha if alpha is None else alpha
        results = []

        for horizon in horizons:
            df = point_forecasts_df[point_forecasts_df['horizon'] == horizon].copy()
            df = df.sort_values('target_time')
            residuals = df['actual'] - df['predicted']

            buffer = (prefill_buffer or []) + residuals.tolist()
            lowers, uppers = [], []

            for i in range(len(df)):
                pred = df.iloc[i]['predicted']

                resid_buffer = buffer[i : i + self.residual_window]
                if len(resid_buffer) < 20:
                    print(f"[t+{horizon}h | {i}] Skipping — buffer too small ({len(resid_buffer)})")
                    continue

                try:
                    selected_lags, phi, innovations = self.fit_ar_model(pd.Series(resid_buffer))
                    samples = self.generate_sieve_bootstrap_paths(
                        residuals=resid_buffer,
                        selected_lags=selected_lags,
                        phi=np.array(phi),
                        innovations=innovations,
                        h=int(horizon),
                        B=self.B
                    )
                    residual_forecasts = samples[:, horizon - 1]
                    lower = pred + np.percentile(residual_forecasts, 100 * alpha / 2)
                    upper = pred + np.percentile(residual_forecasts, 100 * (1 - alpha / 2))
                except Exception as e:
                    print(f"[t+{horizon}h | {i}] AR fitting failed: {e}")
                    continue

                lowers.append(lower)
                uppers.append(upper)

            df = df.iloc[len(df) - len(lowers):].copy()
            df['lower'] = lowers
            df['upper'] = uppers
            results.append(df)

        return pd.concat(results)


def main():
    import time
    from pathlib import Path

    start_time = time.time()

    # --- Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_testset_selectedXGboost.csv')

    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path, index_col=0)
    train_df.index = pd.to_datetime(train_df.index, utc=True)
    print(f"Training data loaded: {train_df.index.min()} to {train_df.index.max()}")

    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path, index_col=0)
    test_df.index = pd.to_datetime(test_df.index, utc=True)
    print(f"Test data loaded: {test_df.index.min()} to {test_df.index.max()}")

    # --- Train RFECV Linear model
    print("\nRunning RFECV Linear on static split...")

    linear_model = RFECVLinearLagsAndDummies(horizons=[14, 24, 38])
    rfecv_preds_list = []

    for h in [14, 24, 38]:
        result = linear_model.train_and_evaluate(train_df, test_df, horizon=h)
        pred_df = result["predictions"].copy()
        pred_df["horizon"] = h
        pred_df = pred_df.reset_index().rename(columns={"index": "target_time"})
        rfecv_preds_list.append(pred_df)

    rfecv_preds = pd.concat(rfecv_preds_list, ignore_index=True)

    # --- Run Online Linear Bootstrap on RFECV predictions
    print("\nRunning Online Linear Bootstrap on RFECV forecasts...")
    model = OnlineLinearBootstrap(residual_window=100)
    results = model.compute_intervals_from_ar_predictions(
        point_forecasts_df=rfecv_preds,
        horizons=[14, 24, 38],
        alpha=0.1
    )

    # --- Evaluation and Plotting
    out_dir = Path('models_14_38/Linear_with_lags/RFECV_linear_with_lags/plots_SBonline')
    out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in [14, 24, 38]:
        df_h = results[results['horizon'] == horizon]
        coverage = ((df_h['actual'] >= df_h['lower']) & (df_h['actual'] <= df_h['upper'])).mean() * 100
        width = (df_h['upper'] - df_h['lower']).mean()

        print(f"\nHorizon t+{horizon}h — Coverage: {coverage:.1f}%, Mean Width: {width:.2f}")

        metrics = calculate_metrics(df_h['actual'], df_h['predicted'])
        print(metrics)
        # Residual Drift Plot
        df_h['residual'] = df_h['actual'] - df_h['predicted']

        plt.figure(figsize=(12, 4))
        plt.plot(df_h['target_time'], df_h['residual'], label="Residuals", alpha=0.7)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f"Residual Drift — RFECV Linear (t+{horizon}h)")
        plt.xlabel("Date")
        plt.ylabel("Residual (EUR/MWh)")
        plt.tight_layout()
        plt.savefig(out_dir / f"residual_drift_rfecv_linear_t{horizon}h.png")
        plt.close()


        plt.figure(figsize=(12, 6))
        plt.plot(df_h['target_time'], df_h['actual'], label="Actual", alpha=0.7)
        plt.plot(df_h['target_time'], df_h['predicted'], label="Predicted", alpha=0.7)
        plt.fill_between(df_h['target_time'], df_h['lower'], df_h['upper'], alpha=0.2, label="90% CI")
        plt.title(f"RFECV Linear + Online Bootstrap — t+{horizon}h")
        plt.xlabel("Date")
        plt.ylabel("Price (EUR/MWh)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"online_bootstrap_rfecv_linear_t{horizon}h.png")
        plt.close()

    print(f"\n✅ Completed RFECV Linear + Online Bootstrap in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
    