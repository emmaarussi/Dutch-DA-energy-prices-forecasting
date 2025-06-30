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

Running ENBPI on RFECV forecasts...

Horizon t+14h — Coverage: 78.4%, Mean Width: 65.83

Horizon t+24h — Coverage: 78.8%, Mean Width: 83.98

Horizon t+38h — Coverage: 78.1%, Mean Width: 67.51

✅ Completed RFECV Linear + ENBPI in 42.55 seconds

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
            test_window = data[(data.index >= forecast_start) & (data.index < forecast_start + pd.Timedelta(step_size))]
            print(f"Forecast start: {forecast_start}")
            print(f"Test window: {test_window.index.min()} to {test_window.index.max()}")
            print(f"Train window: {train_window.index.min()} to {train_window.index.max()}")
            print("\n")

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



# --- ENBPI wrapper for 
class LinearOnlineENBPI:
    def __init__(self, horizons=[14, 24, 38], residual_window=100):
        self.horizons = horizons
        self.residual_window = residual_window
        self.ensemble_online_resid = []

    @staticmethod
    def strided_app(a, L, S):
        a = np.asarray(a)
        if a.ndim != 1 or a.size < L:
            return np.array([]).reshape(0, L)
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

    def compute_intervals_from_ar_predictions(self, point_forecasts_df, horizons, alpha=0.1):
        results = []
        for horizon in horizons:
            df = point_forecasts_df[point_forecasts_df['horizon'] == horizon].copy()
            df = df.sort_values('target_time')
            residuals = np.abs(df['actual'] - df['predicted'])

            self.ensemble_online_resid = residuals[:self.residual_window].tolist()
            lowers, uppers = [], []

            for i in range(len(df)):
                pred = df.iloc[i]['predicted']
                actual = df.iloc[i]['actual']

                resid_buffer = self.ensemble_online_resid[-self.residual_window:]
                resid_windows = self.strided_app(resid_buffer, L=10, S=1)

                if len(resid_windows) > 0:
                    q_per_window = np.percentile(resid_windows, (1 - alpha) * 100, axis=1)
                    q = np.mean(q_per_window)
                else:
                    q = np.percentile(resid_buffer, (1 - alpha) * 100)

                lower = pred - q
                upper = pred + q

                self.ensemble_online_resid.append(abs(actual - pred))
                if len(self.ensemble_online_resid) > self.residual_window:
                    self.ensemble_online_resid.pop(0)

                lowers.append(lower)
                uppers.append(upper)

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

    # --- Rolling window forecasts for ENBPI
    print("\nRunning rolling RFECV Linear + ENBPI...")

    linear_model = RFECVLinearLagsAndDummies(horizons=[14, 24, 38])
    rolling_preds = linear_model.run_rolling_forecast(
        data=pd.concat([train_df, test_df]),
        test_start = pd.to_datetime('2024-01-01', utc=True),
        window_size='365D',
        step_size='7D'
    )

    # Save predictions
    rolling_preds.to_csv('models_14_38/Linear_with_lags/RFECV_linear_with_lags/full_rolling_predictions.csv', index=False)

    # Use those predictions in ENBPI
    print("\nRunning ENBPI on rolling RFECV forecasts...")
    enbpi = LinearOnlineENBPI(residual_window=100)
    results = enbpi.compute_intervals_from_ar_predictions(
        point_forecasts_df=rolling_preds,
        horizons=[14, 24, 38],
        alpha=0.1
    )

    # --- Evaluation and Plotting
    out_dir = Path('models_14_38/Linear_with_lags/RFECV_linear_with_lags/plots_enbpi')
    out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in [14, 24, 38]:
        df_h = results[results['horizon'] == horizon]
        coverage = ((df_h['actual'] >= df_h['lower']) & (df_h['actual'] <= df_h['upper'])).mean() * 100
        width = (df_h['upper'] - df_h['lower']).mean()

        print(f"\nHorizon t+{horizon}h — Coverage: {coverage:.1f}%, Mean Width: {width:.2f}")

        plt.figure(figsize=(12, 6))
        plt.plot(df_h['target_time'], df_h['actual'], label="Actual", alpha=0.7)
        plt.plot(df_h['target_time'], df_h['predicted'], label="Predicted", alpha=0.7)
        plt.fill_between(df_h['target_time'], df_h['lower'], df_h['upper'], alpha=0.2, label="90% CI")
        plt.title(f"RFECV Linear + ENBPI — t+{horizon}h")
        plt.xlabel("Date")
        plt.ylabel("Price (EUR/MWh)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"enbpi_rfecv_linear_t{horizon}h.png")
        plt.close()

    print(f"\n✅ Completed RFECV Linear + ENBPI in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
    