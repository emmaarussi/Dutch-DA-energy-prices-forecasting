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
import time
from pathlib import Path
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
        
    def train_and_predict(self, train_data, test_data, horizon):
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


    
    def evaluate_and_plot(self, y_test, pred_mean, horizon, out_dir, fold_id=None):
        """Evaluate model and save forecast plot with prediction intervals."""
        
        residuals = y_test.values - pred_mean
        metrics = calculate_metrics(y_test.values, pred_mean)

        print(f"\n📊 Evaluation for t+{horizon}h:")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R²: {metrics['R2']:.4f}")
        print(pd.Series(residuals).describe())

        # Create predictions DataFrame
        plot_df = pd.DataFrame({
            'actual': y_test,
            'predicted': pred_mean,
        }, index=y_test.index)

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df.index, plot_df['actual'], label='Actual', alpha=0.7)
        plt.plot(plot_df.index, plot_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices - t+{horizon}h')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f'predictions_{horizon}_fold{fold_id}.png')
        plt.close()

        # Return metrics and residuals for drift analysis
        return {
            "RMSE": metrics["RMSE"],
            "SMAPE": metrics["SMAPE"],
            "R2": metrics["R2"],
            'residuals': residuals,
            'predictions': plot_df,
            'metrics': metrics,
        }


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
    horizons = [14, 24, 38]

    model = RFECVLinearLagsAndDummies(horizons=horizons)
    window_size = pd.Timedelta(days=365)
    step_size = pd.Timedelta(days=7)
    gap_size = pd.Timedelta(days=7)
    out_dir = Path('models_14_38/Linear_with_lags/RFECV_enbpi_retrain/plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    

    rolling_start = pd.to_datetime("2024-03-07 23:00:00", utc=True)  # start of testset
    rolling_end = test_df.index.max() - step_size
    current_forecast_time = rolling_start
    fold_id = 0

    all_preds = []       # Collect all predictions here
    all_metrics = []


    while current_forecast_time + step_size <= rolling_end:
        train_end = current_forecast_time - gap_size
        train_start = train_end - window_size
        test_start = current_forecast_time
        test_end = test_start + step_size

        print(f"\n📆 Fold {fold_id} — Train: {train_start} to {train_end}, Test: {test_start} to {test_end}")

        train_data = train_df[(train_df.index >= train_start) & (train_df.index < train_end)]
        test_data = test_df[(test_df.index >= test_start) & (test_df.index < test_end)]

        if train_data.empty or test_data.empty:
            print("⚠️ Skipping due to missing data in this window.")
            current_forecast_time += step_size
            fold_id += 1
            continue

        for h in horizons:
            results = model.train_and_predict(train_data, test_data, h)
            preds_df = pd.DataFrame(results['predictions'])
            preds_df['horizon'] = h
            preds_df['target_time'] = preds_df.index  # if not already present
            all_preds.append(preds_df)
            all_metrics.append({
                "fold": fold_id,
                "forecast_start": current_forecast_time,
                **results['test_metrics'] 
            })
        
        current_forecast_time += step_size
        fold_id += 1

    all_preds_df = pd.concat(all_preds).reset_index(drop=True)
    all_preds_df.to_csv('models_14_38/Linear_with_lags/RFECV_enbpi_retrain/full_rolling_predictions.csv', index=False)
        
    print(f"Full data range: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

    # --- Rolling window forecasts for ENBPI
    print("\nRunning rolling RFECV Linear + ENBPI...")

    

    
    # Use those predictions in ENBPI
    print("\nRunning ENBPI on rolling RFECV forecasts...")
    enbpi = LinearOnlineENBPI(residual_window=100)
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    results = enbpi.compute_intervals_from_ar_predictions(
        point_forecasts_df=all_preds_df,
        horizons=horizons,
        alpha=0.1
    )



    for horizon in horizons:
        df_h = results[results['horizon'] == horizon]
        metrics = calculate_metrics(df_h['actual'], df_h['predicted'])
        print(f"\nt+{h}h:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}")
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
    