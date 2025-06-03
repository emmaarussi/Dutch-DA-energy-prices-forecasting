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

Metrics for 365D window:
  Horizon t+14h:
    MAE: 13.11
    RMSE: 16.93
    SMAPE: 23.14
    WMAPE: 18.99
    R2: 0.40
  Horizon t+24h:
    MAE: 14.13
    RMSE: 18.06
    SMAPE: 24.16
    WMAPE: 20.54
    R2: 0.31
  Horizon t+38h:
    MAE: 15.70
    RMSE: 20.03
    SMAPE: 26.00
    WMAPE: 22.83
    R2: 0.13

Why do this you might ask, well, perhaps it simply works, and some literature from 2005 said its worth trying
so here we are. We are not even AICing this, its just based on a hunch.
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
            'is_morning', 'is_evening'   # Time of day effects
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
        
        # Store feature importance (based on coefficient magnitude)
        importance_df = pd.DataFrame({
            'feature': list(selected_features),
            'importance': abs(model.coef_)
        }).sort_values('importance', ascending=False)
        self.feature_importance[horizon] = importance_df

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': pd.DataFrame({'actual': y_test, 'predicted': y_pred_test}, index=X_test.index)
        }

    def plot_feature_importance(self, horizon, top_n=20, filename=None):
        """Plot feature importance for a specific horizon."""
        importance_df = self.feature_importance.get(horizon)
        if importance_df is None or importance_df.empty:
            print(f"No importance data for horizon t+{horizon}.")
            return

        plt.figure(figsize=(12, 8))
        importance_df = importance_df.nlargest(top_n, 'importance')
        plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
        plt.title(f'Feature Importance (t+{horizon}h)')
        plt.xlabel('Absolute Coefficient Value')
        plt.ylabel('Feature')
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def predict(self, test_data):
        """Make predictions for all horizons."""
        predictions = pd.DataFrame(index=test_data.index)
        for h in self.horizons:
            X_test, _ = self.prepare_data(test_data, h)
            model_info = self.models[h]
            predictions[f'pred_t{h}'] = model_info['model'].predict(X_test)
        return predictions


def main():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    test_start = '2024-01-01'
    train_data = data[data.index < test_start]
    test_data = data[data.index >= test_start]

    print(f"Full data range: {data.index.min()} to {data.index.max()}")
    print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

    horizons = [14, 24, 38]
    window_sizes = ['365D']

    for window_size in window_sizes:
        print(f"\nEvaluating with {window_size} window:")
        window_predictions = []

        for day in pd.date_range(test_data.index.min(), test_data.index.max(), freq='7D'):
            forecast_start = pd.Timestamp(day.date()).replace(hour=12, tzinfo=test_data.index.tzinfo)

            if forecast_start in test_data.index:
                window_end = forecast_start
                window_start = forecast_start - pd.Timedelta(window_size)
                train_window = data[(data.index >= window_start) & (data.index < window_end)]
                test_window = data[(data.index >= forecast_start)]

                model = RFECVLinearLagsAndDummies(horizons=horizons)

                for h in horizons:
                    result = model.train_and_evaluate(train_window, test_window, h)
                    pred_df = result['predictions'].copy()
                    pred_df['horizon'] = h
                    pred_df['forecast_start'] = forecast_start
                    pred_df = pred_df.reset_index().rename(columns={'index': 'target_time'})
                    for _, row in pred_df.iterrows():
                        window_predictions.append({
                            'window_size': window_size,
                            'forecast_start': forecast_start,
                            'target_time': row['target_time'],
                            'horizon': row['horizon'],
                            'predicted': row['predicted'],
                            'actual': row['actual']
                        })

        results_df = pd.DataFrame(window_predictions)
        print(f"\nMetrics for {window_size} window:")
        for horizon in results_df['horizon'].unique():
            subset = results_df[results_df['horizon'] == horizon]
            metrics = calculate_metrics(subset['actual'], subset['predicted'])
            print(f"  Horizon t+{int(horizon)}h:")
            for key, val in metrics.items():
                print(f"    {key}: {val:.2f}")

        # Plotting
        for h in horizons:
            h_df = results_df[results_df['horizon'] == h]
            plt.figure(figsize=(15, 6))
            plt.plot(h_df['target_time'], h_df['actual'], label='Actual', alpha=0.7)
            plt.plot(h_df['target_time'], h_df['predicted'], label='Predicted', alpha=0.7)
            plt.title(f'Actual vs Predicted Prices Over Time (Linear RFECV, {window_size} window, t+{h}h)')
            plt.xlabel('Date')
            plt.ylabel('Price (EUR/MWh)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            os.makedirs('models_14_38/Linear_with_lags/RFECV_linear_with_lags/plots', exist_ok=True)
            plt.savefig(f'models_14_38/Linear_with_lags/RFECV_linear_with_lags/plots/predictions_over_time_{window_size}_{h}h.png', dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    main()
    