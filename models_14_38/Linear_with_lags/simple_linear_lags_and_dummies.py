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

Why do this you might ask, well, perhaps it simply works, and some literature from 2005 said its worth trying
so here we are. We are not even AICing this, its just based on a hunch.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.linear_model import LinearRegression
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

class SimpleLinearLagsAndDummies:

    def __init__(self, horizons=None):
        if horizons is None:
            self.horizons = range(14, 39)  # Default range
        else:
            self.horizons = horizons  # Custom range
        self.models = {}  # One model per horizon
        self.scalers = {}  # One scaler per horizon
        self.feature_importance = {}
        self.significant_lags = {
            14: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_10h', 'price_eur_per_mwh_lag_154h'],
            15: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_9h', 'price_eur_per_mwh_lag_153h'],
            16: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_8h', 'price_eur_per_mwh_lag_152h'],
            17: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_7h', 'price_eur_per_mwh_lag_151h'],
            18: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_6h', 'price_eur_per_mwh_lag_150h'],
            19: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_5h', 'price_eur_per_mwh_lag_149h'],
            20: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_4h', 'price_eur_per_mwh_lag_148h'],
            21: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_147h'],
            22: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_146h'],
            23: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_145h'],
            24: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_144h'],
            25: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_23h', 'price_eur_per_mwh_lag_143h'],
            26: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_22h', 'price_eur_per_mwh_lag_142h'],
            27: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_21h', 'price_eur_per_mwh_lag_141h'],
            28: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_20h', 'price_eur_per_mwh_lag_140h'],
            29: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_19h', 'price_eur_per_mwh_lag_139h'],
            30: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_18h', 'price_eur_per_mwh_lag_138h'],
            31: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_17h', 'price_eur_per_mwh_lag_137h'],
            32: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_16h', 'price_eur_per_mwh_lag_136h'],
            33: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_15h', 'price_eur_per_mwh_lag_135h'],
            34: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_14h', 'price_eur_per_mwh_lag_134h'],
            35: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_13h', 'price_eur_per_mwh_lag_133h'],
            36: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_12h', 'price_eur_per_mwh_lag_132h'],
            37: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_11h', 'price_eur_per_mwh_lag_131h'],
            38: ['price_eur_per_mwh_lag_1h', 'price_eur_per_mwh_lag_2h', 'price_eur_per_mwh_lag_3h', 'price_eur_per_mwh_lag_10h', 'price_eur_per_mwh_lag_130h'],

        }


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
        lag_features = data[self.significant_lags[horizon]].copy()
        
        # Combine features
        X = pd.concat([time_features, lag_features], axis=1)
        
        # Target variable
        y = data[f'target_t{horizon}']
        
        return X, y
        
    def train_and_evaluate(self, train_data, test_data, horizon):
        # Prepare features and target for training data
        X_train, y_train = self.prepare_data(train_data, horizon)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Store the model
        self.models[horizon] = {'model': model, 'features': X_train.columns}

        # Make predictions on training data
        y_pred_train = model.predict(X_train)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        
        # Prepare features and target for test data
        X_test, y_test = self.prepare_data(test_data, horizon)
        
        # Make predictions on test data
        y_pred_test = model.predict(X_test)
        
        # Calculate test metrics
        test_metrics = calculate_metrics(y_test, y_pred_test)
        
        # Store feature importance
        feature_names = list(X_train.columns)
        importance_df = pd.DataFrame({
            'feature': feature_names,
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
    print("Loading data...")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True).tz_convert('Europe/Amsterdam')

    model = SimpleLinearLagsAndDummies(horizons=[14, 24, 38])

    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')

    train_df = data[train_start:train_end]
    test_df = data[test_start:test_end]
    
    # Train and evaluate
    results = {}
    for horizon in model.horizons:
        print(f"\nTraining and evaluating horizon t+{horizon}h...")
        results[horizon] = model.train_and_evaluate(train_df, test_df, horizon)
    
    # Plot predictions
    for horizon in model.horizons:
        result = results[horizon]
        predictions_df = result['predictions']
        metrics = result['test_metrics']
        print(f"\nt+{horizon}h horizon:")
        print(f"Number of predictions: {len(predictions_df)}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R2: {metrics['R2']:.4f}")

    # Create output directory
    out_dir = 'models_14_38/Linear_with_lags/plots'
    os.makedirs(out_dir, exist_ok=True)

    # Plot feature importance for each horizon
    for horizon in model.horizons:
        model.plot_feature_importance(
            horizon,
            top_n=20,
            filename=f'{out_dir}/feature_importance_h{horizon}.png'
        )
        
    # Plot predictions over time
    for horizon in model.horizons:
        predictions_df = results[horizon]['predictions']
        plt.figure(figsize=(15, 6))
        plt.plot(predictions_df.index, predictions_df['actual'], label='Actual', alpha=0.7)
        plt.plot(predictions_df.index, predictions_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (t+{horizon}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/predictions_over_time_h{horizon}.png')
        plt.close()


if __name__ == "__main__":
    main()



    

   