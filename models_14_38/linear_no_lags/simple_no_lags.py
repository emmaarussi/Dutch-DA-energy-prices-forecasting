"""
Simple dummy regression model for energy price forecasting that uses only time-based features
without any lagged price variables. Features include:
- Hour of day (sine and cosine encoded)
- Day of week (sine and cosine encoded)
- Month of year (sine and cosine encoded)
- Calendar effects (is_weekend, is_holiday)
- Time of day effects (is_morning, is_evening)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance

def create_time_features(data):
    """Use existing time features from the data."""
    # Essential time features and calendar effects
    time_features = [
        'hour_sin', 'hour_cos',      # Hour of day
        'day_of_week_sin', 'day_of_week_cos',  # Day of week
        'month_sin', 'month_cos',    # Month of year
        'is_weekend', 'is_holiday',  # Calendar effects
        'is_morning', 'is_evening'   # Time of day effects
    ]
    
    features = data[time_features].copy()
    
    return features

class SimpleDummyRegression:
    def __init__(self):
        """Initialize the model."""
        self.horizons = range(14, 39)  # 14-38 hours ahead
        self.models = {}  # One model per horizon
        ##self.scalers = {}  # One scaler per horizon
        self.feature_names = None
        self.feature_importance = {}
        
    def prepare_data(self, df, horizon):
        """Prepare features and target for a specific horizon."""
        # Create time features
        features = create_time_features(df)
        self.feature_names = features.columns
        
        # Create target (future price)
        target = df[f'target_t{horizon}']
        
        # Align indices and remove NaN
        combined = pd.concat([features, target], axis=1)
        combined = combined.dropna()
        
        X = combined[features.columns]
        y = combined[target.name]
        
        return X, y
    
    def train(self, train_df):
        """Train models for all horizons."""
        print("\nTraining models for horizons 14-38 hours ahead...")
        
        for h in self.horizons:
            print(f"\nTraining model for t+{h} horizon...")
            
            # Prepare data
            X_train, y_train = self.prepare_data(train_df, h)
            
            # Scale features
            ##scaler = StandardScaler()
            ##X_train_scaled = scaler.fit_transform(X_train)
            ##self.scalers[h] = scaler
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models[h] = model
            
            # Store feature importance
            importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': abs(model.coef_)
            })
            self.feature_importance[h] = importance
            
            # Get predictions
            y_pred = model.predict(X_train)
            
            # Calculate metrics
            metrics = calculate_metrics(y_train, y_pred)
            
            print(f"Training metrics for horizon t+{h}:")
            print(f"MAE: {metrics['MAE']:.2f}")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"R2: {metrics['R2']:.2f}")
            print(f"SMAPE: {metrics['SMAPE']:.2f}%")
            
    def predict(self, df):
        """Make predictions for all horizons."""
        predictions = pd.DataFrame(index=df.index)

        # Select only time-based features for prediction
        features = create_time_features(df)
        
        # Debug info
        print("🚨 Feature DataFrame shape:", features.shape)
        print("🔍 Any NaNs in features:", features.isnull().any().any())
        print("🧪 Sample of features:")
        print(features.head())
        print("📋 Feature columns:", list(features.columns))

        for h in self.horizons:
            if h not in self.models:
                print(f"⚠️ No trained model for t+{h}, skipping.")
                continue

            model = self.models[h]
            pred = model.predict(features)
            predictions[f'pred_t{h}'] = pd.Series(pred, index=features.index)

        return predictions




def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/processed/multivariate_features.csv', index_col=0, parse_dates=True)

    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-01', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-01', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')


    # Split into train and test sets
    train_df = df[(df.index >= train_start) & (df.index < train_end)]
    test_df = df[(df.index >= test_start) & (df.index < test_end)]

    # Train model
    model = SimpleDummyRegression()
    model.train(train_df)

    print("Data shape:", df.shape)

    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_pred = model.predict(test_df)
    
    # Create directories if they don't exist
    os.makedirs('models_14_38/linear_no_lags/plots', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    # Save predictions
    test_pred.index = test_df.index
    test_pred.to_csv('predictions/simple_no_lags_predictions.csv')
    
    # Plot predictions over time for key horizons
    for horizon in [14, 24, 38]:
        actual = test_df[f'target_t{horizon}']
        pred = test_pred[f'pred_t{horizon}']
        
        # Remove NaN values
        mask = ~actual.isna() & ~pred.isna()
        actual = actual[mask]
        pred = pred[mask]
        timestamps = actual.index
        
        # Calculate metrics
        metrics = calculate_metrics(actual, pred)
        
        print(f"\nHorizon t+{horizon}:")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"R2: {metrics['R2']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        
        # Plot predictions vs actual
        plt.figure(figsize=(15, 6))
        plt.plot(timestamps, actual, label='Actual', alpha=0.7)
        plt.plot(timestamps, pred, label='Predicted', alpha=0.7)
        plt.title(f'Predictions vs Actual (t+{horizon}h)')
        plt.xlabel('Time')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'models_14_38/linear_no_lags/plots/predictions_h{horizon}.png')
        plt.close()
        
        # Create plots directory if it doesn't exist
    os.makedirs('models_14_38/linear_no_lags/plots', exist_ok=True)
    
    # Plot feature importance for each horizon
    for horizon in [14, 24, 38]:
        importance_df = model.feature_importance[horizon]
        plot_feature_importance(
            importance_df,
            top_n=20,
            title=f'Feature Importance (t+{horizon}h)',
            filename=f'models_14_38/linear_no_lags/plots/feature_importance_h{horizon}.png'
        )
        plt.close()
        
    for horizon in [14, 24, 38]:    # Plot predictions over time
        predictions_df = all_predictions[horizon]
        plt.figure(figsize=(15, 6))
        plt.plot(predictions_df.index, predictions_df['actual'], label='Actual', alpha=0.7)
        plt.plot(predictions_df.index, predictions_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (t+{horizon}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'models_14_38/linear_no_lags/plots/predictions_over_time_h{horizon}.png')
        plt.close()

if __name__ == "__main__":
    main()
