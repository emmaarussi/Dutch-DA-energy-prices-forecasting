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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
from datetime import datetime, timedelta

def create_time_features(df):
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

class SimpleDummyRegression:
    def __init__(self):
        self.horizons = range(14, 39)  # 14 to 38 hours ahead
        self.models = {}  # One model per horizon
        self.scalers = {}  # One scaler per horizon
        self.feature_names = None
        
    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon."""
        # Create time features
        features = create_time_features(data)
        self.feature_names = features.columns
        
        # Using only time-based features
            
        # Create target (future price)
        target = data[f'target_t{horizon}']
        
        # Align indices and remove NaN
        combined = pd.concat([features, target], axis=1)
        combined = combined.dropna()
        
        X = combined[features.columns]
        y = combined[target.name]
        
        return X, y
    
    def train(self, train_data):
        """Train models for all horizons."""
        print("\nTraining models for horizons 14-38 hours ahead...")
        
        for h in self.horizons:
            print(f"\nTraining model for t+{h} horizon...")
            
            # Prepare data
            X_train, y_train = self.prepare_data(train_data, h)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[h] = scaler
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            self.models[h] = model
            
            # Calculate training metrics
            y_pred = model.predict(X_train_scaled)
            mae = mean_absolute_error(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            r2 = r2_score(y_train, y_pred)
            smape_val = smape(y_train, y_pred)
            
            print(f"Training metrics for horizon t+{h}:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R2: {r2:.2f}")
            print(f"SMAPE: {smape_val:.2f}%")
    
    def predict(self, X):
        """Make predictions for all horizons."""
        predictions = pd.DataFrame(index=X.index)
        
        # Create features
        features = create_time_features(X)
        
        # Using only time-based features
        
        # Align indices
        features = features.loc[X.index]
        
        for h in self.horizons:
            # Scale features
            features_scaled = self.scalers[h].transform(features)
            
            # Make predictions
            pred = self.models[h].predict(features_scaled)
            predictions[f'pred_t{h}'] = pd.Series(pred, index=features.index)
        
        return predictions

def plot_predictions(timestamps, actuals, predictions, horizon, title, save_path=None):
    """Create comparison plot for predictions vs actuals."""
    plt.figure(figsize=(15, 6))
    
    # Plot actual and predicted values
    plt.plot(timestamps, actuals, label='Actual', color='black', linewidth=2)
    plt.plot(timestamps, predictions, label=f'{horizon}-hour Forecast', 
             color='#2ecc71', linestyle='--', alpha=0.8)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (EUR/MWh)', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return plt

def main():
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/processed/multivariate_features.csv')

    data.index = pd.to_datetime(data.index, utc=True)
    data = data.asfreq('H')  # Set hourly frequency
    
    # Split into train/test (matching other models' timeframes)
    train_start = pd.Timestamp('2023-01-01', tz='UTC')
    train_end = pd.Timestamp('2024-01-29', tz='UTC')
    test_end = pd.Timestamp('2024-03-01', tz='UTC')
    
    # Filter data to match training period
    mask = (data.index >= train_start) & (data.index < test_end)
    data = data[mask].copy()
    
    # Split into train and test
    train_data = data[data.index < train_end].copy()
    test_data = data[data.index >= train_end].copy()
 
    
    # Train model
    model = SimpleDummyRegression()
    model.train(train_data)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_pred = model.predict(test_data)
    
    # Create necessary directories if they don't exist
    os.makedirs('plots/simple_no_lags', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)

    # Save predictions
    test_pred.index = test_data.index
    test_pred.to_csv('predictions/simple_no_lags_predictions.csv')
    
    # Calculate test metrics and create plots for specific horizons
    horizons_to_plot = [14, 24, 38]  # Plot 14h, 24h, and 38h forecasts
    
    for h in horizons_to_plot:
        actual = test_data[f'target_t{h}'][test_pred.index]
        pred = test_pred[f'pred_t{h}']
        
        # Remove NaN values
        mask = ~actual.isna() & ~pred.isna()
        actual = actual[mask]
        pred = pred[mask]
        timestamps = actual.index
        
        # Calculate metrics
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        smape_val = smape(actual, pred)
        
        print(f"\nHorizon t+{h}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.2f}")
        print(f"SMAPE: {smape_val:.2f}%")
        
        # Create and save plot
        title = f'Simple Time-based Model - {h}-hour Forecast vs Actual'
        plot_predictions(timestamps, actual, pred, h, title, 
                        save_path=f'plots/simple_no_lags/forecast_h{h}.png')

if __name__ == "__main__":
    main()
