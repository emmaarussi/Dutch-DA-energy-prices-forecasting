"""
Simple linear model combining:
1. Significant price lags (selected per horizon from AR analysis)
2. Calendar features (from dummy model)

Features:
- Price lags: Selected significant lags for each horizon
- Hour of day (sine and cosine encoded)
- Day of week (sine and cosine encoded)
- Month of year (sine and cosine encoded)
- Calendar effects (is_weekend, is_holiday)
- Time of day effects (is_morning, is_evening)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

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

class SimpleLinearLagsAndDummies:
    def __init__(self):
        self.horizons = range(14, 39)  # 14 to 38 hours ahead
        self.models = {}  # One model per horizon
        self.scalers = {}  # One scaler per horizon
        self.significant_lags = {
            14: ['price_lag_1h', 'price_lag_2h', 'price_lag_168h'],  # From AR analysis
            24: ['price_lag_1h', 'price_lag_168h', 'price_lag_2h', 'price_lag_24h', 'price_lag_3h'],
            38: ['price_lag_1h', 'price_lag_168h', 'price_lag_2h', 'price_lag_24h', 'price_lag_3h']
        }
        # For other horizons, we'll determine lags during training
        
    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon."""
        # Get time features
        time_features = create_time_features(data)
        
        # Get price lags (if already determined for this horizon)
        if horizon in self.significant_lags:
            lag_features = data[self.significant_lags[horizon]]
        else:
            # For first run, use all hourly lags 1-24 and weekly lag
            lag_cols = [f'price_lag_{i}h' for i in range(1, 25)] + ['price_lag_168h']
            available_lags = [col for col in lag_cols if col in data.columns]
            lag_features = data[available_lags]
        
        # Combine features
        features = pd.concat([lag_features, time_features], axis=1)
        
        # Get target
        target = data[f'target_t{horizon}']
        
        # Align features and target
        combined = pd.concat([features, target], axis=1)
        combined = combined.dropna()
        
        X = combined[features.columns]
        y = combined[target.name]
        
        return X, y
    
    def select_significant_features(self, X, y, horizon):
        """Select significant features using OLS regression."""
        # Add constant for statsmodels
        X_const = add_constant(X)
        
        # Fit OLS model
        model = OLS(y, X_const)
        results = model.fit()
        
        # Get significant features (p < 0.05)
        significant_mask = results.pvalues[1:] < 0.05  # Skip constant
        significant_features = X.columns[significant_mask]
        
        # Always keep time features
        time_features = [col for col in X.columns if 'price_lag' not in col]
        final_features = list(set(significant_features) | set(time_features))
        
        # Store significant lags for this horizon
        lag_features = [f for f in significant_features if 'price_lag' in f]
        self.significant_lags[horizon] = lag_features
        
        return final_features, results
    
    def train(self, train_data):
        """Train models for all horizons."""
        print("\nTraining models for horizons 14-38 hours ahead...")
        
        # Store feature importance for key horizons
        self.feature_importance = {}
        
        for h in self.horizons:
            print(f"\nTraining model for t+{h} horizon...")
            
            # Prepare data
            X_train, y_train = self.prepare_data(train_data, h)
            
            # Select significant features
            selected_features, ols_results = self.select_significant_features(X_train, y_train, h)
            X_train = X_train[selected_features]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[h] = scaler
            
            # Train final model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            self.models[h] = {'model': model, 'features': selected_features}
            
            # Make predictions
            y_pred = model.predict(X_train_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            r2 = r2_score(y_train, y_pred)
            smape_val = smape(y_train, y_pred)
            
            print(f"Training metrics for horizon t+{h}:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R2: {r2:.2f}")
            print(f"SMAPE: {smape_val:.2f}%")
            
            # Store and print feature importance for key horizons
            if h in [14, 24, 38]:
                importance = pd.DataFrame({
                    'feature': selected_features,
                    'coefficient': model.coef_,
                    'scaled_coef': model.coef_ * scaler.scale_,
                    'p_value': ols_results.pvalues[1:len(selected_features)+1]  # Skip constant, match length
                })
                importance = importance.sort_values('scaled_coef', key=abs, ascending=False)
                self.feature_importance[h] = importance
                
                print("\nFeature importance:")
                print(importance.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    
    def predict(self, test_data):
        """Make predictions for all horizons."""
        predictions = pd.DataFrame(index=test_data.index)
        
        for h in self.horizons:
            # Prepare features
            X_test, _ = self.prepare_data(test_data, h)
            
            # Select only the features used in training
            model_info = self.models[h]
            X_test = X_test[model_info['features']]
            
            # Scale features
            X_test_scaled = self.scalers[h].transform(X_test)
            
            # Make predictions
            pred = model_info['model'].predict(X_test_scaled)
            predictions[f'pred_t{h}'] = pred
        
        return predictions

def plot_predictions(timestamps, actuals, predictions, horizon, title, save_path=None):
    """Create comparison plot for predictions vs actuals."""
    plt.figure(figsize=(15, 6))
    
    # Plot actual and predicted values
    plt.plot(timestamps, actuals, label='Actual', color='black', linewidth=2)
    plt.plot(timestamps, predictions, label=f'{horizon}-hour Forecast', 
             color='#9b59b6', linestyle='--', alpha=0.8)
    
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
    data = pd.read_csv('../../data/features/features_scaled.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
    data.set_index('timestamp', inplace=True)
    
    # Filter data from 2023 onwards
    data = data[data.index.year >= 2023]
    
    # Split into train/test
    train_start = pd.Timestamp('2023-01-01', tz='UTC')
    train_end = pd.Timestamp('2024-01-29', tz='UTC')
    test_end = pd.Timestamp('2024-03-01', tz='UTC')
    
    # Filter data to match training period
    mask = (data.index >= train_start) & (data.index < test_end)
    data = data[mask].copy()
    
    # Split into train and test sets
    train_data = data[data.index < train_end]
    test_data = data[data.index >= train_end]
    
    print("Data shape:", data.shape)
    print("Training period:", train_data.index[0], "to", train_data.index[-1])
    print("Test period:", test_data.index[0], "to", test_data.index[-1])
    
    # Train model
    model = SimpleLinearLagsAndDummies()
    model.train(train_data)
    
    # Make predictions
    predictions = model.predict(test_data)
    
    # Create necessary directories
    os.makedirs('plots/hybrid', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)

    # Save predictions
    predictions.to_csv('predictions/hybrid_predictions.csv')

    # Calculate and print test metrics
    print("\nTest metrics:")
    for h in [14, 24, 38]:  # Show metrics for key horizons
        actuals = test_data[f'target_t{h}']
        pred = predictions[f'pred_t{h}']
        
        print(f"\nHorizon t+{h}:")
        print(f"MAE: {mean_absolute_error(actuals, pred):.2f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(actuals, pred)):.2f}")
        print(f"R²: {r2_score(actuals, pred):.2f}")
        print(f"SMAPE: {smape(actuals, pred):.2f}%")
        
        # Create and save prediction plot
        plot_predictions(
            test_data.index, 
            actuals, 
            pred, 
            h,
            f'Linear Model with Lags and Dummies: {h}-hour Ahead Predictions',
            save_path=f'plots/hybrid/forecast_h{h}.png'
        )

if __name__ == "__main__":
    main()
