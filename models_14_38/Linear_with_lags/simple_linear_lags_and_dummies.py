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
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

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
            14: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_10h','price_lag_154h'],
            15: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_9h','price_lag_153h'],
            16: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_8h','price_lag_152h'],
            17: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_7h','price_lag_151h'],
            18: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_6h','price_lag_150h'],
            19: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_5h','price_lag_149h'],
            20: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_4h','price_lag_148h'],
            21: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_147h'],
            22: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_146h'],
            23: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_145h'],
            24: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_144h'],
            25: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_23h','price_lag_143h'],
            26: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_22h','price_lag_142h'],
            27: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_21h','price_lag_141h'],
            28: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_20h','price_lag_140h'],
            29: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_19h','price_lag_139h'],
            30: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_18h','price_lag_138h'],
            31: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_17h','price_lag_137h'],
            32: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_16h','price_lag_136h'],
            33: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_15h','price_lag_135h'],
            34: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_14h','price_lag_134h'],
            35: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_13h','price_lag_133h'],
            36: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_12h','price_lag_132h'],
            37: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_11h','price_lag_131h'],
            38: ['price_lag_1h', 'price_lag_2h','price_lag_3h','price_lag_10h','price_lag_130h'],
        }
        
        
    def prepare_data(self, data, horizon):
        """Prepare features and target for a specific horizon."""
        # Get time features
        time_features = create_time_features(data)
        
        # Get price lags (if already determined for this horizon)
        lag_features = data[self.significant_lags[horizon]]
        
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
    
    
    def train(self, train_data):
        """Train models for all horizons."""
        print("\nTraining models for horizons 14-38 hours ahead...")
        
        # Store feature importance for key horizons
        self.feature_importance = {}
        
        for h in self.horizons:
            print(f"\nTraining model for t+{h} horizon...")
            
            # Prepare data
            X_train, y_train = self.prepare_data(train_data, h)
            
            
            # Scale features
            ##scaler = StandardScaler()
            ##X_train_scaled = scaler.fit_transform(X_train)
            ##self.scalers[h] = scaler
            
            # Train final model
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models[h] = {'model': model, 'features': features}
            
            # Make predictions
            y_pred = model.predict(X_train)
            
            # Calculate metrics
            metrics = calculate_metrics(y_train, y_pred)
            
            print(f"Training metrics for horizon t+{h}:")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"R2: {metrics['r2']:.2f}")
            print(f"SMAPE: {metrics['smape']:.2f}%")
            
            # Store and print feature importance for key horizons
            if h in [14, 24, 38]:
                importance = pd.DataFrame({
                    'feature': selected_features,
                    'coefficient': model.coef_,
                    ##'scaled_coef': model.coef_ * scaler.scale_,
                    'p_value': ols_results.pvalues[1:len(selected_features)+1]  # Skip constant, match length
                })
                importance = importance.sort_values('coefficient', key=abs, ascending=False)
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
            
            # Make predictions
            pred = model_info['model'].predict(X_test)
            predictions[f'pred_t{h}'] = pred
        
        return predictions



def main():
    # Load data
    # Load data
    print("Loading data...")
    data = pd.read_csv('data/processed/multivariate_features.csv')

    data.index = pd.to_datetime(data.index, utc=True)
    data = data.asfreq('h')  
    
    # Split into training (up to Jan 2024) and test data
    train_data = data[data.index < '2024-01-01']
    test_data = data[data.index >= '2024-01-01']
    
    print("Data shape:", data.shape)

    
    # Train model
    model = SimpleLinearLagsAndDummies()
    model.train(train_data)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    predictions = model.predict(test_data)
    
    # Create necessary directories
    os.makedirs('plots/hybrid', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)



    # Calculate and print test metrics
    print("\nTest metrics:")
    for h in [14, 24, 38]:  # Show metrics for key horizons
        actuals = test_data[f'target_t{h}']
        pred = predictions[f'pred_t{h}']
        
        # Calculate metrics
        metrics = calculate_metrics(actual, pred)
        
        print(f"\nHorizon t+{h}:")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"R2: {metrics['R2']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        
        # Create and save prediction plot
        plot_predictions(test_data.index, actuals, predictions, h, f'Linear Model with Lags and Dummies: {h}-hour Ahead Predictions', save_path=f'plots/hybrid/forecast_h{h}.png')
    
    # Plot predictions vs actuals
        plt.figure(figsize=(15, 6))
        plt.plot(timestamps, actuals, label='Actual', alpha=0.7)
        plt.plot(timestamps, predictions, label='Predicted', alpha=0.7)
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (EUR/MWh)', fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs('plots/linear_with_lags', exist_ok=True)
        plt.savefig(f'plots/linear_with_lags/forecast_h{h}.png')
        plt.close()
    return plt
        
    

if __name__ == "__main__":
    main()
