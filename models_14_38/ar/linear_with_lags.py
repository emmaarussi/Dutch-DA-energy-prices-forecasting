"""
Linear model with lagged features for electricity price forecasting.
Uses only price lags to predict future prices.

Features:
- Recent hours: price_lag_1h, price_lag_2h, price_lag_3h
- Daily patterns: price_lag_24h, price_lag_48h
- Weekly pattern: price_lag_168h

Training period: January 2023 -- January 29, 2024
Test period: January 29, 2024 -- March 1, 2024
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance, rolling_window_evaluation


class SimpleARModel:
    def __init__(self):
        self.horizons = range(14, 39)  # 14 to 38 hours ahead
        self.models = {}  # One model per horizon
        
    def prepare_data(self, data, horizon):
        """Prepare data for a specific horizon."""
        # Create target (future price)
        target = data[f'target_t{horizon}']
        
        # Create lag features (1-24 hours and weekly)
        lag_features = []
        
        # Hourly lags 1-3
        for i in range(1, 4):
            lag_features.append(f'price_lag_{i}h')

        #daily lag
        lag_features.append('price_lag_24h')
        lag_features.append('price_lag_48h')
        
        # Weekly lag
        lag_features.append('price_lag_168h')
        
        # Select features that exist in the data
        available_features = [f for f in lag_features if f in data.columns]
        features = data[available_features]
        
        # Align features and target
        combined = pd.concat([features, target], axis=1)
        combined = combined.dropna()
        
        X = combined[available_features]
        y = combined[target.name]
        
        return X, y, available_features
    
    def train(self, train_data):
        """Train AR models for all horizons."""
        print("\nTraining AR models for horizons 14-38 hours ahead...")
        
        # Store feature importance
        self.feature_importance = {}
        
        for h in self.horizons:
            print(f"\nTraining model for t+{h} horizon...")
            
            # Prepare data with all lags
            X_train, y_train, _ = self.prepare_data(train_data, h)
            
            # First pass: identify significant features
            X_train_const = add_constant(X_train)
            initial_ols = OLS(y_train, X_train_const)
            initial_results = initial_ols.fit()
            
            # Select significant features (p < 0.05)
            significant_mask = initial_results.pvalues[1:] < 0.05
            significant_features = X_train.columns[significant_mask]
            
            if len(significant_features) == 0:
                print("No significant features found, using all features")
                significant_features = X_train.columns
            
            # Second pass: train final model with significant features
            X_train_final = X_train[significant_features]

            # Train sklearn model for predictions
            model = LinearRegression()
            model.fit(X_train_final, y_train)
            self.models[h] = {'model': model, 'features': significant_features}
            
            # Make in-sample predictions
            y_pred = model.predict(X_train_final)
            
            # calculate metrics
            metrics = calculate_metrics(y_train, y_pred)
            
            # Store and print feature importance
            importance = pd.DataFrame({
                'feature': significant_features,
                'coefficient': model.coef_,
                'std_err': model.bse,
                'p_value': model.pvalues,
                'abs_coef': np.abs(model.coef_)
            })
            importance = importance.sort_values('abs_coef', ascending=False)
            
            # Store importance for horizons 12 and 38
            if h in [12, 38]:
                self.feature_importance[h] = importance
            
            print("\nSignificant features:")
            print(importance.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    
            
    
    def predict(self, test_data):
        """Make predictions for all horizons."""
        predictions = pd.DataFrame(index=test_data.index)
        
        for h in self.horizons:
            # Prepare test data
            X_test, _, _ = self.prepare_data(test_data, h)
            
            # Select only the features used in training
            model_info = self.models[h]
            X_test = X_test[model_info['features']]
            
            # Make predictions
            pred = model_info['model'].predict(X_test)
            predictions[f'pred_t{h}'] = pred
        
        return predictions



def main():
 # Load data
    print("Loading data...")
    data = pd.read_csv('data/processed/multivariate_features.csv', index_col=0)
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
    
    # Print data information
    print(f"Data shape: {data.shape}")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Train model
    model = SimpleARModel()
    model.train(train_data)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_pred = model.predict(test_data)

    
    # Calculate test metrics and create plots for specific horizons
    horizons_to_plot = [14, 24, 38]  # Plot 14h, 24h, and 38h forecasts
    
    print("\nTest metrics:")
    for h in horizons_to_plot:
        y_true = test_data[f'target_t{h}'].dropna()
        y_pred = test_pred[f'pred_t{h}'].dropna()
        
        # Align indices
        common_idx = y_true.index.intersection(y_pred.index)
        y_true = y_true[common_idx]
        y_pred = y_pred[common_idx]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"\nHorizon t+{h}:")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R2: {metrics['r2']:.2f}")
        print(f"SMAPE: {metrics['smape']:.2f}%")
        
        # Create and save plot
        plt = plot_predictions(common_idx, y_true, y_pred, h,
                             f'Linear Model with Lagged Features: {h}-Hour Ahead Forecast vs Actual Values')
        # Create plots directory if it doesn't exist
        os.makedirs('models_14_38/ar/plots/linear_with_lags', exist_ok=True)
        plt.savefig(f'models_14_38/ar/plots/linear_with_lags/forecast_{h}h.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
