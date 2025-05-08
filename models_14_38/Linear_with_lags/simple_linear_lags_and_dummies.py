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

    print("Loading data...") 
    # Load multivariate features
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    # Initialize model
    model = SimpleLinearLagsAndDummies(horizons=[14, 24, 38]) 

    # Define train-test split
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
        metrics = result['metrics']
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
        importance_df = results[horizon]['feature_importance']
        plot_feature_importance(
            importance_df,
            top_n=20,
            title=f'Feature Importance (t+{horizon}h)',
            filename=f'{out_dir}/feature_importance_h{horizon}.png'
        )
        plt.close()
        
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
