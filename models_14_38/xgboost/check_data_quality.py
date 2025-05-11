import numpy as np
import pandas as pd
from pathlib import Path

def prepare_data(data, horizon, feature_set='all'):
    """Prepare features and target for a specific horizon"""
    # Get all target columns
    target_cols = [col for col in data.columns if col.startswith('target_t')]
    
    # Define feature patterns to exclude for different feature sets
    excluded_patterns = {
        'price_only': ['wind', 'solar', 'consumption'],
        'no_weather': ['wind', 'solar'],
        'weather_only': ['price_t']
    }
    
    # Get feature columns based on feature set
    if feature_set == 'all':
        feature_cols = [col for col in data.columns if col not in target_cols]
    else:
        patterns = excluded_patterns.get(feature_set, [])
        feature_cols = [col for col in data.columns 
                       if col not in target_cols and 
                       not any(pattern in col for pattern in patterns)]
    
    print(f"Number of features: {len(feature_cols)}")
    print("Sample features:", ', '.join(sorted(feature_cols)[:5]))
    
    # Select features and target
    X = data[feature_cols]
    y = data[f'target_t{horizon}']
    
    return X, y

def main():
    # Load data
    data_path = Path('data/processed/multivariate_features.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')
    
    # Define train-test split
    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    
    train_df = df[train_start:train_end]
    test_df = df[test_start:test_end]
    
    # Check each feature set
    feature_sets = ['all', 'price_only', 'no_weather', 'weather_only']
    horizon = 14  # Check t+14h horizon
    
    for feature_set in feature_sets:
        print(f"\nChecking feature set: {feature_set}")
        X_train, y_train = prepare_data(train_df, horizon, feature_set)
        X_test, y_test = prepare_data(test_df, horizon, feature_set)
        
        print("\nTraining Data Quality:")
        print("NaNs in X_train:", np.isnan(X_train).sum().sum())
        print("NaNs in y_train:", np.isnan(y_train).sum())
        print("Infinities in X_train:", np.isinf(X_train).sum().sum())
        print("Infinities in y_train:", np.isinf(y_train).sum())
        
        print("\nTest Data Quality:")
        print("NaNs in X_test:", np.isnan(X_test).sum().sum())
        print("NaNs in y_test:", np.isnan(y_test).sum())
        print("Infinities in X_test:", np.isinf(X_test).sum().sum())
        print("Infinities in y_test:", np.isinf(y_test).sum())
        
        print("\nValue Ranges:")
        print("X_train min:", X_train.min().min())
        print("X_train max:", X_train.max().max())
        print("y_train min:", y_train.min())
        print("y_train max:", y_train.max())
        
        # Check for very large values
        print("\nFeatures with values > 1000:")
        for col in X_train.columns:
            if X_train[col].abs().max() > 1000:
                print(f"{col}: {X_train[col].abs().max():.2f}")

if __name__ == "__main__":
    main()
