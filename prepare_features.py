import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from sklearn.preprocessing import StandardScaler
import joblib

def create_time_features(df):
    """Create time-based features from the timestamp index.
    
    This function generates both raw and cyclically encoded time features to capture
    temporal patterns in the data. Cyclical encoding ensures continuity at period boundaries.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        pd.DataFrame: Time-based features including:
            - Basic features (hour, day, month, etc.)
            - Cyclical features (sin/cos encoding)
            - Weekend indicator
    """
    
    # Basic time features
    features = pd.DataFrame(index=df.index)
    features['hour'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek
    features['day_of_month'] = df.index.day
    features['month'] = df.index.month
    features['quarter'] = df.index.quarter
    features['year'] = df.index.year
    features['week_of_year'] = df.index.isocalendar().week
    
    # Cyclical encoding of time features
    features['hour_sin'] = np.sin(2 * np.pi * features['hour']/24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour']/24)
    features['month_sin'] = np.sin(2 * np.pi * features['month']/12)
    features['month_cos'] = np.cos(2 * np.pi * features['month']/12)
    features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week']/7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week']/7)
    
    # Part of day features
    features['is_morning'] = (features['hour'] >= 6) & (features['hour'] < 12)
    features['is_afternoon'] = (features['hour'] >= 12) & (features['hour'] < 18)
    features['is_evening'] = (features['hour'] >= 18) & (features['hour'] < 22)
    features['is_night'] = (features['hour'] >= 22) | (features['hour'] < 6)
    
    # Weekend feature
    features['is_weekend'] = features['day_of_week'].isin([5, 6])
    
    return features

def create_holiday_features(df):
    """Create holiday-related features for the Netherlands.
    
    Generates binary indicators for holidays and special days that might
    affect energy prices.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        pd.DataFrame: Holiday features including:
            - is_holiday: Public holiday indicator
            - is_weekend: Weekend indicator
            - is_business_day: Business day indicator
    """
    
    nl_holidays = holidays.NL()
    
    # Create holiday features
    features = pd.DataFrame(index=df.index)
    features['is_holiday'] = [date.date() in nl_holidays for date in df.index]
    
    # Create features for days before and after holidays
    features['is_day_before_holiday'] = features['is_holiday'].shift(-1, fill_value=False)
    features['is_day_after_holiday'] = features['is_holiday'].shift(1, fill_value=False)
    
    return features

def create_lag_features(df, price_col='price_eur_per_mwh', lags=[1, 2, 3, 24, 48, 72, 168]):
    """Create lagged price features for time series forecasting.
    
    Generates lagged values and rolling statistics to capture price history
    and trends at different time scales.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        price_col (str): Name of the price column
        lags (list): List of lag periods in hours
        
    Returns:
        pd.DataFrame: Lagged features including:
            - Price lags
            - Rolling means
            - Rolling standard deviations
            - Price differences
    """
    
    features = pd.DataFrame(index=df.index)
    
    # Create lag features
    for lag in lags:
        features[f'price_lag_{lag}h'] = df[price_col].shift(lag)
    
    # Create rolling mean features
    windows = [3, 6, 12, 24, 48, 72, 168]  # hours
    for window in windows:
        features[f'rolling_mean_{window}h'] = df[price_col].rolling(window=window).mean()
        features[f'rolling_std_{window}h'] = df[price_col].rolling(window=window).std()
    
    return features

def create_target_features(df, price_col='price_eur_per_mwh', forecast_horizon=24):
    """Create target variables for different forecast horizons."""
    
    targets = pd.DataFrame(index=df.index)
    
    # Create target variables for different horizons
    for h in range(1, forecast_horizon + 1):
        targets[f'target_t{h}'] = df[price_col].shift(-h)
    
    return targets

def prepare_features_for_training(df, price_col='price_eur_per_mwh', forecast_horizon=24):
    """Prepare all features for training."""
    
    print("Creating features...")
    
    # Create different feature groups
    time_features = create_time_features(df)
    holiday_features = create_holiday_features(df)
    lag_features = create_lag_features(df, price_col)
    targets = create_target_features(df, price_col, forecast_horizon)
    
    # Combine all features
    features = pd.concat([
        time_features,
        holiday_features,
        lag_features,
    ], axis=1)
    
    # Add the current price
    features['current_price'] = df[price_col]
    
    # Enhanced price differences and momentum
    features['price_diff_1h'] = df[price_col].diff()
    features['price_diff_24h'] = df[price_col].diff(24)
    features['price_diff_168h'] = df[price_col].diff(168)  # week difference
    
    # Price momentum (percentage changes)
    features['price_momentum_1h'] = df[price_col].pct_change()
    features['price_momentum_24h'] = df[price_col].pct_change(24)
    features['price_momentum_168h'] = df[price_col].pct_change(168)
    
    # Volatility features
    for window in [24, 48, 168]:
        features[f'volatility_{window}h'] = df[price_col].rolling(window=window).std() / df[price_col].rolling(window=window).mean()
        features[f'range_{window}h'] = df[price_col].rolling(window=window).max() - df[price_col].rolling(window=window).min()
    
    # Price levels and extremes
    for window in [24, 48, 168]:
        features[f'price_max_{window}h'] = df[price_col].rolling(window=window).max()
        features[f'price_min_{window}h'] = df[price_col].rolling(window=window).min()
        features[f'price_quantile_25_{window}h'] = df[price_col].rolling(window=window).quantile(0.25)
        features[f'price_quantile_75_{window}h'] = df[price_col].rolling(window=window).quantile(0.75)
    
    # Trend indicators
    for window in [24, 48, 168]:
        # Simple moving average crossovers
        sma_short = df[price_col].rolling(window=window//4).mean()
        sma_long = df[price_col].rolling(window=window).mean()
        features[f'trend_sma_{window}h'] = (sma_short - sma_long) / sma_long
        
        # Exponential moving average
        features[f'trend_ema_{window}h'] = df[price_col].ewm(span=window).mean()
    
    # Add targets
    features = pd.concat([features, targets], axis=1)
    
    # Drop rows with NaN values (due to lagging)
    features = features.dropna()
    
    print(f"Created {len(features.columns)} features:")
    print("\nFeature groups:")
    print(f"- Time features: {len(time_features.columns)}")
    print(f"- Holiday features: {len(holiday_features.columns)}")
    print(f"- Lag features: {len(lag_features.columns)}")
    print(f"- Target variables: {len(targets.columns)}")
    
    return features

def scale_features(features, exclude_cols=None, save_scaler=True):
    """Scale features using StandardScaler."""
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Separate columns to scale and not to scale
    cols_to_scale = [col for col in features.columns if col not in exclude_cols]
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale features
    scaled_features = features.copy()
    scaled_features[cols_to_scale] = scaler.fit_transform(features[cols_to_scale])
    
    if save_scaler:
        joblib.dump(scaler, 'data/feature_scaler.joblib')
        print("\nScaler saved to data/feature_scaler.joblib")
    
    return scaled_features, scaler

def main():
    # Load the price data
    print("Loading data...")
    df = pd.read_csv('data/raw_prices.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_convert('Europe/Amsterdam')
    
    # Prepare features
    features = prepare_features_for_training(df)
    
    # Scale features, excluding target variables
    target_cols = [col for col in features.columns if col.startswith('target_')]
    scaled_features, _ = scale_features(features, exclude_cols=target_cols)
    
    # Save features
    features.to_csv('data/features_unscaled.csv')
    scaled_features.to_csv('data/features_scaled.csv')
    
    print("\nFeatures saved to:")
    print("- data/features_unscaled.csv")
    print("- data/features_scaled.csv")
    
    # Print sample of features
    print("\nSample of features (first 5 rows):")
    print(features.head())

if __name__ == "__main__":
    main()
