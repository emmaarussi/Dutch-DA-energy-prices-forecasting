"""
Prepare multivariate dataset combining price and generation data for energy price forecasting.
"""
import pandas as pd
import numpy as np
import holidays
import os

def create_price_time_features(df):
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

def create_price_lag_features(df, price_col='price_eur_per_mwh', lags=[1, 2, 3, 24, 48, 72, 168]):
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
    """
    
    features = pd.DataFrame(index=df.index)
    
    # Create lag features
    for lag in lags:
        features[f'price_lag_{lag}h'] = df[price_col].shift(lag)
    
    # Create rolling mean features
    ##windows = [3, 6, 12, 24, 48, 72, 168]  # hours
    ##or window in windows:
        ##features[f'rolling_mean_{window}h'] = df[price_col].rolling(window=window).mean()
        ##features[f'rolling_std_{window}h'] = df[price_col].rolling(window=window).std()
    
    return features

def create_target_features(df, price_col='price_eur_per_mwh', forecast_horizon=38):
    """Create target variables for different forecast horizons."""
    
    targets = pd.DataFrame(index=df.index)
    
    # Create target variables for different horizons
    for h in range(1, forecast_horizon + 1):
        targets[f'target_t{h}'] = df[price_col].shift(-h)
    
    return targets

def create_generation_lagged_features(df):
    """Create comprehensive lag features."""
    df = df.copy()
    
    # Define lags
    lags = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h to weekly

    
    # Create generation lags
    all_sources = ['wind', 'solar', 'coal', 'consumption', 'consumption_forecast', 'wind_forecast', 'solar_forecast']
    for source in all_sources:
        if source in df.columns:
            for lag in lags:
                df[f'{source}_lag_{lag}h'] = df[source].shift(lag)
    
    return df


##def create_rolling_statistics(df):
    """Create comprehensive rolling statistics."""
    df = df.  copy()
    
    # Define windows
    windows = [6, 12, 24, 48, 168]  # 6h to weekly
    
    # Generation rolling statistics
    all_sources_rolling = ['wind', 'solar', 'coal', 'consumption', 'consumption_forecast', 'wind_forecast', 'solar_forecast']
    for source in all_sources_rolling:
        if source in df.columns:
            for window in windows:
                df[f'{source}_rolling_mean_{window}h'] = df[source].rolling(window=window).mean()
                df[f'{source}_rolling_std_{window}h'] = df[source].rolling(window=window).std()
                df[f'{source}_rolling_min_{window}h'] = df[source].rolling(window=window).min()
                df[f'{source}_rolling_max_{window}h'] = df[source].rolling(window=window).max()
    
    return df

def prepare_features_for_training(df, price_col='price_eur_per_mwh', forecast_horizon=38):
    """Prepare all features for training."""
    
    print("Creating features...")
    
    # Create different feature groups
    time_features = create_price_time_features(df)
    holiday_features = create_holiday_features(df)
    lag_features = create_price_lag_features(df, price_col)
    targets = create_target_features(df, price_col, forecast_horizon)
    generation_lagged_features = create_generation_lagged_features(df)
    ##rolling_statistics = create_rolling_statistics(df)
    
    # Combine all features
    features = pd.concat([
        time_features,
        holiday_features,
        lag_features,
        generation_lagged_features,
        ##rolling_statistics,
        targets
    ], axis=1)
    
    # Add the current price
    features['current_price'] = df[price_col]
    
    # Enhanced price differences and momentum
    ##features['price_diff_1h'] = df[price_col].diff()
    ##features['price_diff_24h'] = df[price_col].diff(24)
    ##features['price_diff_168h'] = df[price_col].diff(168)  # week difference
    
    # Price momentum (percentage changes)
    ##features['price_momentum_1h'] = df[price_col].pct_change()
    ##features['price_momentum_24h'] = df[price_col].pct_change(24)
    ##features['price_momentum_168h'] = df[price_col].pct_change(168)
    
    # Volatility features
    ##for window in [24, 48, 168]:
        ##features[f'volatility_{window}h'] = df[price_col].rolling(window=window).std() / df[price_col].rolling(window=window).mean()
        ##features[f'range_{window}h'] = df[price_col].rolling(window=window).max() - df[price_col].rolling(window=window).min()
    
    # Price levels and extremes
    ##for window in [24, 48, 168]:
        ##features[f'price_max_{window}h'] = df[price_col].rolling(window=window).max()
        ##features[f'price_min_{window}h'] = df[price_col].rolling(window=window).min()
        ##features[f'price_quantile_25_{window}h'] = df[price_col].rolling(window=window).quantile(0.25)
        ##features[f'price_quantile_75_{window}h'] = df[price_col].rolling(window=window).quantile(0.75)
    
    # Trend indicators
    ##for window in [24, 48, 168]:
        ## Simple moving average crossovers
        ##sma_short = df[price_col].rolling(window=window//4).mean()
        ##sma_long = df[price_col].rolling(window=window).mean()
        ##features[f'trend_sma_{window}h'] = (sma_short - sma_long) / sma_long
        
        # Exponential moving average
        ##features[f'trend_ema_{window}h'] = df[price_col].ewm(span=window).mean()
    
    # Add targets
    features = pd.concat([features, targets], axis=1)
    
    # Handle infinities and NaN values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()
    
    print(f"Created {len(features.columns)} features:")
    print("\nFeature groups:")
    print(f"- Time features: {len(time_features.columns)}")
    print(f"- Holiday features: {len(holiday_features.columns)}")
    print(f"- Lag features: {len(lag_features.columns)}")
    print(f"- Generation lagged features: {len(generation_lagged_features.columns)}")
    ##print(f"- Rolling statistics: {len(rolling_statistics.columns)}")
    print(f"- Target variables: {len(targets.columns)}")
    
    return features

    """Create comprehensive time-based features."""
    df = df.copy()
    
    # Basic time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['week_of_year'] = df.index.isocalendar().week
    
    # Cyclical encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Part of day features
    df['is_morning'] = (df['hour'] >= 6) & (df['hour'] < 12)
    df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] < 18)
    df['is_evening'] = (df['hour'] >= 18) & (df['hour'] < 22)
    df['is_night'] = (df['hour'] >= 22) | (df['hour'] < 6)
    
    # Weekend feature
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    return df


def main():
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    #load data
    df = pd.read_csv(os.path.join(base_dir, 'processed', 'merged_dataset_2023_2024_MW.csv'), index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')

    # Prepare features
    print("\nPreparing features...")
    df_features = prepare_features_for_training(df)
    
    # Save features
    print("\nSaving features...")
    df_features.to_csv(os.path.join(base_dir, 'processed', 'multivariate_features.csv'))
    
    print(f"\nFeature preparation complete. Shape: {df_features.shape}")
    print("\nFeatures available:")
    print(df_features.columns.tolist())

if __name__ == "__main__":
    main()
