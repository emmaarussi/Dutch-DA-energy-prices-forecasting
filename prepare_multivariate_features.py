"""
Prepare multivariate dataset combining price and generation data for energy price forecasting.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_and_merge_data():
    """Load price and generation data and merge them."""
    # Load price data
    price_path = os.path.join('data', 'raw_prices.csv')
    price_df = pd.read_csv(price_path)
    price_df['time'] = pd.to_datetime(price_df['time'])
    price_df.set_index('time', inplace=True)
    
    # Load generation data
    gen_path = os.path.join('data', 'generation_by_source.csv')
    gen_df = pd.read_csv(gen_path)
    gen_df['validfrom'] = pd.to_datetime(gen_df['validfrom'])
    gen_df['validto'] = pd.to_datetime(gen_df['validto'])
    
    # Process generation data
    source_mapping = {
        17: 'wind_onshore',
        12: 'solar',
        4: 'wind_offshore'
    }
    
    # Filter for renewable sources
    gen_df = gen_df[gen_df['type'].isin(source_mapping.keys())]
    gen_df['source'] = gen_df['type'].map(source_mapping)
    
    # Pivot and resample to hourly frequency
    gen_hourly = (gen_df.groupby(['validfrom', 'source'])['capacity']
                 .mean()
                 .reset_index()
                 .pivot(index='validfrom', columns='source', values='capacity')
                 .resample('h')
                 .mean()
                 .ffill())
    
    # Merge price and generation data
    df = price_df.join(gen_hourly)
    
    return df

def create_time_features(df):
    """Create time-based features."""
    df = df.copy()
    
    # Extract time components
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # Create cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    return df

def create_lagged_features(df, price_lags=None, generation_lags=None):
    """Create lagged features for both price and generation."""
    if price_lags is None:
        price_lags = [24, 48, 72, 168]  # 1 day, 2 days, 3 days, 1 week
    
    if generation_lags is None:
        generation_lags = [24, 48]  # 1 day, 2 days
    
    df = df.copy()
    
    # Create price lags
    for lag in price_lags:
        df[f'price_lag_{lag}h'] = df['price_eur_per_mwh'].shift(lag)
    
    # Create generation lags for each source
    generation_sources = ['wind_onshore', 'solar', 'wind_offshore']
    for source in generation_sources:
        if source in df.columns:
            for lag in generation_lags:
                df[f'{source}_lag_{lag}h'] = df[source].shift(lag)
    
    return df

def create_rolling_statistics(df, windows=None):
    """Create rolling statistics for price and generation."""
    if windows is None:
        windows = [24, 168]  # 1 day, 1 week
    
    df = df.copy()
    
    # Create rolling stats for price
    for window in windows:
        df[f'price_rolling_mean_{window}h'] = df['price_eur_per_mwh'].rolling(window=window).mean()
        df[f'price_rolling_std_{window}h'] = df['price_eur_per_mwh'].rolling(window=window).std()
    
    # Create rolling stats for generation
    generation_sources = ['wind_onshore', 'solar', 'wind_offshore']
    for source in generation_sources:
        if source in df.columns:
            for window in windows:
                df[f'{source}_rolling_mean_{window}h'] = df[source].rolling(window=window).mean()
                df[f'{source}_rolling_std_{window}h'] = df[source].rolling(window=window).std()
    
    return df

def prepare_features(df):
    """Prepare all features for the multivariate model."""
    # Create time features
    df = create_time_features(df)
    
    # Create lagged features
    df = create_lagged_features(df)
    
    # Create rolling statistics
    df = create_rolling_statistics(df)
    
    # Drop rows with NaN values (due to lagging/rolling operations)
    df = df.dropna()
    
    return df

def main():
    # Load and merge data
    print("Loading and merging data...")
    df = load_and_merge_data()
    
    # Prepare features
    print("\nPreparing features...")
    df_features = prepare_features(df)
    
    # Save features
    print("\nSaving features...")
    df_features.to_csv(os.path.join('data', 'multivariate_features.csv'))
    
    print(f"\nFeature preparation complete. Shape: {df_features.shape}")
    print("\nFeatures available:")
    print(df_features.columns.tolist())

if __name__ == "__main__":
    main()
