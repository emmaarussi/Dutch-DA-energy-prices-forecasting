import pandas as pd
import numpy as np
import holidays
import os

def create_price_time_features(df):
    """Create time-based features from the timestamp index."""
    features = pd.DataFrame(index=df.index)
    features['hour'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek
    features['day_of_month'] = df.index.day
    features['month'] = df.index.month
    features['quarter'] = df.index.quarter
    features['year'] = df.index.year
    features['week_of_year'] = df.index.isocalendar().week

    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

    # Part of day indicators
    features['is_morning'] = (features['hour'] >= 6) & (features['hour'] < 12)
    features['is_afternoon'] = (features['hour'] >= 12) & (features['hour'] < 18)
    features['is_evening'] = (features['hour'] >= 18) & (features['hour'] < 22)
    features['is_night'] = (features['hour'] >= 22) | (features['hour'] < 6)
    features['is_weekend'] = features['day_of_week'].isin([5, 6])
    return features


def create_rolling_statistics(df, price_col='price_eur_per_mwh', windows=[3, 6, 12, 24, 48, 72, 168]):
    """Create rolling statistics for price data."""
    features = pd.DataFrame(index=df.index)
    for window in windows:
        features[f'rolling_mean_{window}h'] = df[price_col].rolling(window=window).mean()
        features[f'rolling_std_{window}h'] = df[price_col].rolling(window=window).std()
    return features

def create_price_differences(df, price_col='price_eur_per_mwh', windows=[1, 2, 3, 24, 48, 72, 168]):
    """Create price differences for time series forecasting."""
    features = pd.DataFrame(index=df.index)
    for window in windows:
        features[f'price_diff_{window}h'] = df[price_col].diff(window)
    return features

def create_holiday_features(df):
    """Create holiday-related features for the Netherlands."""
    nl_holidays = holidays.NL()
    features = pd.DataFrame(index=df.index)
    features['is_holiday'] = [date.date() in nl_holidays for date in df.index]
    features['is_day_before_holiday'] = features['is_holiday'].shift(-1, fill_value=False)
    features['is_day_after_holiday'] = features['is_holiday'].shift(1, fill_value=False)
    return features

def create_target_features(df, price_col='price_eur_per_mwh', forecast_horizon=38):
    """Create target variables for different forecast horizons."""
    targets = pd.DataFrame(index=df.index)
    for h in range(1, forecast_horizon + 1):
        targets[f'target_t{h}'] = df[price_col].shift(-h)
    return targets

def create_price_lagged_features(df):
    lags = list(range(1, 169))
    source = 'price_eur_per_mwh'

    if source not in df.columns:
        return pd.DataFrame(index=df.index)

    lagged_features = pd.DataFrame({
        f'{source}_lag_{lag}h': df[source].shift(lag) for lag in lags
    }, index=df.index)

    return lagged_features


def create_generation_lagged_features(df):
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    sources = ['wind', 'solar', 'coal', 'consumption', 'consumption_forecast', 'wind_forecast', 'solar_forecast']

    lagged_frames = []
    for source in sources:
        if source in df.columns:
            lagged = pd.DataFrame({
                f'{source}_lag_{lag}h': df[source].shift(lag) for lag in lags
            }, index=df.index)
            lagged_frames.append(lagged)

    if lagged_frames:
        return pd.concat(lagged_frames, axis=1)
    else:
        return pd.DataFrame(index=df.index)

def prepare_features_for_training(df, price_col='price_eur_per_mwh', forecast_horizon=38):
    """Prepare all features for training."""
    print("Creating features...")
    time_features = create_price_time_features(df)
    holiday_features = create_holiday_features(df)
    targets = create_target_features(df, price_col, forecast_horizon)
    generation_lags = create_generation_lagged_features(df)
    price_lags = create_price_lagged_features(df)
    rolling_stats = create_rolling_statistics(df)
    price_diffs = create_price_differences(df)
    features = pd.concat([time_features, holiday_features, generation_lags, price_lags, rolling_stats, price_diffs, targets], axis=1)
    features['current_price'] = df[price_col]

    # Clean up
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.loc[:, ~features.columns.duplicated()]
    features.dropna(inplace=True)

    print(f"Created {len(features.columns)} features.")
    return features

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, 'processed', 'merged_dataset_2023_2024_MW.csv'), index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')

    print("\nPreparing features...")
    df_features = prepare_features_for_training(df)

    print("\nSaving features...")
    df_features.to_csv(os.path.join(base_dir, 'processed', 'multivariate_features.csv'))

    print(f"\nFeature preparation complete. Shape: {df_features.shape}")
    print("Features available:")
    print(df_features.columns.tolist())

    cols = pd.read_csv(os.path.join(base_dir, 'processed', 'multivariate_features.csv'), nrows=1).columns
    duplicate_targets = [col for col in cols if col.startswith('target_t') and cols.tolist().count(col) > 1]
    print("Duplicate target columns:", duplicate_targets)

if __name__ == "__main__":
    main()
