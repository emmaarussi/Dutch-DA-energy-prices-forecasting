"""
Process generation data from the NED API and combine with ENTSO-E data.
Only uses types 1 (wind) and 2 (solar) from the generation dataset from NED api
All forecasts from ENTSO-E, prices and consumption from ENTSO-E

Includes data quality checks and interpolation for missing values.
"""
import pandas as pd
from entsoe import EntsoePandasClient
import numpy as np
import time
import os
import requests

def process_raw_data():
    # Set up parameters
    start = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    end = pd.Timestamp('2024-05-01', tz='Europe/Amsterdam')  
    country_code = 'NL'
    
    print("Initializing ENTSO-E client...")
    client = EntsoePandasClient(api_key='23128a97-4438-4a5e-af61-76d1189ebb95')
    client.session.timeout = 60  # 60 seconds timeout
    
    print("Loading historical generation data...")
    # Load the generation data FROM NED API
    df_gen = pd.read_csv('data/generation_by_source.csv')
    
    # Convert timestamps to datetime
    df_gen['validfrom'] = pd.to_datetime(df_gen['validfrom'])
    df_gen['validto'] = pd.to_datetime(df_gen['validto'])
    
    # Filter for wind (type 1, no distinction between off and onshore) and solar (type 2), coal (type 8) and gas (type 23)
    df_gen = df_gen[df_gen['type'].isin([1, 2, 8])]
    
    # Create separate dataframes for wind and solar
    df_wind = df_gen[df_gen['type'] == 1].copy()
    df_solar = df_gen[df_gen['type'] == 2].copy()
    df_kolen = df_gen[df_gen['type'] == 8].copy()

    
    # Set index to validfrom and resample to hourly intervals
    df_wind.set_index('validfrom', inplace=True)
    df_solar.set_index('validfrom', inplace=True)
    df_kolen.set_index('validfrom', inplace=True)

    
    # Resample to hourly intervals using mean
    df_wind = df_wind.resample('h')['capacity'].mean()
    df_solar = df_solar.resample('h')['capacity'].mean()
    df_kolen = df_kolen.resample('h')['capacity'].mean()

    #change to MW
    df_wind = df_wind / 1000
    df_solar = df_solar / 1000
    df_kolen = df_kolen / 1000
    
    # Keeping data in original units, which are kW, for consistency with the raw data.
    # will convert to MW in feature engineering step for consistency with other variables
    
    print("\nFetching forecasts...")
    # Get generation and consumption forecasts
    wind_forecasts = client.query_wind_and_solar_forecast(
        country_code, start=start, end=end, psr_type='B16')
    solar_forecasts = client.query_wind_and_solar_forecast(
        country_code, start=start, end=end, psr_type='B19')
    consumption_forecasts = client.query_load_forecast(
        country_code, start=start, end=end)
    
    # Convert to series (keeping MW units)
    wind_forecasts = pd.Series(wind_forecasts.values.flatten(), 
                            index=wind_forecasts.index, name='wind')
    solar_forecasts = pd.Series(solar_forecasts.values.flatten(), 
                             index=solar_forecasts.index, name='solar')
    consumption_forecasts = pd.Series(consumption_forecasts.values.flatten(), 
                             index=consumption_forecasts.index, name='consumption')
    
    # Resample forecasts to hourly intervals and handle duplicates
    wind_forecasts = wind_forecasts[~wind_forecasts.index.duplicated(keep='last')].resample('h').mean()
    solar_forecasts = solar_forecasts[~solar_forecasts.index.duplicated(keep='last')].resample('h').mean()
    consumption_forecasts = consumption_forecasts[~consumption_forecasts.index.duplicated(keep='last')].resample('h').mean()
    # Fetch total load (consumption) data
    print("Fetching consumption data...")
    load_data = client.query_load(country_code, start=start, end=end)
    df_consumption = pd.Series(load_data.values.flatten(), index=load_data.index)
    df_consumption = df_consumption[~df_consumption.index.duplicated(keep='last')].resample('h').mean()

    print("\nMake merged dataset...")
    # Create historical dataset with aligned indices
    historical = pd.DataFrame(index=pd.date_range(start=start, end=end, freq='h', inclusive='left'))
    
    # Add actual values
    historical['wind'] = df_wind.reindex(historical.index)  # In MW
    historical['solar'] = df_solar.reindex(historical.index)  # In MW
    historical['consumption'] = df_consumption.reindex(historical.index)  # In MW
    historical['coal'] = df_kolen.reindex(historical.index)  # In MW

    
    # Add forecasted values
    historical['wind_forecast'] = wind_forecasts.reindex(historical.index)
    historical['solar_forecast'] = solar_forecasts.reindex(historical.index)
    historical['consumption_forecast'] = consumption_forecasts.reindex(historical.index)
    
    # Add price data from CSV
   # Add price data from CSV
    print("Loading price data from CSV...")
    price_data = pd.read_csv('data/raw_prices.csv', index_col=0)
    # First convert to datetime without timezone
    price_data.index = pd.to_datetime(price_data.index, utc=True)
    # Then convert to Amsterdam timezone
    price_data.index = price_data.index.tz_convert('Europe/Amsterdam')
    historical['price_eur_per_mwh'] = price_data['price_eur_per_mwh']


    
    # Check data quality and interpolate
    print("\nChecking for missing data...")
    missing_mask = historical.isna()
    for col in historical.columns:
        missing_dates = historical[missing_mask[col]].index
        if len(missing_dates) > 0:
            print(f"\nMissing {col} data for {len(missing_dates)} hours:")
            print(missing_dates)
            
            # For forecasts (24h gap), use the values from the day before
            if col in ['wind_forecast', 'solar_forecast']:
                # First try to fill with values from exactly 24h before
                historical[col] = historical[col].ffill(limit=24)
                # If any values are still missing, use linear interpolation
                historical[col] = historical[col].interpolate(method='linear', limit=24)
            
            # For prices (single value), use linear interpolation
            elif col == 'price_eur_per_mwh':
                historical[col] = historical[col].interpolate(method='linear')
    
    # Verify all missing values are filled
    print("\nVerifying data completeness after interpolation:")
    print(historical.isna().sum())
    
    # Save historical dataset
    historical.to_csv('data/processed/callibration_dataset_2024_cleaned.csv')
    print("\nHistorical dataset info:")
    print(historical.info())
    print("\nHistorical summary statistics:")
    print(historical.describe())
    print("\nFirst few rows of historical data:")
    print(historical.head())
    


if __name__ == "__main__":
    process_raw_data()
