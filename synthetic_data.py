import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(start_date=None, end_date=None):
    """
    Generate synthetic energy price data for testing.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        pd.DataFrame: DataFrame with timestamp index and price column
    """
    if start_date is None:
        start_date = '2024-01-01'
    if end_date is None:
        end_date = '2024-12-31'
    
    # Create hourly timestamps
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Generate synthetic prices with daily and seasonal patterns
    n_hours = len(dates)
    
    # Base price around €50/MWh
    base_price = 50
    
    # Add daily pattern (higher during day, lower at night)
    hour_of_day = dates.hour
    daily_pattern = 10 * np.sin(2 * np.pi * (hour_of_day - 12) / 24)
    
    # Add seasonal pattern (higher in winter, lower in summer)
    day_of_year = dates.dayofyear
    seasonal_pattern = 15 * np.sin(2 * np.pi * (day_of_year - 45) / 365)
    
    # Add random noise
    noise = np.random.normal(0, 5, n_hours)
    
    # Combine all components
    prices = base_price + daily_pattern + seasonal_pattern + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': prices
    }, index=dates)
    
    return df
