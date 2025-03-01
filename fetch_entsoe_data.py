from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_day_ahead_prices(start_date, end_date, country_code='NL'):
    """
    Fetch day-ahead prices from ENTSO-E for a specific country and time period.
    
    Args:
        start_date (datetime): Start date for data fetching (timezone-aware)
        end_date (datetime): End date for data fetching (timezone-aware)
        country_code (str): Country code (default: 'NL' for Netherlands)
    
    Returns:
        pandas.DataFrame: Day-ahead prices
    """
    try:
        # Get API key from environment variable
        api_key = os.getenv('ENTSOE_API_KEY')
        if not api_key:
            raise ValueError("ENTSOE_API_KEY not found in environment variables")

        # Initialize client
        client = EntsoePandasClient(api_key=api_key)

        # Make sure we have timezone-aware datetime objects
        if start_date.tzinfo is None or end_date.tzinfo is None:
            raise ValueError("start_date and end_date must be timezone-aware")

        # Convert to pandas Timestamp
        start_pd = pd.Timestamp(start_date)
        end_pd = pd.Timestamp(end_date)

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Fetch day-ahead prices
        prices = client.query_day_ahead_prices(
            country_code=country_code,
            start=start_pd,
            end=end_pd
        )
        
        # prices is already a pandas Series with DateTimeIndex
        prices_df = pd.DataFrame({
            'timestamp': prices.index,
            'price_eur_per_mwh': prices.values,
            'price_eur_per_kwh': prices.values / 1000
        })
        
        # Set timestamp as index
        prices_df.set_index('timestamp', inplace=True)
        
        # Save to CSV
        output_path = 'data/raw_prices.csv'
        prices_df.to_csv(output_path)
        
        return prices_df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage:
    # Fetch last 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    prices = fetch_day_ahead_prices(start_date, end_date)
    if prices is not None:
        print(f"Successfully fetched {len(prices)} price points")
