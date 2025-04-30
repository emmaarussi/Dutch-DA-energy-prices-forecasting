"""
Convert units in the merged dataset:
- Convert historical wind and solar from KW to MW (divide by 1000)
- Remove unused gas column
"""
import pandas as pd
import numpy as np

def convert_units():
    # Load the merged dataset
    print("Loading merged dataset...")
    df = pd.read_csv('data/processed/merged_dataset_2023_2024_cleaned.csv', index_col=0, parse_dates=True)
    
    print("\nBefore conversion:")
    print("Wind mean:", df['wind'].mean())
    print("Solar mean:", df['solar'].mean())
    
    # Convert historical wind and solar from KW to MW
    df['wind'] = df['wind'] / 1000
    df['solar'] = df['solar'] / 1000
    
    print("\nAfter conversion:")
    print("Wind mean:", df['wind'].mean())
    print("Solar mean:", df['solar'].mean())
    
    # Remove gas column since we don't have this data
    df = df.drop('gas', axis=1)
    print("\nRemoved gas column")
    
    # Save the converted dataset
    output_file = 'data/processed/merged_dataset_2023_2024_MW.csv'
    df.to_csv(output_file)
    print(f"\nSaved converted dataset to {output_file}")
    
    return df

if __name__ == "__main__":
    df = convert_units()
    
    # Display summary statistics
    print("\nDataset summary after conversion:")
    print(df.describe())
