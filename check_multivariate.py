import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv('data/processed/multivariate_features.csv', index_col=0, parse_dates=True)

print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nMissing Values Count:")
missing = df.isnull().sum()
print(missing[missing > 0])  # Only show columns with missing values

print("\nPercentage of Missing Values:")
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0])  # Only show columns with missing values

if df.isnull().any().any():
    print("\nFirst few rows with missing values:")
    print(df[df.isnull().any(axis=1)].head())
    
    print("\nColumns with most missing values:")
    print(missing_pct.sort_values(ascending=False).head())

# Check for infinite values
inf_count = np.isinf(df.select_dtypes(include=np.number)).sum()
if inf_count.any():
    print("\nInfinite Values Count:")
    print(inf_count[inf_count > 0])
