import pandas as pd

# Load the data
df = pd.read_csv('data/processed/multivariate_features.csv', index_col=0)

# Print all column names
print("\nColumns starting with 'price':")
price_cols = [col for col in df.columns if 'price' in col.lower()]
for col in sorted(price_cols):
    print(col)

print("\nFirst few rows of price-related columns:")
print(df[price_cols].head())
