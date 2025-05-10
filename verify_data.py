import pandas as pd

print("Loading historical generation data...")
df_gen = pd.read_csv('data/raw/generation_by_source_2023_2024.csv')

print("\nFirst few rows:")
print(df_gen.head())

print("\nData info:")
print(df_gen.info())

print("\nUnique types:")
print(df_gen['type'].unique())
