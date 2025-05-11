import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
data_path = Path('data/processed/multivariate_features.csv')
df = pd.read_csv(data_path, index_col=0)
df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')

# Create plots directory
Path('models_14_38/xgboost/plots/coal_analysis').mkdir(parents=True, exist_ok=True)

# Basic statistics
print("Coal Distribution Statistics:")
print(df['coal'].describe())

# Value counts for unique values
print("\nTop 10 most common coal values:")
print(df['coal'].value_counts().head(10))

# Distribution plot
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='coal', bins=50)
plt.title('Distribution of Coal Values')
plt.xlabel('Coal Value')
plt.ylabel('Count')
plt.savefig('models_14_38/xgboost/plots/coal_analysis/coal_distribution.png')
plt.close()

# Time series plot
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['coal'])
plt.title('Coal Values Over Time')
plt.xlabel('Date')
plt.ylabel('Coal Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('models_14_38/xgboost/plots/coal_analysis/coal_timeseries.png')
plt.close()

# Daily average
daily_avg = df['coal'].resample('D').mean()
plt.figure(figsize=(15, 6))
plt.plot(daily_avg.index, daily_avg)
plt.title('Daily Average Coal Values')
plt.xlabel('Date')
plt.ylabel('Coal Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('models_14_38/xgboost/plots/coal_analysis/coal_daily_avg.png')
plt.close()

# Check for patterns in values
print("\nUnique values analysis:")
unique_values = df['coal'].unique()
print(f"Number of unique values: {len(unique_values)}")
print(f"Min value: {unique_values.min():.2f}")
print(f"Max value: {unique_values.max():.2f}")
print(f"Step between consecutive values (first 10):")
sorted_values = np.sort(unique_values)
for i in range(min(10, len(sorted_values)-1)):
    print(f"Step {i+1}: {sorted_values[i+1] - sorted_values[i]:.2f}")

# Distribution of differences between consecutive values
coal_diff = df['coal'].diff()
plt.figure(figsize=(12, 6))
sns.histplot(data=coal_diff[coal_diff != 0], bins=50)  # Excluding zero differences
plt.title('Distribution of Changes in Coal Values')
plt.xlabel('Change in Coal Value')
plt.ylabel('Count')
plt.savefig('models_14_38/xgboost/plots/coal_analysis/coal_changes_distribution.png')
plt.close()

# Print summary of changes
print("\nChanges in coal values:")
print(coal_diff[coal_diff != 0].describe())
