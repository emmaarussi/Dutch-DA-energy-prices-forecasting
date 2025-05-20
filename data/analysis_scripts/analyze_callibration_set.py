import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
import os

file_path = '/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/data/processed/callibration_dataset_2024_cleaned.csv'

if not os.path.exists(file_path):
    print(f"Error: File does not exist at path: {file_path}")
    print("\nAvailable files in directory:")
    dir_path = os.path.dirname(file_path)
    if os.path.exists(dir_path):
        print("\n".join(os.listdir(dir_path)))
    exit(1)

try:
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
except Exception as e:
    print(f"Error reading file: {str(e)}")
    print(f"File path: {file_path}")
    print(f"Current working directory: {os.getcwd()}")
    exit(1)

# Calculate basic statistics
print("\nBasic Statistics:")
print("================")
stats = df.describe()
print(stats)

# Identify outliers using IQR method


try:
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
except Exception as e:
    print(f"Error reading file: {str(e)}")
    print(f"File path: {file_path}")
    print(f"Current working directory: {os.getcwd()}")
    exit(1)

# Calculate basic statistics
print("\nBasic Statistics:")
print("================")
stats = df.describe()
print(stats)

# Identify outliers using IQR method
def find_outliers(df):
    outliers = {}
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers[column] = {
            'count': df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0],
            'percentage': (df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0] / df.shape[0]) * 100,
            'min': df[column].min(),
            'max': df[column].max()
        }
    return outliers

print("\nOutlier Analysis:")
print("================")
outliers = find_outliers(df)
for column, stats in outliers.items():
    print(f"\n{column}:")
    print(f"Number of outliers: {stats['count']}")
    print(f"Percentage of outliers: {stats['percentage']:.2f}%")
    print(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]")


# Plot prices over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['price_eur_per_mwh'], linewidth=1)
plt.title('Electricity Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (EUR/MWh)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/data/analysis_output/price_over_time_callibration.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results to file
with open('/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/data/analysis_output/statistics_summary.txt', 'w') as f:
    f.write("Basic Statistics:\n")
    f.write("================\n")
    f.write(stats.to_string())
    f.write("\n\nOutlier Analysis:\n")
    f.write("================\n")
    for column, stats in outliers.items():
        f.write(f"\n{column}:\n")
        f.write(f"Number of outliers: {stats['count']}\n")
        f.write(f"Percentage of outliers: {stats['percentage']:.2f}%\n")
        f.write(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n")

print("\nResults have been saved to 'statistics_summary.txt'")