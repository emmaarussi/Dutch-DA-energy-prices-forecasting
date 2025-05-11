import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def analyze_feature(data, feature_name):
    """Analyze a single feature for outliers"""
    # Calculate statistics
    Q1 = data[feature_name].quantile(0.25)
    Q3 = data[feature_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = data[(data[feature_name] < lower_bound) | (data[feature_name] > upper_bound)][feature_name]
    
    print(f"\nAnalysis for {feature_name}:")
    print(f"Mean: {data[feature_name].mean():.2f}")
    print(f"Median: {data[feature_name].median():.2f}")
    print(f"Std Dev: {data[feature_name].std():.2f}")
    print(f"Min: {data[feature_name].min():.2f}")
    print(f"Max: {data[feature_name].max():.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage of outliers: {(len(outliers) / len(data)) * 100:.2f}%")
    if len(outliers) > 0:
        print(f"Sample outliers: {outliers.head().values}")
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[feature_name])
    plt.title(f'Distribution of {feature_name}')
    plt.savefig(f'models_14_38/xgboost/plots/outliers/{feature_name}_boxplot.png')
    plt.close()
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=feature_name, bins=50)
    plt.title(f'Histogram of {feature_name}')
    plt.savefig(f'models_14_38/xgboost/plots/outliers/{feature_name}_histogram.png')
    plt.close()

def main():
    # Create plots directory if it doesn't exist
    Path('models_14_38/xgboost/plots/outliers').mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path('data/processed/multivariate_features.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')
    
    # Features to analyze
    base_features = ['wind', 'solar', 'consumption', 'coal']
    
    # Analyze each base feature
    for feature in base_features:
        analyze_feature(df, feature)
        
        # Also analyze its forecast if available
        if f'{feature}_forecast' in df.columns:
            analyze_feature(df, f'{feature}_forecast')
            
            # Create scatter plot to compare actual vs forecast
            plt.figure(figsize=(10, 6))
            plt.scatter(df[feature], df[f'{feature}_forecast'], alpha=0.5)
            plt.xlabel(f'Actual {feature}')
            plt.ylabel(f'Forecast {feature}')
            plt.title(f'{feature}: Actual vs Forecast')
            plt.savefig(f'models_14_38/xgboost/plots/outliers/{feature}_actual_vs_forecast.png')
            plt.close()

if __name__ == "__main__":
    main()
