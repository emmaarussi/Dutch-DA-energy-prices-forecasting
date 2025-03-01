"""
Analyze energy generation data from Nederlandse Energie Dashboard (NED).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_data():
    """Load and preprocess generation data."""
    path = os.path.join('data', 'generation_by_source.csv')
    df = pd.read_csv(path, parse_dates=['validfrom', 'validto'])
    
    # Map source types
    source_mapping = {
        0: 'total_generation',
        1: 'wind',
        2: 'solar',
        17: 'wind_offshore'
    }
    
    # Filter for sources we're interested in
    df = df[df['type'].isin(source_mapping.keys())]
    
    # Map type numbers to source names
    df['source'] = df['type'].map(source_mapping)
    
    # Group by timestamp and source, taking the mean of duplicate entries
    df_grouped = df.groupby(['validfrom', 'source'])['capacity'].mean().reset_index()
    
    # Pivot the data to get sources as columns
    df_pivot = df_grouped.pivot(index='validfrom', columns='source', values='capacity')
    
    # Resample to hourly frequency and forward fill missing values
    df_hourly = df_pivot.resample('H').mean().ffill()
    
    return df_hourly

def plot_generation_overview(df):
    """Plot overview of generation from different sources."""
    plt.figure(figsize=(15, 8))
    
    # Plot renewable sources
    renewable_sources = ['wind_onshore', 'solar', 'wind_offshore']
    for source in renewable_sources:
        if source in df.columns:
            plt.plot(df.index, df[source], label=source, alpha=0.7)
    
    # Plot total generation and type classifications
    other_sources = ['total_generation', 'fossil', 'renewable']
    for source in other_sources:
        if source in df.columns:
            plt.plot(df.index, df[source], label=source, linestyle='--', alpha=0.5)
    
    plt.title('Energy Generation by Source')
    plt.xlabel('Date')
    plt.ylabel('Generation (MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('data', 'generation_overview.png'))
    plt.close()

def plot_daily_patterns(df):
    """Plot average daily generation patterns."""
    df = df.copy()
    df['hour'] = df.index.hour
    
    # Select only renewable generation sources
    generation_sources = ['wind_onshore', 'solar', 'wind_offshore']
    available_sources = [s for s in generation_sources if s in df.columns]
    
    fig, axes = plt.subplots(len(available_sources), 1, figsize=(12, 15))
    
    for idx, source in enumerate(available_sources):
        sns.boxplot(data=df, x='hour', y=source, ax=axes[idx])
        axes[idx].set_title(f'Daily Pattern: {source}')
        axes[idx].set_xlabel('Hour of Day')
        axes[idx].set_ylabel('Generation (MW)')
    
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'generation_daily_patterns.png'))
    plt.close()

def plot_seasonal_patterns(df):
    """Plot seasonal generation patterns."""
    df = df.copy()
    df['month'] = df.index.month
    
    # Select only renewable generation sources
    generation_sources = ['wind_onshore', 'solar', 'wind_offshore']
    available_sources = [s for s in generation_sources if s in df.columns]
    
    fig, axes = plt.subplots(len(available_sources), 1, figsize=(12, 15))
    
    for idx, source in enumerate(available_sources):
        sns.boxplot(data=df, x='month', y=source, ax=axes[idx])
        axes[idx].set_title(f'Seasonal Pattern: {source}')
        axes[idx].set_xlabel('Month')
        axes[idx].set_ylabel('Generation (MW)')
    
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'generation_seasonal_patterns.png'))
    plt.close()

def calculate_statistics(df):
    """Calculate key statistics for each source."""
    stats = pd.DataFrame()
    
    for column in df.columns:
        if column not in ['hour', 'month']:
            stats[column] = pd.Series({
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'median': df[column].median(),
                'missing_values': df[column].isna().sum()
            })
    
    return stats

def plot_correlation_with_price(gen_df):
    """Plot correlation between generation and price."""
    # Load price data
    price_path = os.path.join('data', 'raw_prices.csv')
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    
    # Merge data
    merged_df = gen_df.join(price_df['price_eur_per_mwh'])
    
    # Calculate correlations
    corr = merged_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between Generation Sources and Price')
    plt.savefig(os.path.join('data', 'generation_price_correlation.png'))
    plt.close()
    
    return corr

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Generate plots
    print("\nGenerating plots...")
    plot_generation_overview(df)
    plot_daily_patterns(df)
    plot_seasonal_patterns(df)
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(df)
    print("\nGeneration Statistics:")
    print(stats)
    
    # Calculate and plot correlations with price
    print("\nAnalyzing correlation with price...")
    corr = plot_correlation_with_price(df)
    print("\nCorrelation with price:")
    print(corr['price_eur_per_mwh'])

if __name__ == "__main__":
    main()
