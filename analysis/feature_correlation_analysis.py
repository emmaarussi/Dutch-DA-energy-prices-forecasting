"""
Analysis of feature correlations with electricity prices.
Creates correlation matrices and visualizations to understand feature relationships.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_data():
    """Load the processed features dataset and exclude target features."""
    project_root = Path(__file__).parent.parent
    features_path = project_root / 'data' / 'processed' / 'multivariate_features.csv'
    df = pd.read_csv(features_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Exclude target features (target_t1 through target_t38)
    target_cols = [col for col in df.columns if col.startswith('target_t')]
    df = df.drop(columns=target_cols)
    
    return df

def get_feature_groups():
    """Define feature groups for analysis."""
    return {
        'Base Features': [
            'current_price',
            'wind', 'solar', 'consumption',
            'wind_forecast', 'solar_forecast', 'consumption_forecast'
        ],
        'Time Features': [
            'hour', 'day_of_month', 'month',
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos'
        ],
        'Calendar Features': [
            'is_weekend', 'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday'
        ]
    }

def create_correlation_matrix(df, feature_list, target='current_price'):
    """Create correlation matrix for specified features with price."""
    features_df = df[feature_list]
    correlations = pd.Series({
        feature: features_df[feature].corr(df[target])
        for feature in feature_list
    }).sort_values(ascending=False)
    return correlations

def plot_correlation_heatmap(corr_series, title, figsize=(10, 6)):
    """Plot correlation heatmap for price correlations."""
    plt.figure(figsize=figsize)
    
    # Create heatmap from series
    corr_df = pd.DataFrame(corr_series).T
    
    # Create heatmap
    sns.heatmap(corr_df,
                annot=True,
                cmap='RdBu',
                center=0,
                vmin=-1,
                vmax=1,
                fmt='.2f',
                cbar_kws={'label': 'Correlation with Price'})
    
    plt.title(title)
    plt.ylabel('')  # Remove y-label as we only have one row
    plt.tight_layout()
    return plt.gcf()

def analyze_lagged_correlations(df, sources, lags=[1, 24, 48, 168]):
    """Analyze correlations for lagged features."""
    lag_correlations = {}
    for source in sources:
        source_lags = {}
        for lag in lags:
            col_name = f'{source}_lag_{lag}h'
            if col_name in df.columns:
                corr = df['current_price'].corr(df[col_name])
                source_lags[lag] = corr
        lag_correlations[source] = source_lags
    return pd.DataFrame(lag_correlations)

def main():
    # Load data
    df = load_data()
    feature_groups = get_feature_groups()
    
    # Create output directory
    output_dir = Path(__file__).parent / 'correlation_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Analyze base features
    base_corr = create_correlation_matrix(df, feature_groups['Base Features'])
    base_fig = plot_correlation_heatmap(base_corr, 'Base Feature Correlations with Price')
    base_fig.savefig(output_dir / 'base_features_correlation.png')
    
    # Analyze time features
    time_corr = create_correlation_matrix(df, feature_groups['Time Features'])
    time_fig = plot_correlation_heatmap(time_corr, 'Time Feature Correlations with Price')
    time_fig.savefig(output_dir / 'time_features_correlation.png')
    
    # Analyze calendar features
    calendar_corr = create_correlation_matrix(df, feature_groups['Calendar Features'])
    calendar_fig = plot_correlation_heatmap(calendar_corr, 'Calendar Feature Correlations with Price')
    calendar_fig.savefig(output_dir / 'calendar_features_correlation.png')
    
    # Analyze lagged correlations
    sources = ['wind', 'solar', 'consumption', 'current_price']
    lag_corr = analyze_lagged_correlations(df, sources)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(lag_corr, 
                annot=True, 
                cmap='RdBu',
                center=0,
                vmin=-1,
                vmax=1,
                fmt='.2f')
    plt.title('Lagged Feature Correlations with Price')
    plt.tight_layout()
    plt.savefig(output_dir / 'lagged_features_correlation.png')
    
    # Print strongest correlations
    print("\nStrongest correlations with price:")
    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)
    all_corr = create_correlation_matrix(df, all_features)
    print(all_corr.head(10))

if __name__ == "__main__":
    main()
