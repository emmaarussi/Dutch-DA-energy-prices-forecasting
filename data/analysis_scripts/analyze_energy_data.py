import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from datetime import datetime
import os

# Set style for all plots
sns.set_style('whitegrid')
seasonal_colors = ['#2ecc71', '#e67e22', '#e84393', '#55efc4', '#fdcb6e', '#fd79a8']
sns.set_palette(seasonal_colors)

# Additional plot styling
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Create output directory for plots and tables
output_dir = 'data/analysis_output'
os.makedirs(output_dir, exist_ok=True)

# Load the data
print('Loading data...')
historical = pd.read_csv('data/processed/merged_dataset_2023_2024_MW.csv', index_col=0)
historical.index = pd.to_datetime(historical.index, utc=True).tz_convert('Europe/Amsterdam')


# Define train-test split date
split_date = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')

# Separate train and test data
train = historical[historical.index < split_date]
test = historical[historical.index >= split_date]

def plot_price_analysis():
    """Generate price analysis plots with improved styling"""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Full price series with train/test split
    ax1 = plt.subplot(2, 1, 1)
    sns.lineplot(data=train, x=train.index, y='price_eur_per_mwh', label='Training', color='#2ecc71', alpha=0.8)
    sns.lineplot(data=test, x=test.index, y='price_eur_per_mwh', label='Testing', color='#e67e22', alpha=0.8)
    plt.axvline(pd.to_datetime(split_date), color='#e84393', linestyle='--', label='Split', alpha=0.5)
    plt.title('Electricity Prices: Training vs Testing Periods', pad=20)
    plt.ylabel('Price (€/MWh)')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    
    # Plot 2: Test period only
    ax2 = plt.subplot(2, 1, 2)
    sns.lineplot(data=test, x=test.index, y='price_eur_per_mwh', color='#e67e22', alpha=0.8)
    plt.title('Electricity Prices: Test Period', pad=20)
    plt.ylabel('Price (€/MWh)')
    
    # Improve layout
    plt.tight_layout()
    plt.savefig(f'{output_dir}/price_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistics():
    """Generate statistical analysis for historical and forecast data"""
    # Historical data statistics
    hist_stats = historical[['price_eur_per_mwh', 'wind', 'solar', 'consumption']].describe()
    hist_stats.to_latex(f'{output_dir}/historical_statistics.tex')
    
    # Forecast data statistics
    forecast_stats = historical[['wind_forecast', 'solar_forecast', 'consumption_forecast']].describe()
    forecast_stats.columns = ['wind', 'solar', 'consumption']  # Rename for consistency
    forecast_stats.to_latex(f'{output_dir}/forecast_statistics.tex')
    
    return hist_stats, forecast_stats

def stationarity_analysis():
    """Perform stationarity analysis on all variables"""
    variables = ['price_eur_per_mwh', 'wind', 'solar', 'consumption']
    results = {}
    
    for var in variables:
        adf_result = adfuller(historical[var].dropna())
        results[var] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical values': adf_result[4]
        }
    
    # Convert to DataFrame and save
    df_results = pd.DataFrame(results).round(4)
    df_results.to_latex(f'{output_dir}/stationarity_analysis.tex')
    return df_results

def plot_temporal_analysis(variable):
    """Generate temporal analysis plots for a given variable"""
    data = historical.copy()
    data['hour'] = data.index.hour
    data['day'] = data.index.day
    data['month'] = data.index.month
    data['dayofweek'] = data.index.dayofweek
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Daily pattern
    sns.boxplot(x='hour', y=variable, data=data, ax=axes[0,0])
    axes[0,0].set_title(f'Daily Pattern - {variable}')
    
    # Weekly pattern
    sns.boxplot(x='dayofweek', y=variable, data=data, ax=axes[0,1])
    axes[0,1].set_title(f'Weekly Pattern - {variable}')
    
    # Monthly pattern
    sns.boxplot(x='month', y=variable, data=data, ax=axes[1,0])
    axes[1,0].set_title(f'Monthly Pattern - {variable}')
    
    # Trend
    data.groupby('month')[variable].mean().plot(ax=axes[1,1])
    axes[1,1].set_title(f'Monthly Trend - {variable}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_analysis_{variable}.png')
    plt.close()

def price_volatility_analysis():
    """Analyze price volatility"""
    # Calculate daily volatility
    daily_vol = historical.groupby(historical.index.date)['price_eur_per_mwh'].std()
    
    plt.figure(figsize=(15, 6))
    daily_vol.plot()
    plt.title('Daily Price Volatility')
    plt.ylabel('Standard Deviation of Price')
    plt.savefig(f'{output_dir}/price_volatility.png')
    plt.close()
    
    # Save volatility statistics
    vol_stats = daily_vol.describe().round(2)
    vol_stats.to_latex(f'{output_dir}/volatility_statistics.tex')
    return vol_stats

def seasonal_analysis():
    """Perform seasonal decomposition analysis"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Resample to daily for clearer seasonal patterns
    daily_prices = historical['price_eur_per_mwh'].resample('D').mean()
    
    # Perform decomposition
    decomposition = seasonal_decompose(daily_prices, period=30)
    
    # Plot decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/seasonal_decomposition.png')
    plt.close()

def forecast_accuracy():
    """Analyze forecast accuracy"""
    variables = ['wind', 'solar', 'consumption']
    metrics = {}
    
    for var in variables:
        actual = historical[var]
        predicted = historical[f'{var}_forecast']
        
        # Remove any rows where actual is 0 to avoid division by zero in MAPE
        mask = actual != 0
        actual_filtered = actual[mask]
        predicted_filtered = predicted[mask]
        
        rmse = np.sqrt(((actual - predicted) ** 2).mean())
        mae = np.abs(actual - predicted).mean()
        mape = (np.abs((actual_filtered - predicted_filtered) / actual_filtered) * 100).mean()
        
        metrics[var] = {
            'RMSE (MW)': rmse,
            'MAE (MW)': mae,
            'MAPE (%)': mape
        }
    
    # Convert to DataFrame and save
    df_metrics = pd.DataFrame(metrics).round(2)
    df_metrics.to_latex(f'{output_dir}/forecast_accuracy.tex')
    
    # Also save a more detailed analysis
    with open(f'{output_dir}/forecast_accuracy_details.txt', 'w') as f:
        for var in variables:
            f.write(f'\n{var.upper()} FORECAST ANALYSIS:\n')
            f.write(f'Mean actual value: {historical[var].mean():.2f} MW\n')
            f.write(f'Mean forecast value: {historical[f"{var}_forecast"].mean():.2f} MW\n')
            f.write(f'Correlation: {historical[var].corr(historical[f"{var}_forecast"]):.3f}\n')
            f.write('---\n')
    
    return df_metrics

def plot_forecast_comparison():
    """Plot forecasts vs historical values for wind, solar, and consumption"""
    variables = ['wind', 'solar', 'consumption']
    
    # Define colors for each variable
    colors = {
        'wind': '#2ecc71',    # Green
        'solar': '#e67e22',    # Orange
        'consumption': '#e84393'  # Pink
    }
    
    plt.figure(figsize=(15, 12))
    for i, var in enumerate(variables, 1):
        plt.subplot(3, 1, i)
        
        # Plot historical and forecast with same color but different alpha
        color = colors[var]
        # Plot forecast first (so historical is on top)
        forecast_color = '#e84393' if var == 'solar' else color  # Use pink for solar forecast
        plt.plot(historical.index, historical[f'{var}_forecast'], 
                 label='Forecast', color=forecast_color, alpha=0.7, linewidth=2.5, linestyle='--')
        plt.plot(historical.index, historical[var], 
                 label='Historical', color=color, alpha=0.8, linewidth=2)
        
        plt.title(f'{var.title()} - Historical vs Forecast', pad=20, fontsize=12)
        plt.ylabel('MW', fontsize=10)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add text with correlation and RMSE
        corr = historical[var].corr(historical[f'{var}_forecast'])
        rmse = np.sqrt(((historical[var] - historical[f'{var}_forecast']) ** 2).mean())
        plt.text(0.02, 0.95, f'Correlation: {corr:.3f}\nRMSE: {rmse:.1f} MW', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecast_comparison.png')
    plt.close()

def plot_renewable_variability():
    """Analyze and visualize the variability of wind and solar generation"""
    variables = ['wind', 'solar']
    
    # Create figure with 3 rows (volatility, day changes, daily patterns) and 2 columns (wind, solar)
    fig = plt.figure(figsize=(15, 18))
    
    for col, var in enumerate(variables):
        # 1. Daily volatility
        ax1 = plt.subplot(3, 2, col+1)
        daily_vol = historical[var].groupby(historical.index.date).std()
        daily_vol.plot(ax=ax1)
        ax1.set_title(f'Daily {var.title()} Volatility')
        ax1.set_ylabel('Standard Deviation (MW)')
        
        # 2. Day-to-day changes
        ax2 = plt.subplot(3, 2, col+3)
        daily_mean = historical[var].groupby(historical.index.date).mean()
        daily_changes = daily_mean.diff()
        daily_changes.plot(ax=ax2)
        ax2.set_title(f'Day-to-Day Changes in {var.title()}')
        ax2.set_ylabel('Change in Mean Generation (MW)')
        
        # Add statistics to the plot
        stats_text = f'Mean daily change: {daily_changes.mean():.0f} MW\n'
        stats_text += f'Std of daily changes: {daily_changes.std():.0f} MW\n'
        stats_text += f'Max increase: {daily_changes.max():.0f} MW\n'
        stats_text += f'Max decrease: {daily_changes.min():.0f} MW'
        ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), va='top')
        
        # 3. Daily patterns (spaghetti plot)
        ax3 = plt.subplot(3, 2, col+5)
        
        # Get data for the last 30 days
        end_date = historical.index.max()
        start_date = end_date - pd.Timedelta(days=30)
        last_30_days = historical[historical.index >= start_date]
        
        # Get unique dates
        unique_dates = pd.Series(last_30_days.index.date).unique()
        
        # Plot each day as a separate line
        for date in unique_dates:
            day_data = last_30_days[last_30_days.index.date == date][var]
            hours = pd.Series(day_data.index.hour)
            ax3.plot(hours, day_data.values, alpha=0.2, color='gray')
        
        # Plot the mean pattern
        hourly_mean = last_30_days.groupby(last_30_days.index.hour)[var].mean()
        ax3.plot(range(24), hourly_mean.values, color='red', linewidth=2, label='Mean Pattern')
        
        ax3.set_title(f'Daily {var.title()} Patterns (Last 30 Days)')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Generation (MW)')
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/renewable_variability.png')
    plt.close()

def plot_correlation_matrix():
    """Create and save correlation matrix plot focusing on price relationships."""
    # Select relevant variables
    variables = ['price_eur_per_mwh', 'wind', 'solar', 'consumption', 'coal',
                'wind_forecast', 'solar_forecast', 'consumption_forecast']
    
    # Calculate correlation matrix
    corr_matrix = historical[variables].corr()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create mask for upper triangle
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Create custom colormap from seasonal colors
    seasonal_green = '#2ecc71'  # Our nice green
    seasonal_orange = '#e67e22'  # Our warm orange
    seasonal_white = '#ffffff'  # White for center
    
    # Create custom colormap
    colors = [seasonal_green, seasonal_white, seasonal_orange]
    n_bins = 256  # Number of color gradations
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot correlation matrix
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap=cmap,
                vmin=-1, vmax=1, center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                annot=True,
                fmt='.2f')
    
    plt.title('Correlation Matrix with Price', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def main():
    print('Generating plots and analyses...')
    
    # 1. Price Analysis
    plot_price_analysis()
    
    # 2. Statistical Analysis
    hist_stats, forecast_stats = generate_statistics()
    
    # 3. Stationarity Analysis
    stationarity_results = stationarity_analysis()
    
    # 4. Temporal Analysis
    for variable in ['price_eur_per_mwh', 'wind', 'solar', 'consumption']:
        plot_temporal_analysis(variable)
    
    # 5. Price Volatility
    volatility_stats = price_volatility_analysis()
    
    # 6. Seasonal Analysis
    seasonal_analysis()
    
    # 7. Forecast Accuracy
    forecast_metrics = forecast_accuracy()
    
    # 8. Forecast Comparison Plots
    plot_forecast_comparison()
    
    # 9. Renewable Variability Analysis
    plot_renewable_variability()
    
    # 10. Correlation Analysis
    correlation_matrix = plot_correlation_matrix()
    
    print('\nCorrelation with price:')
    price_correlations = correlation_matrix['price_eur_per_mwh'].sort_values(ascending=False)
    for var, corr in price_correlations.items():
        if var != 'price_eur_per_mwh':
            print(f'{var:25} {corr:6.3f}')
    
    print('\nAnalysis complete! Results saved in:', output_dir)

if __name__ == '__main__':
    main()
