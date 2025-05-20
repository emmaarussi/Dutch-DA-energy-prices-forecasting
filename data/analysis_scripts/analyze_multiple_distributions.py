import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import json
from datetime import datetime

# Set style for better visualizations
sns.set_style('whitegrid')

# Create a custom seasonal color palette
seasonal_colors = ['#2ecc71', '#e67e22', '#e84393', '#55efc4', '#fdcb6e', '#fd79a8']
sns.set_palette(seasonal_colors)

# Additional plot styling
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Define paths
DATA_PATH = Path(__file__).parents[2] / "data"
OUTPUT_PATH = DATA_PATH / "analysis_output"
OUTPUT_PATH.mkdir(exist_ok=True)

def load_data():
    """Load the dataset."""
    df = pd.read_csv(DATA_PATH / "processed" / "merged_dataset_2023_2024_MW.csv")
    return df

def analyze_distribution(data):
    """Analyze distribution using sqrt(n) bins."""
    n_datapoints = len(data)
    n_bins = int(np.sqrt(n_datapoints))
    
    # Calculate histogram
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate statistics
    stats = {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "n_datapoints": n_datapoints,
        "n_bins": n_bins,
        "skewness": float(pd.Series(data).skew()),
        "kurtosis": float(pd.Series(data).kurtosis())
    }
    
    return hist, bin_edges, bin_centers, stats

def plot_distributions(df):
    """Create and save distribution plots for multiple variables."""
    variables = {
        'wind': 'Wind Generation (MW)',
        'wind_forecast': 'Wind Generation Forecast (MW)',
        'solar': 'Solar Generation (MW)',
        'solar_forecast': 'Solar Generation Forecast (MW)',
        'consumption': 'Consumption (MW)',
        'coal': 'Coal Generation (MW)',
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    # Set figure background color
    fig.patch.set_facecolor('white')
    
    # Store all statistics
    all_stats = {}
    
    for idx, (var, label) in enumerate(variables.items()):
        if var in df.columns:
            data = df[var].values
            hist, bin_edges, _, stats = analyze_distribution(data)
            
            # Plot histogram
            ax = axes[idx]
            
            # Choose color based on variable type
            if 'wind' in var:
                color = '#2ecc71'  # Green for wind
                alpha = 0.8 if var == 'wind' else 0.5  # Lower alpha for forecast
            elif 'solar' in var:
                color = '#e67e22'  # Orange for solar
                alpha = 0.8 if var == 'solar' else 0.5  # Lower alpha for forecast
            elif var == 'consumption':
                color = '#e84393'  # Pink for consumption
                alpha = 0.7
            else:  # coal
                color = '#55efc4'  # Light green for coal
                alpha = 0.7
                
            ax.hist(bin_edges[:-1], bin_edges, weights=hist, alpha=alpha, 
                    color=color, edgecolor='white')
            
            # Enhanced title and statistics
            title = f'{label}'
            stats_text = f'μ={stats["mean"]:.1f}, σ={stats["std"]:.1f}\n'
            stats_text += f'skew={stats["skewness"]:.2f}, kurt={stats["kurtosis"]:.2f}'
            
            ax.set_title(title, pad=20, fontsize=12)
            ax.text(0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Add mean and median lines
            ax.axvline(stats['mean'], color='#e67e22', linestyle='--', label=f"Mean", alpha=0.8)
            ax.axvline(stats['median'], color='#e84393', linestyle='--', label=f"Median", alpha=0.8)
            ax.legend(frameon=True, fancybox=True, shadow=True)
            
            # Store statistics
            all_stats[var] = stats
    
    # Remove empty subplot if we have odd number of variables
    if len(variables) % 2 == 1:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(OUTPUT_PATH / f"multiple_distributions_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return all_stats

def save_results(stats):
    """Save numerical results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_PATH / f"multiple_distributions_analysis_{timestamp}.json", 'w') as f:
        json.dump(stats, f, indent=4)

def print_summary(stats):
    """Print summary of all distributions."""
    print("\nDistribution Analysis Summary:")
    print("=" * 80)
    
    for var, stat in stats.items():
        print(f"\n{var.upper()}")
        print("-" * 40)
        print(f"Mean: {stat['mean']:.2f}")
        print(f"Median: {stat['median']:.2f}")
        print(f"Standard deviation: {stat['std']:.2f}")
        print(f"Range: [{stat['min']:.2f}, {stat['max']:.2f}]")
        print(f"Skewness: {stat['skewness']:.2f}")
        print(f"Kurtosis: {stat['kurtosis']:.2f}")

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Create plots and get statistics
    print("Analyzing distributions...")
    stats = plot_distributions(df)
    
    # Save numerical results
    print("Saving results...")
    save_results(stats)
    
    # Print summary
    print_summary(stats)
    
    print("\nAnalysis complete! Results saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
