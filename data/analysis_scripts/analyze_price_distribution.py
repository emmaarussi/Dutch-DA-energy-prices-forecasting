import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import json
from datetime import datetime
from scipy import stats
from scipy.stats import probplot

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
    """Load the price data from the processed dataset."""
    df = pd.read_csv(DATA_PATH / "processed" / "merged_dataset_2023_2024_cleaned.csv")
    return df

def analyze_price_distribution(df):
    """Analyze price distribution using sqrt(n) bins."""
    prices = df['price_eur_per_mwh'].values
    n_datapoints = len(prices)
    n_bins = int(np.sqrt(n_datapoints))  # Square root rule for number of bins
    
    # Calculate histogram
    hist, bin_edges = np.histogram(prices, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Perform normality tests
    shapiro_stat, shapiro_p = stats.shapiro(prices)
    ks_stat, ks_p = stats.kstest(prices, 'norm', args=(np.mean(prices), np.std(prices)))
    
    # Calculate basic statistics
    stats_dict = {
        "mean": float(np.mean(prices)),
        "median": float(np.median(prices)),
        "std": float(np.std(prices)),
        "min": float(np.min(prices)),
        "max": float(np.max(prices)),
        "n_datapoints": n_datapoints,
        "n_bins": n_bins,
        "skewness": float(pd.Series(prices).skew()),
        "kurtosis": float(pd.Series(prices).kurtosis()),
        "shapiro_test": {
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_p)
        },
        "ks_test": {
            "statistic": float(ks_stat),
            "p_value": float(ks_p)
        }
    }
    
    return hist, bin_edges, bin_centers, stats_dict

def plot_distribution(hist, bin_edges, stats, df):
    """Create and save distribution plots."""
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 15))
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Regular histogram
    ax1.hist(bin_edges[:-1], bin_edges, weights=hist, alpha=0.7, color='#2ecc71', edgecolor='white')
    ax1.set_title('Price Distribution', pad=20)
    ax1.set_xlabel('Price (€/MWh)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(stats['mean'], color='#e67e22', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
    ax1.axvline(stats['median'], color='#e84393', linestyle='--', label=f"Median: {stats['median']:.2f}")
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    
    # Log-scale histogram
    ax2.hist(bin_edges[:-1], bin_edges, weights=hist, alpha=0.7, color='#2ecc71', edgecolor='white')
    ax2.set_yscale('log')
    ax2.set_title('Price Distribution (Log Scale)', pad=20)
    ax2.set_xlabel('Price (€/MWh)')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.axvline(stats['mean'], color='#e67e22', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
    ax2.axvline(stats['median'], color='#e84393', linestyle='--', label=f"Median: {stats['median']:.2f}")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add QQ plot
    probplot(df['price_eur_per_mwh'].values, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot vs Normal Distribution', pad=20)
    ax3.get_lines()[0].set_color('#2ecc71')  # Data points
    ax3.get_lines()[1].set_color('#e67e22')  # Reference line
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(OUTPUT_PATH / f"price_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_results(hist, bin_edges, stats):
    """Save numerical results to JSON file."""
    results = {
        "statistics": stats,
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_PATH / f"price_distribution_analysis_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Analyze distribution
    print("Analyzing price distribution...")
    hist, bin_edges, bin_centers, stats = analyze_price_distribution(df)
    
    # Create and save plots
    print("Creating plots...")
    plot_distribution(hist, bin_edges, stats, df)
    
    # Save numerical results
    print("Saving results...")
    save_results(hist, bin_edges, stats)
    
    print("\nAnalysis complete! Results saved to:", OUTPUT_PATH)
    print(f"\nKey statistics:")
    print(f"Number of data points: {stats['n_datapoints']}")
    print(f"Number of bins: {stats['n_bins']}")
    print(f"Mean price: {stats['mean']:.2f} €/MWh")
    print(f"Median price: {stats['median']:.2f} €/MWh")
    print(f"Standard deviation: {stats['std']:.2f}")
    print(f"Price range: [{stats['min']:.2f}, {stats['max']:.2f}] €/MWh")
    print(f"Skewness: {stats['skewness']:.2f}")
    print(f"Kurtosis: {stats['kurtosis']:.2f}")
    print(f"\nNormality Tests:")
    print(f"Shapiro-Wilk test p-value: {stats['shapiro_test']['p_value']:.2e}")
    print(f"Kolmogorov-Smirnov test p-value: {stats['ks_test']['p_value']:.2e}")

if __name__ == "__main__":
    main()
