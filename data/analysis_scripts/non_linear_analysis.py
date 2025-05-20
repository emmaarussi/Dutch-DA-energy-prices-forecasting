import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s
from pathlib import Path

# Set up paths
BASE_PATH = Path(__file__).parents[2]
DATA_PATH = BASE_PATH / "data" / "processed" / "merged_dataset_2023_2024_MW.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Drop NA values to avoid fit issues
df = df.dropna()

# Define covariates and target
covariates = [
    'wind', 'solar', 'consumption', 'coal',
    'wind_forecast', 'solar_forecast', 'consumption_forecast'
]
target = 'price_eur_per_mwh'

# Set up paths
BASE_PATH = Path(__file__).parents[2]
OUTPUT_PATH = BASE_PATH / "data" / "analysis_output" / "gam_plots"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 8

# Create 4x4 subplot figure
plt.figure(figsize=(20, 20))

# Track plot position
plot_num = 1

# Loop through covariates
for col in covariates:
    print(f"Fitting GAM for {col}...")

    # Create subplot
    plt.subplot(4, 4, plot_num)
    
    # Prepare data
    X = df[[col]].values
    y = df[target].values

    # Fit GAM model
    gam = LinearGAM(s(0)).fit(X, y)

    # Create smooth curve
    XX = np.linspace(X.min(), X.max(), 100)
    y_pred = gam.predict(XX)
    
    # Choose color based on variable
    if 'wind' in col:
        color = '#2ecc71'  # Green
    elif 'solar' in col:
        color = '#e67e22'  # Orange
    elif 'consumption' in col:
        color = '#e84393'  # Pink
    else:  # coal
        color = '#55efc4'  # Light green
    
    # Plot data and fit
    plt.scatter(X, y, alpha=0.2, color=color, label="Observed", s=10)
    plt.plot(XX, y_pred, color=color, label="GAM Fit", linewidth=2)
    
    # Add confidence intervals
    intervals = gam.prediction_intervals(XX, width=0.95)
    plt.fill_between(XX.ravel(), intervals[:, 0], intervals[:, 1],
                    color=color, alpha=0.2)
    
    # Add R² score
    r2 = gam.score(X, y)
    plt.text(0.02, 0.95, f'R² = {r2:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
    
    plt.title(f"{col} vs Price", pad=10)
    plt.xlabel(f"{col} (MW)")
    plt.ylabel("Price (€/MWh)")
    plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Increment plot position
    plot_num += 1

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig(OUTPUT_PATH / "all_gam_plots.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ GAM plots saved to: {OUTPUT_PATH}")

