import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load features (contains train and validation periods)
    features_path = os.path.join(project_root, 'data', 'processed', 'merged_dataset_2023_2024_MW.csv')
    df = pd.read_csv(features_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Define periods
    train_start = pd.Timestamp('2023-01-01', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-01', tz='Europe/Amsterdam')
    validation_start = pd.Timestamp('2024-01-01', tz='Europe/Amsterdam')
    validation_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    
    # Split into train and validation
    train_df = df[(df.index >= train_start) & (df.index < train_end)]
    validation_df = df[(df.index >= validation_start) & (df.index < validation_end)]
    
    # Load test features (March 2024 - May 2024)
    test_features_path = os.path.join(project_root, 'data', 'processed', 'callibration_dataset_2024_cleaned.csv')
    test_df = pd.read_csv(test_features_path, index_col=0)
    test_df.index = pd.to_datetime(test_df.index)
    
    # Split into validation and training
    validation_df = df[(df.index >= validation_start) & (df.index < validation_end)]
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    
    # Select only price column
    validation_df = validation_df[['price_eur_per_mwh']]
    train_df = train_df[['price_eur_per_mwh']]
    test_df = test_df[['price_eur_per_mwh']]
    
    return train_df, validation_df, test_df

def generate_summary_statistics(train_df, validation_df, test_df):
    # Calculate statistics
    stats = pd.DataFrame({
        'Train Count': [len(train_df)],
        'Train Mean': [train_df['price_eur_per_mwh'].mean()],
        'Train Std': [train_df['price_eur_per_mwh'].std()],
        'Train Min': [train_df['price_eur_per_mwh'].min()],
        'Train Max': [train_df['price_eur_per_mwh'].max()],
        'Validation Count': [len(validation_df)],
        'Validation Mean': [validation_df['price_eur_per_mwh'].mean()],
        'Validation Std': [validation_df['price_eur_per_mwh'].std()],
        'Validation Min': [validation_df['price_eur_per_mwh'].min()],
        'Validation Max': [validation_df['price_eur_per_mwh'].max()],
        'Test Count': [len(test_df)],
        'Test Mean': [test_df['price_eur_per_mwh'].mean()],
        'Test Std': [test_df['price_eur_per_mwh'].std()],
        'Test Min': [test_df['price_eur_per_mwh'].min()],
        'Test Max': [test_df['price_eur_per_mwh'].max()]
    })
    
    # Create LaTeX table
    latex_table = "\\begin{table}[htbp]\n\\centering\n"
    latex_table += "\\caption{Summary Statistics of Electricity Prices (EUR/MWh)}\n"
    latex_table += "\\begin{tabular}{lrrr}\n"
    latex_table += "\\hline\n"
    latex_table += "Statistic & Training Set & Validation Set & Test Set \\\\\n"
    latex_table += "\\hline\n"
    latex_table += f"Observations & {stats['Train Count'][0]:.0f} & {stats['Validation Count'][0]:.0f} & {stats['Test Count'][0]:.0f} \\\\\n"
    latex_table += f"Mean & {stats['Train Mean'][0]:.2f} & {stats['Validation Mean'][0]:.2f} & {stats['Test Mean'][0]:.2f} \\\\\n"
    latex_table += f"Standard Deviation & {stats['Train Std'][0]:.2f} & {stats['Validation Std'][0]:.2f} & {stats['Test Std'][0]:.2f} \\\\\n"
    latex_table += f"Minimum & {stats['Train Min'][0]:.2f} & {stats['Validation Min'][0]:.2f} & {stats['Test Min'][0]:.2f} \\\\\n"
    latex_table += f"Maximum & {stats['Train Max'][0]:.2f} & {stats['Validation Max'][0]:.2f} & {stats['Test Max'][0]:.2f} \\\\\n"
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\label{tab:price_summary_stats}\n"
    latex_table += "\\end{table}"
    
    # Save LaTeX table to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'price_summary_statistics.tex')
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table has been saved to: {output_path}")

def plot_price_analysis(train_df, validation_df, test_df):
    """Generate price analysis plots with improved styling"""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Full price series with all periods
    ax1 = plt.subplot(2, 1, 1)
    sns.lineplot(data=train_df, x=train_df.index, y='price_eur_per_mwh', label='Training', color='#2ecc71', alpha=0.8)
    sns.lineplot(data=validation_df, x=validation_df.index, y='price_eur_per_mwh', label='Validation', color='#e84393', alpha=0.8)
    sns.lineplot(data=test_df, x=test_df.index, y='price_eur_per_mwh', label='Testing', color='#e67e22', alpha=0.8)
    
    # Add vertical lines for period splits
    plt.axvline(pd.Timestamp('2024-01-01', tz='Europe/Amsterdam'), color='#e84393', linestyle='--', label='Validation Start', alpha=0.5)
    plt.axvline(pd.Timestamp('2024-03-01', tz='Europe/Amsterdam'), color='#2ecc71', linestyle='--', label='Test Start', alpha=0.5)
    
    plt.title('Electricity Prices: Training, Validation, and Testing Periods', pad=20)
    plt.ylabel('Price (€/MWh)')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    
    # Plot 2: Test period only
    ax2 = plt.subplot(2, 1, 2)
    sns.lineplot(data=test_df, x=test_df.index, y='price_eur_per_mwh', color='#e67e22', alpha=0.8)
    plt.title('Electricity Prices: Test Period (March-May 2024)', pad=20)
    plt.ylabel('Price (€/MWh)')
    
    # Improve layout
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(f'{output_dir}/price_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    train_df, validation_df, test_df = load_data()
    generate_summary_statistics(train_df, validation_df, test_df)
    plot_price_analysis(train_df, validation_df, test_df)

if __name__ == "__main__":
    main()
