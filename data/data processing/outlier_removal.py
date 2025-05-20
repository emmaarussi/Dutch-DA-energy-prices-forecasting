import pandas as pd
import os
import logging
logger = logging.getLogger(__name__)

def remove_outliers(df):
    """Remove extreme and incorrect values using statistical and domain-based approaches."""
    df_clean = df.copy()
    total_outliers = 0

    valid_ranges = {
        'price_eur_per_mwh': (0, 300),  
        'wind': (0, 6000), 
        'solar': (0, 6000), 
        'coal': (600, 700), 
        'consumption': (0, 25000),  
    }

    # Filter relevant columns only
    target_vars = [col for col in df_clean.columns if any(key in col.lower() for key in valid_ranges.keys())]

    # Collect outlier indices per column
    outlier_masks = {}

    for var in target_vars:
        Q1 = df_clean[var].quantile(0.25)
        Q3 = df_clean[var].quantile(0.75)
        IQR = Q3 - Q1

        if 'price_eur_per_mwh' in var.lower():
            threshold = 3.0
        elif any(x in var.lower() for x in ['wind', 'solar', 'consumption']):
            threshold = 2.5
        else:
            threshold = 3.0

        # Initialize column mask
        mask = pd.Series(False, index=df_clean.index)

        # Range-based filtering
        if var in valid_ranges:
            min_val, max_val = valid_ranges[var]
            range_mask = (df_clean[var] < min_val) | (df_clean[var] > max_val)
            n_range = range_mask.sum()
            if n_range > 0:
                logger.info(f"{var}: removed {n_range} values outside range [{min_val}, {max_val}]")
            mask |= range_mask

        # IQR-based filtering
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        iqr_mask = (df_clean[var] < lower) | (df_clean[var] > upper)
        n_iqr = iqr_mask.sum()
        if n_iqr > 0:
            logger.info(f"{var}: removed {n_iqr} statistical outliers")
        mask |= iqr_mask

        # Store mask
        if mask.any():
            outlier_masks[var] = mask
            total_outliers += mask.sum()

    # Apply all masks in one go
    for var, mask in outlier_masks.items():
        df_clean.loc[mask, var] = None

    # Fill missing values
    df_clean = df_clean.ffill().bfill()

    logger.info(f"Total outliers removed: {total_outliers} ({total_outliers/len(df)*100:.2f}% of data)")
    return df_clean


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, 'processed', 'merged_dataset_2023_2024_MW.csv'), index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')

    print("\nremoving outliers...")
    df_nooutliers = remove_outliers(df)

    print("\nSaving datast...")
    df_nooutliers.to_csv(os.path.join(base_dir, 'processed', 'dataset_nooutliers.csv'))


    cols = pd.read_csv(os.path.join(base_dir, 'processed', 'dataset_nooutliers.csv'), nrows=1).columns
    duplicate_targets = [col for col in cols if col.startswith('target_t') and cols.tolist().count(col) > 1]
    print("Duplicate target columns:", duplicate_targets)

if __name__ == "__main__":
    main()