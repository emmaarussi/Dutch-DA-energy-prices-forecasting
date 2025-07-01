import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pathlib import Path

# --- Load predictions
csv_path = '/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/models_14_38/Linear_with_lags/RFECV_Noexogenous/plots/optimizedretrain/full_predictions.csv'
df = pd.read_csv(csv_path, parse_dates=['target_time', 'forecast_start'])
df = df.sort_values('target_time')
df['residual'] = df['actual'] - df['predicted']

# --- Set output dir
out_dir = Path(csv_path).parent
out_dir.mkdir(parents=True, exist_ok=True)

# --- Analyze residuals by horizon
for h in sorted(df['horizon'].unique()):
    subset = df[df['horizon'] == h].copy()
    residuals = subset['residual'].dropna()

    # ACF plot
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, lags=40, ax=ax)
    ax.set_title(f'Residual ACF - t+{int(h)}h')
    plt.tight_layout()
    plt.savefig(out_dir / f'residual_acf_h{int(h)}.png')
    plt.close()

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    pval = lb_test['lb_pvalue'].iloc[0]
    print(f"t+{h}h → Ljung-Box p-value (lags=10): {pval:.4f}")
    if pval < 0.05:
        print("  ↪️ Residuals show autocorrelation (bad).")
    else:
        print("  ✅ Residuals look like white noise (good).")
