import pandas as pd
import numpy as np
from scipy import stats

def diebold_mariano_test(e1, e2, h=1, power=2):
    """
    Diebold-Mariano test for equal predictive accuracy.
    e1 and e2 are forecast errors (e.g., squared or absolute errors).
    h = forecast horizon (default 1), power = 1 (MAE) or 2 (MSE).
    """
    d = (np.abs(e1) ** power - np.abs(e2) ** power)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    n = len(d)
    
    # Small-sample adjustment (Harvey et al., 1997)
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = d_mean / (np.sqrt(d_var / n)) * adj
    p_value = 2 * stats.norm.cdf(-abs(dm_stat))
    return dm_stat, p_value

# --- Load predictions
ar_df = pd.read_csv("/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/models_14_38/ar/plots/ARonlypointeachhour/full_predictions.csv", parse_dates=['target_time'])
lr_df = pd.read_csv("/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/models_14_38/Linear_with_lags/RFECV_Noexogenous/plots/optimizedretrain/full_predictions.csv", parse_dates=['target_time'])
xgb_df = pd.read_csv("/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/models_14_38/xgboost/plots/optimizedretrain/full_predictions.csv", parse_dates=['target_time'])
xgb_noretrain_df = pd.read_csv("/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/models_14_38/xgboost/plots/optimized_noretrain/full_predictions.csv", parse_dates=['target_time'])

for df in [ar_df, lr_df, xgb_df, xgb_noretrain_df]:
    df['target_time'] = pd.to_datetime(df['target_time'], utc=True)

# --- Merge data
merged = ar_df[['target_time', 'horizon', 'actual', 'predicted']].rename(columns={'predicted': 'ar_pred'})
merged = merged.merge(
    lr_df[['target_time', 'horizon', 'predicted']].rename(columns={'predicted': 'lr_pred'}),
    on=['target_time', 'horizon']
)
merged = merged.merge(
    xgb_df[['target_time', 'horizon', 'predicted']].rename(columns={'predicted': 'xgb_pred'}),
    on=['target_time', 'horizon']
)
merged = merged.merge(
    xgb_noretrain_df[['target_time', 'horizon', 'predicted']].rename(columns={'predicted': 'xgb_noretrain_pred'}),
    on=['target_time', 'horizon']
)

# --- Compute squared errors
merged['ar_error'] = (merged['actual'] - merged['ar_pred'])**2
merged['lr_error'] = (merged['actual'] - merged['lr_pred'])**2
merged['xgb_error'] = (merged['actual'] - merged['xgb_pred'])**2
merged['xgb_noretrain_error'] = (merged['actual'] - merged['xgb_noretrain_pred'])**2

# --- Run Diebold-Mariano tests
print("\n📊 Diebold-Mariano Test Results (Squared Error Loss):")

stat_ar_lr, pval_ar_lr = diebold_mariano_test(merged['ar_error'], merged['lr_error'])
print(f"AR vs Linear:      DM Stat = {stat_ar_lr:.4f}, p-value = {pval_ar_lr:.4f}")

stat_ar_xgb, pval_ar_xgb = diebold_mariano_test(merged['ar_error'], merged['xgb_error'])
print(f"AR vs XGBoost:     DM Stat = {stat_ar_xgb:.4f}, p-value = {pval_ar_xgb:.4f}")

"""XGboost vs XGBoost (no retrain)"""
stat_xgb_noretrain_xgb, pval_xgb_noretrain_xgb = diebold_mariano_test(merged['xgb_error'], merged['xgb_noretrain_error'])
print(f"XGBoost vs XGBoost (no retrain): DM Stat = {stat_xgb_noretrain_xgb:.4f}, p-value = {pval_xgb_noretrain_xgb:.4f}")

stat_lr_xgb, pval_lr_xgb = diebold_mariano_test(merged['lr_error'], merged['xgb_error'])
print(f"Linear vs XGBoost: DM Stat = {stat_lr_xgb:.4f}, p-value = {pval_lr_xgb:.4f}")
