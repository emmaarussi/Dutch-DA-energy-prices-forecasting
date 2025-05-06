"""
Utility functions for model visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss

def check_stationarity(series, title):
    """Perform stationarity tests and create diagnostic plots."""
    print(f"\nStationarity Tests for {title}")
    
    # Augmented Dickey-Fuller test
    adf_result = adfuller(series)
    print('\nAugmented Dickey-Fuller Test:')
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print('Critical values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    # KPSS test
    kpss_result = kpss(series)
    print('\nKPSS Test:')
    print(f'KPSS Statistic: {kpss_result[0]:.4f}')
    print(f'p-value: {kpss_result[1]:.4f}')
    print('Critical values:')
    for key, value in kpss_result[3].items():
        print(f'\t{key}: {value:.4f}')
    
    # Create diagnostic plots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Time series plot
    axes[0].plot(series.index, series.values)
    axes[0].set_title(f'{title} - Time Series Plot')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].grid(True)
    
    # ACF plot (using numpy for autocorrelation)
    lags = 50
    autocorr = [1.] + [np.corrcoef(series[:-i], series[i:])[0,1] for i in range(1, lags+1)]
    axes[1].plot(range(len(autocorr)), autocorr)
    axes[1].set_title('Autocorrelation Function')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Correlation')
    axes[1].grid(True)
    
    plt.tight_layout()
    # Create plots directory if it doesn't exist
    os.makedirs('models_14_38/ar/plots/linear_with_lags', exist_ok=True)
    plt.savefig(f'models_14_38/ar/plots/linear_with_lags/stationarity_{title.lower().replace(" ", "_")}.png')
    plt.close()

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics for model evaluation.
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary containing MAE, RMSE, adapted SMAPE, weighted MAPE, and R² metrics
    """
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Adapted SMAPE (handles zero and negative values better)
    epsilon = 1e-10  # Small constant to avoid division by zero
    abs_diff = np.abs(y_pred - y_true)
    abs_sum = np.abs(y_true) + np.abs(y_pred)
    smape = np.mean(2 * abs_diff / (abs_sum + epsilon)) * 100
    
    # Weighted MAPE (gives more weight to high-price periods)
    weights = np.abs(y_true)
    weights = weights / (weights.sum() + epsilon)  # Normalize weights
    wmape = np.sum(weights * abs_diff / (np.abs(y_true) + epsilon)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'SMAPE': smape,
        'WMAPE': wmape,
        'R2': r2
    }

def plot_feature_importance(importance_df, top_n=20, title='Feature Importance', filename=None):
    """Plot feature importance scores from an XGBoost model.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
        top_n (int, optional): Number of top features to display. Defaults to 20
        title (str, optional): Plot title. Defaults to 'Feature Importance'
        filename (str, optional): Path to save the plot. If None, plot is not saved.
    """
    plt.figure(figsize=(12, 8))
    importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
    
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)

def rolling_window_evaluation_recursive(model, data, window_size='365D', step_size='7D', test_size='7D'):
    """Perform rolling window cross-validation for recursive models."""
    # Convert string periods to timedelta
    window_td = pd.Timedelta(window_size)
    step_td = pd.Timedelta(step_size)
    test_td = pd.Timedelta(test_size)
    
    # Initialize results storage
    all_metrics = {h: [] for h in model.horizons}
    all_predictions = {h: pd.DataFrame() for h in model.horizons}
    
    # Calculate windows
    start_time = data.index.min()
    end_time = data.index.max() - test_td
    
    # Iterate over windows
    current_start = start_time
    window_count = 0
    
    while current_start + window_td <= end_time:
        window_count += 1
        print(f"\nEvaluating window {window_count}...")
        
        # Define window boundaries
        train_start = current_start
        train_end = current_start + window_td
        test_start = train_end
        test_end = test_start + test_td
        
        # Split data
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end]
        
        # Train and evaluate for each horizon
        for horizon in model.horizons:
            print(f"Training horizon t+{horizon}h...")
            
            # Get predictions for this window
            predictions_df = model.predict_for_window(train_data, test_data, horizon)
            all_predictions[horizon] = pd.concat([all_predictions[horizon], predictions_df])
            
            # Calculate metrics
            valid_mask = ~predictions_df['actual'].isna()
            if valid_mask.any():
                metrics = calculate_metrics(
                    predictions_df.loc[valid_mask, 'actual'],
                    predictions_df.loc[valid_mask, 'predicted']
                )
                all_metrics[horizon].append(metrics)
        
        # Move to next window
        current_start += step_td
    
    # Calculate average metrics
    final_metrics = {}
    for horizon in model.horizons:
        horizon_metrics = all_metrics[horizon]
        final_metrics[horizon] = {
            'RMSE': np.mean([m['RMSE'] for m in horizon_metrics]),
            'RMSE_std': np.std([m['RMSE'] for m in horizon_metrics]),
            'SMAPE': np.mean([m['SMAPE'] for m in horizon_metrics]),
            'SMAPE_std': np.std([m['SMAPE'] for m in horizon_metrics]),
            'R2': np.mean([m['R2'] for m in horizon_metrics]),
            'R2_std': np.std([m['R2'] for m in horizon_metrics])
        }
    
    return final_metrics, all_predictions

def rolling_window_evaluation(self, data, window_size='365D', step_size='7D', test_size='7D'):
    """Perform rolling window cross-validation."""
    # Convert string periods to timedelta
    window_td = pd.Timedelta(window_size)
    step_td = pd.Timedelta(step_size)
    test_td = pd.Timedelta(test_size)
    
    # Initialize results storage
    all_metrics = {h: [] for h in self.horizons}
    all_importance = {h: [] for h in self.horizons}
    all_predictions = {h: pd.DataFrame() for h in self.horizons}
    
    # Calculate windows
    start_time = data.index.min()
    end_time = data.index.max() - test_td
    
    # Iterate over windows
    current_start = start_time
    window_count = 0
    
    while current_start + window_td <= end_time:
        window_count += 1
        print(f"\nEvaluating window {window_count}...")
        
        # Define window boundaries
        train_start = current_start
        train_end = current_start + window_td
        test_start = train_end
        test_end = test_start + test_td
        
        # Split data
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end]
        
        # Train and evaluate for each horizon
        for horizon in self.horizons:
            print(f"Training horizon t+{horizon}h...")
            model, metrics, importance = self.train_and_evaluate_window(
                train_data, test_data, horizon
            )
            
            # Store predictions
            X_test, y_test = self.prepare_data(test_data, horizon)
            predictions = model.predict(X_test)
            temp_df = pd.DataFrame({
                'actual': y_test,
                'predicted': predictions
            }, index=test_data.index)
            all_predictions[horizon] = pd.concat([all_predictions[horizon], temp_df])
            
            all_metrics[horizon].append(metrics)
            all_importance[horizon].append(importance)
        
        # Move to next window
        current_start += step_td
    
    # Calculate average metrics
    final_metrics = {}
    for horizon in self.horizons:
        horizon_metrics = all_metrics[horizon]
        final_metrics[horizon] = {
            'RMSE': np.mean([m['RMSE'] for m in horizon_metrics]),
            'RMSE_std': np.std([m['RMSE'] for m in horizon_metrics]),
            'SMAPE': np.mean([m['SMAPE'] for m in horizon_metrics]),
            'SMAPE_std': np.std([m['SMAPE'] for m in horizon_metrics]),
            'R2': np.mean([m['R2'] for m in horizon_metrics]),
            'R2_std': np.std([m['R2'] for m in horizon_metrics])
        }
    
    return final_metrics, all_importance, all_predictions


def plot_predictions(data, y_true, y_pred, horizon, title):
    """Plot actual vs predicted values for a specific horizon."""
    plt.figure(figsize=(15, 6))
    plt.plot(data.index, y_true, label='Actual', alpha=0.7)
    plt.plot(data.index, y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


