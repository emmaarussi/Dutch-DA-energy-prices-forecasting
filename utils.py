import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics for model evaluation.
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary containing MAE, RMSE, MAPE, and R² metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def plot_feature_importance(importance_df, top_n=20, title='Feature Importance'):
    """Plot feature importance scores from an XGBoost model.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
        top_n (int, optional): Number of top features to display. Defaults to 20
        title (str, optional): Plot title. Defaults to 'Feature Importance'
    """
    """Plot feature importance."""
    plt.figure(figsize=(12, 8))
    importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
    
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()

def plot_predictions(y_true, y_pred, title='Actual vs Predicted'):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(15, 6))
    
    # Plot actual and predicted values
    plt.plot(y_true.index, y_true.values, label='Actual', alpha=0.7)
    plt.plot(y_pred.index, y_pred.values, label='Predicted', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price (€/MWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt

def plot_error_distribution(y_true, y_pred, title='Error Distribution'):
    """Plot error distribution."""
    errors = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(title)
    plt.xlabel('Error (€/MWh)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    
    return plt

class TimeSeriesCV:
    """Time series cross-validation with expanding window."""
    
    def __init__(self, train_start, test_start, end, validation_window, step_size):
        """
        Initialize TimeSeriesCV.
        
        Args:
            train_start: Start date of the training data
            test_start: Start date of the testing data
            end: End date of the data
            validation_window: Size of validation window in days
            step_size: Step size in days
        """
        self.train_start = pd.to_datetime(train_start)
        self.test_start = pd.to_datetime(test_start)
        self.end = pd.to_datetime(end)
        self.validation_window = pd.Timedelta(days=validation_window)
        self.step_size = pd.Timedelta(days=step_size)
    
    def split(self):
        """Generate train-validation splits."""
        val_start = self.test_start
        while val_start + self.validation_window <= self.end:
            val_end = val_start + self.validation_window
            
            yield {
                'train_start': self.train_start,
                'train_end': val_start,
                'val_start': val_start,
                'val_end': val_end
            }
            
            val_start += self.step_size
