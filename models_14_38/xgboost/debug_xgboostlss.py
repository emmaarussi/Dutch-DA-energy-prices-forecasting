import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
import matplotlib.pyplot as plt

def load_and_prepare_data(horizon=14):
    """Load and prepare data for a specific horizon"""
    data_path = Path('data/processed/multivariate_features.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')
    
    # Train-test split
    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
    
    train_df = df[train_start:train_end]
    test_df = df[test_start:test_end]
    
    # Get target columns and feature columns
    target_cols = [col for col in df.columns if col.startswith('target_t')]
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[f'target_t{horizon}']
    X_test = test_df[feature_cols]
    y_test = test_df[f'target_t{horizon}']
    
    return X_train, y_train, X_test, y_test

def train_and_analyze_model():
    """Train model and analyze its behavior"""
    # Load data
    X_train, y_train, X_test, y_test = load_and_prepare_data()
    
    print("Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Initialize and train model with simple parameters first
    model = XGBoostLSS(Gaussian())
    
    # Basic parameters
    params = {
        'eta': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1
    }
    
    print("\nTraining model with basic parameters...")
    model.train(params=params, dtrain=dtrain, num_boost_round=100, verbose_eval=10)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(dtest)
    
    # Debug prediction output
    print("\nPrediction output:")
    print(f"predictions type: {type(predictions)}")
    print(f"predictions columns: {predictions.columns}")
    
    # Extract mean and std from predictions
    pred_mean = predictions['loc'].values
    pred_std = predictions['scale'].values
    y_test_values = y_test.values
    
    # Calculate basic metrics
    mse = np.mean((pred_mean - y_test_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_mean - y_test_values))
    
    print("\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': pred_mean,
        'Lower': pred_mean - 1.96 * pred_std,
        'Upper': pred_mean + 1.96 * pred_std
    }, index=y_test.index)
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df.index, plot_df['Actual'], label='Actual', alpha=0.7)
    plt.plot(plot_df.index, plot_df['Predicted'], label='Predicted', alpha=0.7)
    plt.fill_between(plot_df.index, 
                     plot_df['Lower'],
                     plot_df['Upper'],
                     alpha=0.2, label='95% CI')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models_14_38/xgboost/plots/debug/predictions.png')
    plt.close()
    
    # Calculate prediction intervals coverage
    coverage = np.mean((plot_df['Actual'] >= plot_df['Lower']) & 
                      (plot_df['Actual'] <= plot_df['Upper']))
    print(f"\nPrediction interval coverage: {coverage:.2%}")
    
    # Analyze residuals
    residuals = y_test - pred_mean
    plt.figure(figsize=(12, 6))
    plt.scatter(pred_mean, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig('models_14_38/xgboost/plots/debug/residuals.png')
    plt.close()
    
    # Print residuals statistics
    print("\nResiduals Statistics:")
    print(pd.Series(residuals).describe())
    
    return model, pred_mean, pred_std, y_test

if __name__ == "__main__":
    # Create plots directory
    Path('models_14_38/xgboost/plots/debug').mkdir(parents=True, exist_ok=True)
    
    # Train and analyze model
    model, pred_mean, pred_std, y_test = train_and_analyze_model()
