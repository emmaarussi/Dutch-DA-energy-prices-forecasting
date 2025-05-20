"""
This is the quantile regression model i am using to predict quantiles for CI creation
What this model does, essentially, is the same as the XGBoost model, which predicts the 
mean of the target variable. Here the loss function is a pinball loss, which is a 
loss function that is used to predict quantiles of the target variable.


The hyperparameters and the features where already optimized for the XGBoost model, 
so i will use the same hyperparameters and features for the quantile regression model.
so basically, this going to be used later for the Sequential Predictive Conformal Inference.




"""












from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_pinball_loss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

import sys
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

class XGBoostQuantileForecaster:
    def __init__(self):
        self.horizons = range(14, 15)  # Only t+14 for now
        self.quantiles = [0.05, 0.5, 0.95]
        self.models = {}
        self.scaler = RobustScaler()

    def get_hyperparameters(self, horizon):
        if horizon <= 24:
            return {
                'max_depth': 8, 'learning_rate': 0.0110, 'n_estimators': 1300,
                'min_child_weight': 3.1910, 'subsample': 0.8837, 'colsample_bytree': 0.8369,
                'gamma': 0.1393, 'random_state': 42
            }
        elif horizon <= 31:
            return {
                'max_depth': 9, 'learning_rate': 0.0693, 'n_estimators': 1400,
                'min_child_weight': 1.4524, 'subsample': 0.7879, 'colsample_bytree': 0.8769,
                'gamma': 0.1656, 'random_state': 42
            }
        else:  # for t+38h
            return {
                'max_depth': 9, 'learning_rate': 0.0179, 'n_estimators': 1100,
                'min_child_weight': 5.4722, 'subsample': 0.8081, 'colsample_bytree': 0.8531,
                'gamma': 0.2176, 'random_state': 42
            }

    def prepare_data(self, df, horizon):
        feature_cols = [col for col in df.columns 
                        if not col.startswith('target_t')]
        X = df[feature_cols]
        y = df[f'target_t{horizon}']
        return X, y

    def train_and_predict(self, train_window, test_window, window_size_str, horizon):
        predictions_all = []

        X_train, y_train = self.prepare_data(train_window, horizon)
        X_test, y_test = self.prepare_data(test_window, horizon)

        if len(X_train) < 100 or len(X_test) == 0:
            print(f"⚠️ Insufficient data for horizon {horizon}")
            return pd.DataFrame()

        quantile_predictions = {}

        for q in self.quantiles:
            params = self.get_hyperparameters(horizon)
            try:
                model = XGBRegressor(
                    objective='reg:quantileerror',
                    quantile_alpha=q,
                    **params
                )
                model.fit(X_train, y_train)
                self.models[(horizon, q)] = model
                quantile_predictions[q] = model.predict(X_test)
            except Exception as e:
                print(f"⚠️ Error training model for t+{horizon}h, q={q}: {e}")
                return pd.DataFrame()

        if all(q in quantile_predictions for q in self.quantiles):
            for idx, target_time in enumerate(y_test.index):
                predictions_all.append({
                    'window_size': window_size_str,
                    'target_time': target_time,
                    'horizon': horizon,
                    'actual': y_test.iloc[idx],
                    'q05': quantile_predictions[0.05][idx],
                    'q50': quantile_predictions[0.5][idx],
                    'q95': quantile_predictions[0.95][idx]
                })

        residuals = y_test.values - quantile_predictions[0.5]
        self.residuals = pd.Series(residuals, index=y_test.index)

        results_df = pd.DataFrame(predictions_all)
        if not results_df.empty:
            results_df['target_time'] = pd.to_datetime(results_df['target_time'])
            results_df = results_df.sort_values('target_time')
        return results_df

    def test_residual_dependence(self, lags=20):
        if hasattr(self, 'residuals') and not self.residuals.empty:
            print("\n🧪 Residual Autocorrelation Test (Ljung-Box)")
            lb_test = acorr_ljungbox(self.residuals, lags=lags, return_df=True)
            print(lb_test[['lb_stat', 'lb_pvalue']])
        
            # Optional: visualize ACF
            plt.figure(figsize=(8, 4))
            plot_acf(self.residuals, lags=lags)
            plt.title("Residual ACF (Median Prediction)")
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ No residuals found for autocorrelation test.")

    def evaluate_predictions(self, results_df, horizon):
        df = results_df[results_df['horizon'] == horizon].copy()
        if df.empty:
            print(f"No results to evaluate for t+{horizon}h.")
            return

        metrics = calculate_metrics(df['actual'], df['q50'])
        print(f"\n📊 Evaluation for t+{horizon}h horizon:")
        print(f"Number of predictions: {len(df)}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R2: {metrics['R2']:.4f}")

        coverage = np.mean((df['actual'] >= df['q05']) & (df['actual'] <= df['q95'])) * 100
        print(f"90% Prediction Interval Coverage: {coverage:.1f}%")

    def plot_predictions(self, results_df, horizon):
        df = results_df[results_df['horizon'] == horizon].copy()
        if df.empty:
            return

        df = df.sort_values('target_time')
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'plots', 'quantile')
        os.makedirs(output_dir, exist_ok=True)

        # Main prediction plot with confidence intervals
        plt.figure(figsize=(12, 6))
        plt.plot(df['target_time'], df['actual'], label='Actual', alpha=0.7)
        plt.plot(df['target_time'], df['q50'], label='Predicted', alpha=0.7)
        plt.fill_between(df['target_time'], df['q05'], df['q95'], alpha=0.2, label='90% CI')
        plt.title(f'Actual vs Predicted Prices - t+{horizon}h')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'predictions_{horizon}.png'))
        plt.close()

        # Residuals plot
        residuals = df['actual'] - df['q50']
        plt.figure(figsize=(12, 6))
        plt.scatter(df['q50'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'Residuals vs Predicted Values - t+{horizon}h')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'residuals_{horizon}.png'))
        plt.close('all')

def main():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
    df = pd.read_csv(features_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')

    # Define train-test split
    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')

    train_df = df[train_start:train_end]
    test_df = df[test_start:test_end]

    # Initialize and train model
    model = XGBoostQuantileForecaster()
    results_df = model.train_and_predict(train_df, test_df, '12m', 14)
    model.test_residual_dependence()

    if results_df.empty:
        print("❌ No predictions to evaluate or plot.")
        return

    for horizon in model.horizons:
        model.plot_predictions(results_df, horizon)
        model.evaluate_predictions(results_df, horizon)

if __name__ == "__main__":
    main()
