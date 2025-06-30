"""
This is the quantile regression model i am using to predict quantiles for CI creation
What this model does, essentially, is the same as the XGBoost model, which predicts the 
mean of the target variable. Here the loss function is a pinball loss, which is a 
loss function that is used to predict quantiles of the target variable.


The hyperparameters and the features where already optimized for the XGBoost model, 
so i will use the same hyperparameters and features for the quantile regression model.
so basically, this going to be used later for the Sequential Predictive Conformal Inference. Where XGboost is fit on full train and 
predicts on last 3 months test set. and then the quantile regression model is fit on the last 3 months test set and 
predicts on the last 3 months test set.

 Residual Autocorrelation Test (Ljung-Box)
        lb_stat      lb_pvalue
1    569.962983  5.719643e-126
2    967.891628  6.683492e-211
3   1240.777650  1.041581e-268
4   1425.669744  1.876490e-307
5   1547.550394   0.000000e+00
6   1625.150360   0.000000e+00
7   1675.322139   0.000000e+00
8   1710.873654   0.000000e+00
9   1738.729263   0.000000e+00
10  1757.981011   0.000000e+00
11  1767.511726   0.000000e+00
12  1770.279435   0.000000e+00
13  1770.320808   0.000000e+00
14  1771.180426   0.000000e+00
15  1772.944065   0.000000e+00
16  1774.654984   0.000000e+00
17  1776.419888   0.000000e+00
18  1778.063444   0.000000e+00
19  1778.442436   0.000000e+00
20  1779.122839   0.000000e+00

final try

📦 Predicting for t+14h

📊 Evaluation for t+14h horizon:
Number of predictions: 730
RMSE: 14.46
SMAPE: 20.27%
R2: 0.4998
90% Prediction Interval Coverage: 87.5%

📦 Predicting for t+24h

📊 Evaluation for t+24h horizon:
Number of predictions: 730
RMSE: 17.06
SMAPE: 22.91%
R2: 0.2799
90% Prediction Interval Coverage: 86.8%

📦 Predicting for t+38h

📊 Evaluation for t+38h horizon:
Number of predictions: 730
RMSE: 15.70
SMAPE: 20.41%
R2: 0.3872
90% Prediction Interval Coverage: 87.1%


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
    def __init__(self, window_size=100):
        self.horizons = [14, 24, 38]  # Only 14 24 and 38 
        self.quantiles = [0.05, 0.5, 0.95]
        self.models = {}
        self.scaler = RobustScaler()
        self.window_size = window_size

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

    def train_and_predict(self, residuals, X_test_times, window_size_str, horizon):
        if len(residuals) < self.window_size + 1:
            print(f"⚠️ Not enough residuals for horizon {horizon}")
            return pd.DataFrame()

        # Bouw trainingsset
        X_train, y_train = self.build_residual_window_dataset(residuals, self.window_size)

        # Check of trainingsdata geldig is
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            print(f"❌ Empty training data for t+{horizon}h")
            return pd.DataFrame()

        print(f"✅ Training data shape for t+{horizon}h: X={X_train.shape}, y={y_train.shape}")

        # Init collecties
        quantile_predictions = {q: [] for q in self.quantiles}
        timestamps = []

        # Sliding prediction op testset
        for t_idx in range(self.window_size, len(X_test_times) - horizon):
            x_window = residuals.iloc[t_idx - self.window_size:t_idx].values
            x_input = x_window.reshape(1, -1)

            for q in self.quantiles:
                if (horizon, q) not in self.models:
                    params = self.get_hyperparameters(horizon)
                    model = XGBRegressor(
                        objective='reg:quantileerror',
                        quantile_alpha=q,
                        **params
                    )
                    model.fit(X_train, y_train)
                    self.models[(horizon, q)] = model

                model = self.models[(horizon, q)]
                pred = model.predict(x_input)[0]
                quantile_predictions[q].append(pred)

            timestamps.append(X_test_times[t_idx + horizon])

        # Resultaten structureren
        predictions_all = []
        for i, ts in enumerate(timestamps):
            predictions_all.append({
                'window_size': window_size_str,
                'target_time': ts,
                'horizon': horizon,
                'q05': quantile_predictions[0.05][i],
                'q50': quantile_predictions[0.5][i],
                'q95': quantile_predictions[0.95][i],
            })

        results_df = pd.DataFrame(predictions_all)
        results_df['target_time'] = pd.to_datetime(results_df['target_time'], utc=True)
        results_df = results_df.sort_values('target_time')
        return results_df


    def build_residual_window_dataset(self, residuals, window_size):
        X, y = [], []
        for i in range(window_size, len(residuals)):
            window = residuals.iloc[i - window_size:i].values
            if len(window) == window_size:
                X.append(window)
                y.append(residuals.iloc[i])
        if len(X) == 0 or len(y) == 0:
            print("⚠️ No valid training samples constructed from residuals.")
        return np.vstack(X), np.array(y)




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
        width = np.mean(df['q95'] - df['q05'])
        print(f"90% Prediction Interval Coverage: {coverage:.1f}%")
        print(f"Mean Width: {width:.2f}")

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
    
    # Loop through all horizons
    for horizon in model.horizons:
        print(f"\n📦 Predicting for t+{horizon}h")
        results_df = model.train_and_predict(train_df, test_df, '12m', horizon)

        if results_df.empty:
            print(f"❌ No predictions for t+{horizon}h.")
            continue

        model.plot_predictions(results_df, horizon)
        model.evaluate_predictions(results_df, horizon)

if __name__ == "__main__":
    main()
