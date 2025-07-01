import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Gaussian import Gaussian
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics

class XGBoostLSSForecaster:
    def __init__(self, horizons=[14, 24, 38]):
        self.horizons = horizons

    def prepare_data(self, df, horizon):
        """Extract features and target for a given horizon"""
        excluded_patterns = ['target_t']
        feature_cols = [col for col in df.columns if not any(p in col for p in excluded_patterns)]
        X = df[feature_cols]
        y = df[f'target_t{horizon}']
        return X, y

    def train_and_evaluate_window(self, train_data, test_data, horizon):
        """Train model on one window and return metrics, importance"""
        X_train, y_train = self.prepare_data(train_data, horizon)
        X_test, y_test = self.prepare_data(test_data, horizon)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        model = XGBoostLSS(Gaussian())
        params = {
            'eta': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1
        }
        model.train(params=params, dtrain=dtrain, num_boost_round=100)

        # Predict mean and std
        pred_dist = model.predict(dtest)
        pred_mean = pred_dist['loc'].values
        pred_std = pred_dist['scale'].values

        # Compute metrics
        metrics = calculate_metrics(y_test, pred_mean)
        

        # No SHAP or tree importance currently extracted
        importance = None

        return model, metrics, importance

    def rolling_window_evaluation(self, data, window_size='365D', step_size='7D', test_size='7D'):
        """Perform rolling window cross-validation."""
        window_td = pd.Timedelta(window_size)
        step_td = pd.Timedelta(step_size)
        test_td = pd.Timedelta(test_size)

        all_metrics = {h: [] for h in self.horizons}
        all_predictions = {h: pd.DataFrame() for h in self.horizons}

        start_time = data.index.min()
        end_time = data.index.max() - test_td
        current_start = start_time
        window_count = 0

        while current_start + window_td <= end_time:
            window_count += 1
            print(f"\n📦 Window {window_count}: {current_start.date()} to {(current_start + window_td).date()}")

            train_data = data[current_start: current_start + window_td]
            test_data = data[current_start + window_td: current_start + window_td + test_td]

            for horizon in self.horizons:
                print(f"⏱️  Training t+{horizon}h...")
                model, metrics, _ = self.train_and_evaluate_window(train_data, test_data, horizon)

                X_test, y_test = self.prepare_data(test_data, horizon)
                dtest = xgb.DMatrix(X_test)
                pred_dist = model.predict(dtest)

                df_pred = pd.DataFrame({
                    'actual': y_test,
                    'predicted_mean': pred_dist['loc'].values,
                    'predicted_std': pred_dist['scale'].values,
                }, index=y_test.index)

                all_predictions[horizon] = pd.concat([all_predictions[horizon], df_pred])
                all_metrics[horizon].append(metrics)

            current_start += step_td

        # Aggregate metrics
        final_metrics = {}
        for horizon in self.horizons:
            mlist = all_metrics[horizon]
            final_metrics[horizon] = {
                'RMSE': np.mean([m['RMSE'] for m in mlist]),
                'RMSE_std': np.std([m['RMSE'] for m in mlist]),
                'SMAPE': np.mean([m['SMAPE'] for m in mlist]),
                'SMAPE_std': np.std([m['SMAPE'] for m in mlist]),
                'R2': np.mean([m['R2'] for m in mlist]),
                'R2_std': np.std([m['R2'] for m in mlist])
            }

        return final_metrics, all_predictions

    

if __name__ == "__main__":
    # Load full dataset
    data_path = Path('data/processed/multivariate_features.csv')
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Amsterdam')

    # Initialize and run forecaster
    model = XGBoostLSSForecaster(horizons=[14, 24, 38])
    final_metrics, all_predictions = model.rolling_window_evaluation(
        data=df,
        window_size='365D',
        step_size='7D',
        test_size='7D'
    )

    # Save predictions
    Path("predictions").mkdir(exist_ok=True)
    for horizon, preds in all_predictions.items():
        preds.to_csv(f'predictions/xgboostlss_rolling_t{horizon}.csv')
        print(f"✅ Saved predictions for t+{horizon}h.")

    # Print metrics
    print("\n📊 Final cross-validated metrics:")
    for h, m in final_metrics.items():
        print(f"\nt+{h}h:")
        for k, v in m.items():
            print(f"  {k}: {v:.2f}")

       # Create output directory
    out_dir = 'models_14_38/xgboost/plots/lss'
    os.makedirs(out_dir, exist_ok=True)

    # Plot feature importance for each horizon
    for horizon in model.horizons:
        importance_df = all_importance[horizon]
        plot_feature_importance(
            importance_df,
            top_n=20,
            title=f'Feature Importance (t+{horizon}h)',
            filename=f'{out_dir}/feature_importance_h{horizon}.png'
        )
        plt.close()
        
    # Plot predictions over time
    for horizon in model.horizons:
        predictions_df = results[horizon]['predictions']
        plt.figure(figsize=(15, 6))
        plt.plot(predictions_df.index, predictions_df['actual'], label='Actual', alpha=0.7)
        plt.plot(predictions_df.index, predictions_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (t+{horizon}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/CV_predictions_over_time_h{horizon}_lss.png')
        plt.close()