

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import calculate_metrics, plot_feature_importance

class XGBoostclean:
    def __init__(self, horizons=range(14, 39)):
        self.horizons = horizons
        self.features_saved = False  # Track if features have been saved

    def prepare_data(self, data, horizon):
        feature_cols = [col for col in data.columns if not col.startswith('target_t')]
        X = data[feature_cols]
        y = data[f'target_t{horizon}']
        return X, y

    def train_and_evaluate(self, train_data, test_data, horizon, full_data):
        X_train, y_train = self.prepare_data(train_data, horizon)
        X_test, y_test = self.prepare_data(test_data, horizon)

        # Feature selection model
        fs_model = xgb.XGBRegressor(**self.get_hyperparameters(horizon))
        fs_model.fit(X_train, y_train)
        importances = pd.Series(fs_model.feature_importances_, index=X_train.columns)
        top_features = importances.sort_values(ascending=False).head(100).index

        # Subset data
        X_train = X_train[top_features]
        X_test = X_test[top_features]

        # Final model
        model = xgb.XGBRegressor(**self.get_hyperparameters(horizon))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = calculate_metrics(y_test, predictions)

        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions
        }, index=test_data.index)

        # Save selected features once
        if not self.features_saved:
            selected_features = list(top_features)
            selected_features_with_targets = selected_features + [col for col in full_data.columns if col.startswith('target_t')]
            selected_df = full_data[selected_features_with_targets]
            selected_df.to_csv('data/processed/multivariate_features_selectedXGboost.csv')
            print("✅ Saved multivariate_features_selectedXGboost.csv with top 100 features.")
            self.features_saved = True

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': importance,
            'predictions': predictions_df
        }

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

def main():
    plots_dir = 'models_14_38/xgboost/plots'
    os.makedirs(plots_dir, exist_ok=True)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_nooutliers.csv')
    data = pd.read_csv(features_path, index_col=0)
    data.index = pd.to_datetime(data.index)

    model = XGBoostclean(horizons=[14, 24, 38])

    train_start = pd.Timestamp('2023-01-08', tz='Europe/Amsterdam')
    train_end = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_start = pd.Timestamp('2024-01-29', tz='Europe/Amsterdam')
    test_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')

    train_df = data[train_start:train_end]
    test_df = data[test_start:test_end]

    results = {}
    for horizon in model.horizons:
        print(f"\nTraining and evaluating horizon t+{horizon}h...")
        results[horizon] = model.train_and_evaluate(train_df, test_df, horizon, full_data=data)

    out_dir = 'models_14_38/xgboost/plots/cleanfull'
    os.makedirs(out_dir, exist_ok=True)

    for horizon in model.horizons:
        result = results[horizon]
        predictions_df = result['predictions']
        metrics = result['metrics']
        print(f"\nt+{horizon}h horizon:")
        print(f"Number of predictions: {len(predictions_df)}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"SMAPE: {metrics['SMAPE']:.2f}%")
        print(f"R2: {metrics['R2']:.4f}")

        importance_df = result['feature_importance']
        plot_feature_importance(
            importance_df,
            top_n=20,
            title=f'Feature Importance (t+{horizon}h)',
            filename=f'{out_dir}/feature_importance_h{horizon}.png'
        )
        plt.close()

        plt.figure(figsize=(15, 6))
        plt.plot(predictions_df.index, predictions_df['actual'], label='Actual', alpha=0.7)
        plt.plot(predictions_df.index, predictions_df['predicted'], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Prices Over Time (t+{horizon}h)')
        plt.xlabel('Date')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/predictions_over_time_h{horizon}.png')
        plt.close()

if __name__ == "__main__":
    main()