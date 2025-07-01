import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models_14_38.xgboost.quantile_prediction_modelperq import XGBoostQuantileForecaster


def main():
    try:
        # Load full data (up to March 2024)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        features_path = os.path.join(project_root, 'data', 'processed', 'multivariate_features_selectedXGboost.csv')
        print(f"\nAttempting to load data from: {features_path}")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Data file not found at: {features_path}")

        data = pd.read_csv(features_path, index_col=0)
        data.index = pd.to_datetime(data.index)
        print(f"Successfully loaded data with shape: {data.shape}")

        # Split full train/test based on March 1 cutoff
        train_end = pd.Timestamp('2024-03-01', tz='Europe/Amsterdam')
        full_data = data[:train_end]

        # Load and initialize quantile forecaster
        model = XGBoostQuantileForecaster()

        # Train model for all quantiles
        output_dir = os.path.join(project_root, 'models_14_38/xgboost/final_models_quantile')
        os.makedirs(output_dir, exist_ok=True)

        for horizon in model.horizons:
            print(f"\nTraining full quantile models for t+{horizon}h...")
            X, y = model.prepare_data(full_data, horizon)

            for q in model.quantiles:
                params = model.get_hyperparameters(horizon)
                q_model = XGBRegressor(
                    objective='reg:quantileerror',
                    quantile_alpha=q,
                    **params
                )
                q_model.fit(X, y)
                model.models[(horizon, q)] = q_model

                # Save each quantile model separately
                filename = f"quantile_model_h{horizon}_q{int(q * 100)}.pkl"
                save_path = os.path.join(output_dir, filename)
                joblib.dump(q_model, save_path)
                print(f"✅ Saved model: {save_path}")

        print("\n✅ All quantile models trained and saved successfully.")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
