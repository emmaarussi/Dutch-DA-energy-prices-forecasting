import pandas as pd

# Load test set
test_df = pd.read_csv('data/processed/multivariate_features_testset.csv', index_col=0)
test_df.index = pd.to_datetime(test_df.index)

# Load selected and ordered features from train set
train_df = pd.read_csv('data/processed/multivariate_features_selectedXGboost.csv', index_col=0)
train_df.index = pd.to_datetime(train_df.index)

# Get selected feature columns (excluding target_t*)
selected_feature_cols = [col for col in train_df.columns if not col.startswith('target_t')]

# Get target columns from test set
target_cols = [col for col in test_df.columns if col.startswith('target_t')]

# Combine features and targets
required_cols = selected_feature_cols + target_cols

# Check if all required columns exist in test set
missing = set(required_cols) - set(test_df.columns)
if missing:
    raise ValueError(f"❌ Test set is missing columns: {missing}")

# Subset and reorder test set
test_selected_df = test_df[required_cols]

# Save final aligned test set
output_path = 'data/processed/multivariate_features_testset_selectedXGboost.csv'
test_selected_df.to_csv(output_path)
print(f"✅ Aligned test features + targets saved to: {output_path}")