import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Base path
base_path = "/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/models_14_38/xgboost/plots/optimizedretrain"

# Prepare filenames
horizons = [14, 24, 38]
n_folds = 7
images = []

for h in horizons:
    for fold in range(n_folds):
        path = os.path.join(base_path, f"predictions_{h}_fold{fold}.png")
        images.append((f"t+{h} Fold {fold}", path))

# Create subplots
fig, axs = plt.subplots(nrows=3, ncols=7, figsize=(21, 9))  # Adjust size if needed
fig.suptitle("XGBoost Forecasts: Folds and Horizons", fontsize=16)

for ax, (title, img_path) in zip(axs.flat, images):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
plt.show()
