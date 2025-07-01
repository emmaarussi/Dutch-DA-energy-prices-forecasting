import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Base path
base_path = "/Users/emmaarussi/CascadeProjects/thesis-dutch-energy-analysis/models_14_38/xgboost/plots/sieve_online"
 
# Prepare filenames
horizons = [14, 24, 38]
images = []

for h in horizons:
    path = os.path.join(base_path, f"predictions_{h}.png")
    images.append((f"t+{h}", path))

# Create subplots
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(21, 9))  # Adjust size if needed
fig.suptitle("XGBoost Forecasts and Online Sieves Bootstrap Uncertainty Estimation", fontsize=16)

for ax, (title, img_path) in zip(axs.flat, images):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

plt.tight_layout()  # Leave space for suptitle
plt.show()
