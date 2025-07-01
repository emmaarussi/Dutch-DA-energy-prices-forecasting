import matplotlib.pyplot as plt
import numpy as np

# Configuration
n_points = 20
n_folds = 5
window_size = 8  # fixed train window size
test_size = 2
colors = {'train': '#4CAF50', 'test': 'orange'}  # Grassier green

fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

# ----- Rolling-Origin Cross-Validation (Fixed Window) -----
for i in range(n_folds):
    train_start = i * test_size
    train_end = train_start + window_size
    test_start = train_end
    test_end = test_start + test_size

    axs[0].plot(range(train_start, train_end), [i]*window_size, color=colors['train'], lw=6, label='Train' if i == 0 else "")
    axs[0].plot(range(test_start, test_end), [i]*test_size, color=colors['test'], lw=6, label='Test' if i == 0 else "")

axs[0].set_yticks(range(n_folds))
axs[0].set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])
axs[0].set_title('Rolling-Origin Cross-Validation (Fixed Window)')
axs[0].legend(loc='upper right')
axs[0].grid(True, axis='x', linestyle='--', alpha=0.5)

# ----- K-Fold Cross-Validation -----
indices = np.arange(n_points)
folds = np.array_split(indices, n_folds)

for i, test_idx in enumerate(folds):
    train_idx = np.setdiff1d(indices, test_idx)
    axs[1].plot(train_idx, [i]*len(train_idx), color=colors['train'], lw=6, label='Train' if i == 0 else "")
    axs[1].plot(test_idx, [i]*len(test_idx), color=colors['test'], lw=6, label='Test' if i == 0 else "")

axs[1].set_yticks(range(n_folds))
axs[1].set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])
axs[1].set_title('Standard K-Fold Cross-Validation')
axs[1].set_xlabel('Time Index')
axs[1].legend(loc='upper right')
axs[1].grid(True, axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

