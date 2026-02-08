"""Visualize MNIST digits under varying Gaussian noise levels."""

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from utils import centroid_representative_indices


def main():
    # Load MNIST test set
    test_ds = datasets.MNIST("./datasets", train=False, download=True, transform=transforms.ToTensor())
    X = test_ds.data.numpy().astype(np.float32) / 255.0
    y = test_ds.targets.numpy()

    # One sample per digit (0-9), chosen as closest to class centroid
    X_flat = X.reshape(len(X), -1)  # (n_samples, 784)
    base_idxs = centroid_representative_indices(X_flat, y)
    X_base = X[base_idxs]

    sigmas = np.arange(0.0, 0.51, 0.17)

    # Grid: rows = noise levels, cols = sigma label + 10 digits
    fig, axs = plt.subplots(
        len(sigmas),
        11,
        figsize=(12, 1.0 * len(sigmas)),
        gridspec_kw={"width_ratios": [0.8] + [1] * 10, "wspace": -0.55, "hspace": 0.05},
    )

    for row, sigma in enumerate(sigmas):
        X_noisy = np.clip(X_base + np.random.randn(*X_base.shape).astype(np.float32) * sigma, 0.0, 1.0)

        # Sigma label
        axs[row, 0].text(0.5, 0.5, f"${sigma:.2f}$", ha="center", va="center", fontsize=11)
        axs[row, 0].axis("off")

        # Digit images
        for col in range(10):
            axs[row, col + 1].imshow(X_noisy[col], cmap="gray")
            axs[row, col + 1].axis("off")
            if row == 0:
                axs[row, col + 1].set_title(str(col), fontsize=10)

    plt.subplots_adjust(top=0.96, bottom=0.02, left=0.02, right=0.98)
    plt.savefig("./output/images/noisy_mnist.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
