"""Visualize MNIST digits under varying Gaussian noise levels."""
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def main():
    # Load MNIST test set
    test_ds = datasets.MNIST("./datasets", train=False, download=True, transform=transforms.ToTensor())
    X = test_ds.data.numpy().astype(np.float32) / 255.0
    y = test_ds.targets.numpy()

    # One sample per digit (0-9)
    base_idxs = [np.where(y == c)[0][0] for c in range(10)]
    X_base = X[base_idxs]

    sigmas = np.arange(0.05, 1.05, 0.05)

    # Grid: rows = noise levels, cols = sigma label + 10 digits
    fig, axs = plt.subplots(
        len(sigmas), 11,
        figsize=(16, 1.5 * len(sigmas)),
        gridspec_kw={"width_ratios": [1] + [1] * 10},
    )

    for row, sigma in enumerate(sigmas):
        X_noisy = np.clip(X_base + np.random.randn(*X_base.shape).astype(np.float32) * sigma, 0.0, 1.0)

        # Sigma label
        axs[row, 0].text(0.5, 0.5, f"$\\sigma={sigma:.2f}$", ha="center", va="center", fontsize=11)
        axs[row, 0].axis("off")

        # Digit images
        for col in range(10):
            axs[row, col + 1].imshow(X_noisy[col], cmap="gray")
            axs[row, col + 1].axis("off")
            if row == 0:
                axs[row, col + 1].set_title(str(col), fontsize=10)

    plt.suptitle("Noisy MNIST digits (one per class, varying noise levels)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("./output/images/noisy_mnist.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
