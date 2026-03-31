"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_05_histogram_equalization_assignment.py

Optional:
    Put your own image in the same folder and set IMAGE_PATH below.
    If IMAGE_PATH is None or the file is missing, the script uses a built-in sample image.

Libraries:
    pip install opencv-python matplotlib scikit-image pillow
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data

OUTPUT_DIR = Path("week_05_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_hist(ax, image, title):
    ax.hist(image.ravel(), bins=256, range=(0, 256))
    ax.set_title(title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")


def main():
    img = data.moon()
    img_dark = np.clip(img * 0.55, 0, 255).astype(np.uint8)

    print("\n=== Task 1: Histogram equalization ===")
    equalized = cv2.equalizeHist(img_dark)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(img_dark, cmap="gray")
    axes[0, 0].set_title("Original dark image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(equalized, cmap="gray")
    axes[0, 1].set_title("Histogram equalized")
    axes[0, 1].axis("off")

    plot_hist(axes[1, 0], img_dark, "Histogram before")
    plot_hist(axes[1, 1], equalized, "Histogram after")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_histogram_equalization.png", bbox_inches="tight")
    plt.show()

    print("Discussion: histogram equalization spreads intensity values over a wider range, which improves visual contrast in dim or low-contrast images.")

    print("\n=== Extra: CLAHE comparison ===")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_dark)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, title in zip(
        axes,
        [img_dark, equalized, clahe],
        ["Original dark", "Equalized", "CLAHE"],
    ):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_equalized_vs_clahe.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
