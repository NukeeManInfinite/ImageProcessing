"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_06_filtering_assignment.py

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

OUTPUT_DIR = Path("week_06_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def show_grid(images, titles, save_name):
    plt.figure(figsize=(14, 8))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_name, bbox_inches="tight")
    plt.show()


def main():
    img = data.camera()
    rng = np.random.default_rng(42)

    print("\n=== Create noisy versions ===")
    gaussian_noise = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
    sp_noise = img.copy()
    salt = rng.random(img.shape) < 0.02
    pepper = rng.random(img.shape) < 0.02
    sp_noise[salt] = 255
    sp_noise[pepper] = 0

    print("\n=== Filtering ===")
    mean_filtered = cv2.blur(gaussian_noise, (5, 5))
    gaussian_filtered = cv2.GaussianBlur(gaussian_noise, (5, 5), 0)
    median_filtered = cv2.medianBlur(sp_noise, 5)
    bilateral_filtered = cv2.bilateralFilter(gaussian_noise, d=7, sigmaColor=50, sigmaSpace=50)

    show_grid(
        [img, gaussian_noise, mean_filtered, gaussian_filtered, sp_noise, median_filtered],
        ["Original", "Gaussian noise", "Mean filter", "Gaussian blur", "Salt & pepper", "Median filter"],
        "01_filtering_results.png",
    )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gaussian_noise, cmap="gray")
    plt.title("Noisy image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(bilateral_filtered, cmap="gray")
    plt.title("Bilateral filter")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_bilateral.png", bbox_inches="tight")
    plt.show()

    print("Observation:")
    print("- Mean and Gaussian filters smooth noise but can blur edges.")
    print("- Median filtering is especially effective for salt-and-pepper noise.")
    print("- Bilateral filtering reduces noise while preserving edges better than standard smoothing.")


if __name__ == "__main__":
    main()
