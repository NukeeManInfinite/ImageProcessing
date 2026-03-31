"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_08_edge_detection_intro_assignment.py

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

OUTPUT_DIR = Path("week_08_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def show_grid(images, titles, save_name):
    plt.figure(figsize=(12, 8))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(2, 2, i)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_name, bbox_inches="tight")
    plt.show()


def main():
    img = data.camera()

    # Simple gradient
    grad_x = np.diff(img.astype(np.float32), axis=1, prepend=img[:, :1])
    grad_y = np.diff(img.astype(np.float32), axis=0, prepend=img[:1, :])
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = np.clip(grad_mag, 0, 255).astype(np.uint8)

    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    sobel = np.clip(np.abs(sobel), 0, 255).astype(np.uint8)

    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    laplacian = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)

    canny = cv2.Canny(img, 80, 160)

    show_grid(
        [grad_mag, sobel, laplacian, canny],
        ["Simple gradient", "Sobel", "Laplacian", "Canny"],
        "01_edge_methods.png",
    )

    print("This script covers the main Week 08 edge-detection methods: simple gradient, Sobel, Laplacian, and Canny.")


if __name__ == "__main__":
    main()
