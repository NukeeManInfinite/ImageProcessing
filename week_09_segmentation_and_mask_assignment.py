"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_09_segmentation_and_mask_assignment.py

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

OUTPUT_DIR = Path("week_09_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_synthetic_image(size=256, noise_std=12):
    img = np.full((size, size), 40, dtype=np.float32)
    cv2.circle(img, (size // 2, size // 2), 65, 180, -1)
    cv2.rectangle(img, (30, 30), (90, 90), 120, -1)
    rng = np.random.default_rng(1)
    noisy = np.clip(img + rng.normal(0, noise_std, img.shape), 0, 255).astype(np.uint8)
    return noisy


def global_threshold(image, t):
    return (image >= t).astype(np.uint8)


def main():
    img = create_synthetic_image()

    print("\n=== Task 1: Try different thresholds ===")
    thresholds = [60, 100, 140, 180, 220]
    plt.figure(figsize=(15, 3))
    for i, t in enumerate(thresholds, start=1):
        mask = global_threshold(img, t)
        plt.subplot(1, 5, i)
        plt.imshow(mask, cmap="gray")
        plt.title(f"T={t}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_threshold_sweep.png", bbox_inches="tight")
    plt.show()

    print("\n=== Task 2: Build segmentation function ===")
    def my_threshold(image, t):
        out = np.zeros_like(image, dtype=np.uint8)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                out[y, x] = 1 if image[y, x] >= t else 0
        return out

    mask_builtin = global_threshold(img, 140)
    mask_loop = my_threshold(img, 140)
    print("Functions produce same result:", np.array_equal(mask_builtin, mask_loop))

    print("\n=== Task 3: Noise challenge ===")
    noisy_more = create_synthetic_image(noise_std=28)
    mask_noisy = global_threshold(noisy_more, 140)
    improved = cv2.GaussianBlur(noisy_more, (5, 5), 0)
    mask_improved = global_threshold(improved, 140)

    plt.figure(figsize=(12, 4))
    for i, (image, title) in enumerate(
        [
            (noisy_more, "More noise"),
            (mask_noisy, "Mask on noisy image"),
            (mask_improved, "Blur + threshold"),
        ],
        start=1,
    ):
        plt.subplot(1, 3, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_noise_challenge.png", bbox_inches="tight")
    plt.show()

    print("Answer: With stronger noise the mask becomes less stable. The same threshold may fail more often. Smoothing before thresholding improves the result.")


if __name__ == "__main__":
    main()
