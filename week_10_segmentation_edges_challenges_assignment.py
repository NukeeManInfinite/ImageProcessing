
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

OUTPUT_DIR = Path("week_10_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_synthetic_image(size=256, noise_std=10):
    image = np.full((size, size), 50, dtype=np.float32)
    cv2.circle(image, (80, 80), 45, 165, -1)
    cv2.rectangle(image, (140, 60), (220, 150), 210, -1)
    cv2.ellipse(image, (150, 200), (50, 25), 0, 0, 360, 120, -1)
    rng = np.random.default_rng(7)
    image = np.clip(image + rng.normal(0, noise_std, image.shape), 0, 255).astype(np.uint8)
    return image


def global_threshold(image, t):
    return (image >= t).astype(np.uint8)


def my_threshold(image, t):
    out = np.zeros_like(image, dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            out[y, x] = 1 if image[y, x] >= t else 0
    return out


def gradient_edges(image, t_edge=60):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_uint8 = np.clip(mag, 0, 255).astype(np.uint8)
    edges = (mag_uint8 >= t_edge).astype(np.uint8)
    return mag_uint8, edges


def main():
    img = create_synthetic_image()

    print("\n=== Task 1: different threshold values ===")
    thresholds = [60, 100, 140, 180, 220]
    plt.figure(figsize=(15, 3))
    for i, t in enumerate(thresholds, start=1):
        plt.subplot(1, 5, i)
        plt.imshow(global_threshold(img, t), cmap="gray")
        plt.title(f"T={t}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_threshold_values.png", bbox_inches="tight")
    plt.show()

    print("\n=== Task 2: custom threshold function ===")
    custom = my_threshold(img, 140)
    builtin = global_threshold(img, 140)
    print("Custom matches vectorized version:", np.array_equal(custom, builtin))

    print("\n=== Task 3 + Challenge 1: noise and Otsu ===")
    noisy = create_synthetic_image(noise_std=24)
    otsu_t = int(threshold_otsu(noisy))
    mask_manual = global_threshold(noisy, 140)
    mask_otsu = global_threshold(noisy, otsu_t)
    print("Otsu threshold:", otsu_t)

    plt.figure(figsize=(12, 4))
    for i, (image, title) in enumerate(
        [(noisy, "Noisy image"), (mask_manual, "Manual threshold"), (mask_otsu, "Otsu threshold")],
        start=1,
    ):
        plt.subplot(1, 3, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_otsu_comparison.png", bbox_inches="tight")
    plt.show()

    print("\n=== Challenge 2: count white pixels ===")
    white_pixels = int(mask_otsu.sum())
    percentage = 100.0 * white_pixels / mask_otsu.size
    print("Foreground pixels:", white_pixels)
    print(f"Foreground percentage: {percentage:.2f}%")

    print("\n=== Challenge 3: Sobel vs Canny ===")
    sobel_mag, sobel_edges = gradient_edges(noisy, t_edge=70)
    canny_edges = cv2.Canny(cv2.GaussianBlur(noisy, (5, 5), 0), 70, 150)

    plt.figure(figsize=(12, 4))
    for i, (image, title) in enumerate(
        [(noisy, "Noisy image"), (sobel_edges, "Sobel thresholded"), (canny_edges, "Canny")],
        start=1,
    ):
        plt.subplot(1, 3, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_edge_comparison.png", bbox_inches="tight")
    plt.show()

    print("Answer: Canny usually gives cleaner and thinner edges, while Sobel is easier to compute but more sensitive to threshold choice and noise.")

    print("\n=== Final Mini Project ===")
    masked = img * mask_otsu
    _, gradient_edge_map = gradient_edges(img, 60)

    plt.figure(figsize=(16, 4))
    for i, (image, title) in enumerate(
        [
            (img, "Input image"),
            (mask_otsu, "Threshold mask"),
            (masked, "Masked image"),
            (gradient_edge_map, "Gradient edge map"),
        ],
        start=1,
    ):
        plt.subplot(1, 4, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_final_pipeline.png", bbox_inches="tight")
    plt.show()

    cv2.imwrite(str(OUTPUT_DIR / "final_mask.png"), (mask_otsu * 255).astype(np.uint8))
    cv2.imwrite(str(OUTPUT_DIR / "final_edges.png"), (gradient_edge_map * 255).astype(np.uint8))
    print("Saved final mask and final edge image.")


if __name__ == "__main__":
    main()
