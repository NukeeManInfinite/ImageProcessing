"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_07_edge_detection_assignment.py

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

OUTPUT_DIR = Path("week_07_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def show(img, title, cmap="gray", save_name=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    if save_name:
        plt.savefig(OUTPUT_DIR / save_name, bbox_inches="tight")
    plt.show()


def make_step_image(size=256):
    img = np.zeros((size, size), dtype=np.uint8)
    img[:, size // 2 :] = 220
    return img


def make_square_image(size=256):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(img, (64, 64), (192, 192), 255, -1)
    return img


def make_circle_image(size=256):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), 70, 255, -1)
    return img


def main():
    img_step = make_step_image()
    img_square = make_square_image()
    img_circle = make_circle_image()

    print("\n=== Topic 2: Sobel tasks ===")
    sobel_x = cv2.Sobel(img_circle, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_circle, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)
    show(np.abs(sobel_x), "Sobel X - circle", save_name="01_sobel_x.png")
    show(np.abs(sobel_y), "Sobel Y - circle", save_name="02_sobel_y.png")
    show(sobel_mag, "Sobel magnitude - circle", save_name="03_sobel_mag.png")

    print("\n=== Topic 3: Laplacian tasks ===")
    rng = np.random.default_rng(0)
    noisy_square = np.clip(img_square + rng.normal(0, 25, img_square.shape), 0, 255).astype(np.uint8)
    lap_no_blur = cv2.Laplacian(noisy_square, cv2.CV_64F, ksize=3)
    blurred = cv2.GaussianBlur(noisy_square, (5, 5), 0)
    lap_blur = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    show(noisy_square, "Noisy square", save_name="04_noisy_square.png")
    show(np.abs(lap_no_blur), "Laplacian without blur", save_name="05_laplacian_no_blur.png")
    show(np.abs(lap_blur), "Laplacian after Gaussian blur", save_name="06_laplacian_blur.png")

    print("\n=== Topic 4: Canny tasks ===")
    threshold_pairs = [(30, 90), (50, 150), (100, 200), (150, 250)]
    for low, high in threshold_pairs:
        edges = cv2.Canny(img_step, low, high)
        show(edges, f"Canny on step image: low={low}, high={high}", save_name=f"07_canny_{low}_{high}.png")

    sobel_step_x = cv2.Sobel(noisy_square, cv2.CV_64F, 1, 0, ksize=3)
    sobel_step_y = cv2.Sobel(noisy_square, cv2.CV_64F, 0, 1, ksize=3)
    sobel_noisy_mag = np.sqrt(sobel_step_x ** 2 + sobel_step_y ** 2)
    sobel_noisy_mag = np.clip(sobel_noisy_mag, 0, 255).astype(np.uint8)
    lap_noisy = np.abs(cv2.Laplacian(noisy_square, cv2.CV_64F, ksize=3)).astype(np.uint8)
    canny_noisy = cv2.Canny(cv2.GaussianBlur(noisy_square, (5, 5), 0), 50, 150)

    plt.figure(figsize=(12, 4))
    for i, (image, title) in enumerate(
        [
            (noisy_square, "Noisy image"),
            (sobel_noisy_mag, "Sobel magnitude"),
            (lap_noisy, "Laplacian"),
            (canny_noisy, "Canny"),
        ],
        start=1,
    ):
        plt.subplot(1, 4, i)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_comparison.png", bbox_inches="tight")
    plt.show()

    print("Observations:")
    print("- Sobel highlights directional changes.")
    print("- Laplacian is more sensitive to noise because it is based on second derivatives.")
    print("- Canny usually produces the thinnest and cleanest edge map when thresholds are tuned well.")


if __name__ == "__main__":
    main()
