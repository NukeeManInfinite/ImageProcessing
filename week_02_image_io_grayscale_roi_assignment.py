"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_02_image_io_grayscale_roi_assignment.py

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

IMAGE_PATH = None
OUTPUT_DIR = Path("week_02_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_image(path=None):
    if path and Path(path).exists():
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return data.astronaut()


def show_side_by_side(img1, img2, title1, title2, cmap1=None, cmap2=None, save_name=None):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap=cmap1)
    plt.title(title1)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap=cmap2)
    plt.title(title2)
    plt.axis("off")
    if save_name:
        plt.savefig(OUTPUT_DIR / save_name, bbox_inches="tight")
    plt.show()


def main():
    img = load_image(IMAGE_PATH)
    h, w = img.shape[:2]
    print("Loaded image shape:", img.shape, "| dtype:", img.dtype)

    print("\n=== Task 2.1: Inspect pixels ===")
    print("Top-left pixel:", img[0, 0])
    print("Center pixel:", img[h // 2, w // 2])
    print("Answer: In an RGB image, the three numbers represent the Red, Green, and Blue channel intensities.")

    print("\n=== Task 2.2: Save and reload ===")
    out_path = OUTPUT_DIR / "output_saved.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    reloaded = cv2.cvtColor(cv2.imread(str(out_path)), cv2.COLOR_BGR2RGB)
    print("Reloaded shape:", reloaded.shape, "| dtype:", reloaded.dtype)
    show_side_by_side(img, reloaded, "Original", "Reloaded", save_name="01_reload_compare.png")

    print("\n=== Task 3.1: Grayscale ===")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    show_side_by_side(img, gray, "RGB", "Grayscale", cmap2="gray", save_name="02_grayscale.png")

    print("\n=== Task 3.2: Binary thresholding ===")
    _, binary_manual = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    otsu_value, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu threshold:", otsu_value)
    show_side_by_side(binary_manual, binary_otsu, "Manual Threshold", "Otsu Threshold", cmap1="gray", cmap2="gray", save_name="03_thresholds.png")
    print("Answer: Manual thresholding uses a fixed value, while Otsu automatically chooses a threshold based on the intensity histogram.")

    print("\n=== Task 4.1: ROI cropping ===")
    x1, y1 = int(0.25 * w), int(0.25 * h)
    x2, y2 = int(0.75 * w), int(0.75 * h)
    roi = img[y1:y2, x1:x2].copy()
    show_side_by_side(img, roi, "Original", f"ROI x[{x1}:{x2}], y[{y1}:{y2}]", save_name="04_roi.png")
    print("ROI shape:", roi.shape)

    print("\n=== Task 4.2: Coordinate explanation ===")
    print("Answer: NumPy stores images row-first, so indexing is image[y, x] where y means row and x means column.")

    print("\n=== Task 5.1: HSV channels ===")
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    plt.figure(figsize=(12, 3))
    for i, (channel, name) in enumerate([(H, "H"), (S, "S"), (V, "V")], start=1):
        plt.subplot(1, 3, i)
        plt.imshow(channel, cmap="gray")
        plt.title(name)
        plt.axis("off")
    plt.savefig(OUTPUT_DIR / "05_hsv_channels.png", bbox_inches="tight")
    plt.show()

    print("\n=== Task 6: Arithmetic operations ===")
    img_sub = cv2.subtract(img, 100)
    img_add = cv2.add(img, 100)

    img_red_sub = img.copy()
    img_red_sub[:, :, 0] = cv2.subtract(img_red_sub[:, :, 0], 80)

    img_red_add = img.copy()
    img_red_add[:, :, 0] = cv2.add(img_red_add[:, :, 0], 80)

    show_side_by_side(img, img_sub, "Original", "Subtract 100", save_name="06_subtract.png")
    show_side_by_side(img, img_add, "Original", "Add 100", save_name="07_add.png")
    show_side_by_side(img, img_red_sub, "Original", "Subtract from Red only", save_name="08_red_subtract.png")
    show_side_by_side(img, img_red_add, "Original", "Add to Red only", save_name="09_red_add.png")

    print("Answer 6.1: Increasing subtraction makes the image darker because channel intensities are reduced toward 0.")
    print("Answer 6.3: Increasing addition makes the image brighter because channel intensities move toward 255.")


if __name__ == "__main__":
    main()
