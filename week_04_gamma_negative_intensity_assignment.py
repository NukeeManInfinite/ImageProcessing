"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_04_gamma_negative_intensity_assignment.py

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

OUTPUT_DIR = Path("week_04_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_gray_image():
    return cv2.cvtColor(data.camera(), cv2.COLOR_GRAY2RGB)


def show_row(images, titles, save_name):
    plt.figure(figsize=(4 * len(images), 4))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, len(images), i)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_name, bbox_inches="tight")
    plt.show()


def gamma_correction(image, gamma):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def image_negative(image):
    return 255 - image


def intensity_range_mapping(image, a, b):
    if a > b:
        raise ValueError("A must be <= B")
    out = image.copy()
    mask = (image >= a) & (image <= b)
    out[mask] = 255
    return out


def main():
    img = load_gray_image()

    print("\n=== Challenge 1.1: gamma_correction() ===")
    corrected = gamma_correction(img, 0.7)
    show_row([img, corrected], ["Original", "Gamma 0.7"], "01_gamma_basic.png")

    print("\n=== Challenge 1.2: Gamma curve analysis ===")
    r = np.linspace(0, 1, 256)
    gammas = [0.3, 0.5, 0.8, 1.5, 2.5]
    plt.figure(figsize=(7, 5))
    for g in gammas:
        plt.plot(r, r ** g, label=f"gamma={g}")
    plt.xlabel("Input intensity r")
    plt.ylabel("Output intensity s")
    plt.title("Gamma Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "02_gamma_curves.png", bbox_inches="tight")
    plt.show()

    print("\n=== Challenge 1.3: Comparative study ===")
    gamma_05 = gamma_correction(img, 0.5)
    gamma_10 = gamma_correction(img, 1.0)
    gamma_20 = gamma_correction(img, 2.0)
    show_row([img, gamma_05, gamma_10, gamma_20], ["Original", "Gamma 0.5", "Gamma 1.0", "Gamma 2.0"], "03_gamma_compare.png")

    print("\n=== Challenge 2.1: Image negative ===")
    negative = image_negative(img)
    show_row([img, negative], ["Original", "Negative"], "04_negative.png")

    print("\n=== Challenge 2.2: Intensity range mapping ===")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mapped = intensity_range_mapping(gray, 80, 150)
    show_row([gray, mapped], ["Original Gray", "Mapped [80, 150] -> 255"], "05_range_mapping.png")

    print("\n=== Challenge 2.3 and Final Task: Integrated enhancement ===")
    gamma_stage = gamma_correction(gray, 0.8)
    range_stage = intensity_range_mapping(gamma_stage, 100, 180)
    final_negative = image_negative(range_stage)
    show_row(
        [gray, gamma_stage, range_stage, final_negative],
        ["Original", "Gamma", "Range Mapping", "Negative of Result"],
        "06_full_pipeline.png",
    )

    print("Done. All required parts are implemented in one reusable script.")


if __name__ == "__main__":
    main()
