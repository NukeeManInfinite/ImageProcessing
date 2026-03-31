"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_03_masking_arithmetic_gamma_assignment.py

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

OUTPUT_DIR = Path("week_03_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def show(img, title, cmap=None, save_name=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    if save_name:
        plt.savefig(OUTPUT_DIR / save_name, bbox_inches="tight")
    plt.show()


def gamma_correction(image, gamma):
    image_float = image.astype(np.float32) / 255.0
    corrected = np.power(image_float, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def main():
    img = data.astronaut()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    print("\n=== Alpha channel demo ===")
    alpha = np.full(gray.shape, 180, dtype=np.uint8)
    rgba = np.dstack([img, alpha])
    print("RGBA shape:", rgba.shape)
    show(rgba, "RGB + Alpha", save_name="01_rgba.png")

    print("\n=== Bitwise masking ===")
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (gray.shape[1] // 2, gray.shape[0] // 2), 120, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    show(mask, "Circular Mask", cmap="gray", save_name="02_mask.png")
    show(masked, "Bitwise AND with Mask", save_name="03_masked.png")

    print("\n=== Drawing and annotation ===")
    annotated = img.copy()
    cv2.rectangle(annotated, (60, 60), (220, 220), (255, 0, 0), 3)
    cv2.circle(annotated, (300, 170), 60, (0, 255, 0), 3)
    cv2.putText(annotated, "Week 03", (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
    show(annotated, "Drawing and Annotation", save_name="04_annotated.png")

    print("\n=== Arithmetic operations ===")
    added = cv2.add(img, 40)
    subtracted = cv2.subtract(img, 40)
    multiplied = cv2.multiply(img.astype(np.float32), 1.2)
    multiplied = np.clip(multiplied, 0, 255).astype(np.uint8)
    divided = cv2.divide(img.astype(np.float32), 1.2)
    divided = np.clip(divided, 0, 255).astype(np.uint8)

    show(added, "Added +40", save_name="05_added.png")
    show(subtracted, "Subtracted -40", save_name="06_subtracted.png")
    show(multiplied, "Multiplied x1.2", save_name="07_multiplied.png")
    show(divided, "Divided /1.2", save_name="08_divided.png")

    print("\n=== Linear brightness/contrast adjustment ===")
    alpha_gain = 1.4
    beta_shift = 20
    adjusted = cv2.convertScaleAbs(img, alpha=alpha_gain, beta=beta_shift)
    show(adjusted, f"Contrast alpha={alpha_gain}, brightness beta={beta_shift}", save_name="09_linear_adjustment.png")

    print("\n=== Gamma correction ===")
    gamma_half = gamma_correction(img, 0.5)
    gamma_two = gamma_correction(img, 2.0)
    show(gamma_half, "Gamma 0.5", save_name="10_gamma_05.png")
    show(gamma_two, "Gamma 2.0", save_name="11_gamma_20.png")


if __name__ == "__main__":
    main()
