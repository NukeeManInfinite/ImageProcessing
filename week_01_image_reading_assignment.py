"""
Self-contained Digital Image Processing assignment solution.

How to run:
    python week_01_image_reading_assignment.py

Optional:
    Put your own image in the same folder and set IMAGE_PATH below.
    If IMAGE_PATH is None or the file is missing, the script uses a built-in sample image.

Libraries:
    pip install opencv-python matplotlib scikit-image pillow
"""

import glob
from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import img_as_float, io, data

IMAGE_PATH = None  # e.g. "test_image.jpg"
OUTPUT_DIR = Path("week_01_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_image_path():
    if IMAGE_PATH and Path(IMAGE_PATH).exists():
        return IMAGE_PATH
    fallback = OUTPUT_DIR / "sample_image.jpg"
    if not fallback.exists():
        sample_rgb = data.astronaut()
        sample_bgr = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fallback), sample_bgr)
    return str(fallback)


def show(title, image, cmap=None, save_name=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    if save_name:
        plt.savefig(OUTPUT_DIR / save_name, bbox_inches="tight")
    plt.show()


def task_pil(image_path):
    print("\n=== TODO 1: PIL ===")
    pil_img = Image.open(image_path)
    print("Type:", type(pil_img))
    print("Size:", pil_img.size)
    print("Mode:", pil_img.mode)
    show("PIL Image", pil_img, save_name="01_pil.png")
    return pil_img


def task_pil_to_numpy(pil_img):
    print("\n=== TODO 2: PIL -> NumPy ===")
    arr = np.array(pil_img)
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    return arr


def task_matplotlib_read(image_path):
    print("\n=== TODO 3: Matplotlib ===")
    img = mpimg.imread(image_path)
    print("Shape:", img.shape)
    print("Dtype:", img.dtype)
    show("Matplotlib imread", img, save_name="02_matplotlib.png")
    print("Answer: Matplotlib treats images as NumPy arrays because plotting and numerical operations are easier on array data.")
    return img


def task_skimage_read(image_path):
    print("\n=== TODO 4: scikit-image ===")
    img = io.imread(image_path)
    img_float = img_as_float(img)
    print("Min pixel value:", float(img_float.min()))
    print("Max pixel value:", float(img_float.max()))
    show("scikit-image float image", img_float, save_name="03_skimage.png")
    print("Answer: img_as_float rescales image values correctly based on the original dtype, while astype(float) only changes type and may keep wrong ranges.")
    return img


def task_opencv(image_path):
    print("\n=== TODO 5: OpenCV ===")
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    print("Gray shape:", gray.shape, "| dtype:", gray.dtype)
    print("Color shape:", color_rgb.shape, "| dtype:", color_rgb.dtype)
    show("OpenCV RGB", color_rgb, save_name="04_opencv_rgb.png")
    return gray, color_rgb


def task_canny(gray):
    print("\n=== TODO 6: Canny edge detection ===")
    edges = cv2.Canny(gray, 100, 200)
    show("Canny Edges", edges, cmap="gray", save_name="05_canny.png")
    return edges


def bonus_read_multiple_images():
    print("\n=== BONUS: Read multiple images in a folder ===")
    paths = glob.glob(str(OUTPUT_DIR / "*.*"))
    for file in paths[:5]:
        img = cv2.imread(file)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        show(f"Bonus: {Path(file).name}", rgb)


def main():
    image_path = get_image_path()
    print("Using image:", image_path)

    pil_img = task_pil(image_path)
    _ = task_pil_to_numpy(pil_img)
    _ = task_matplotlib_read(image_path)
    _ = task_skimage_read(image_path)
    gray, _ = task_opencv(image_path)
    _ = task_canny(gray)

    print("\n=== Reflection answers ===")
    print("1. Easiest library: PIL for quick loading and metadata inspection.")
    print("2. Best for real-time video: OpenCV.")
    print("3. RGB stores channels as Red-Green-Blue, while OpenCV often loads color images as BGR.")

    bonus_read_multiple_images()


if __name__ == "__main__":
    main()
