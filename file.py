

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.datasets import mnist
import os

# Create output folder
os.makedirs("results", exist_ok=True)

# Load MNIST dataset
(x_train, y_train), _ = mnist.load_data()
images = x_train[:5]

# ---------- Preprocessing Functions ----------

def normalize(img):
    return img.astype("float32") / 255.0

def standardize(img):
    img_f = img.astype("float32") / 255.0
    return (img_f - img_f.mean()) / (img_f.std() + 1e-6)

def histogram_equalization(img):
    pil = Image.fromarray(img)
    return np.array(ImageOps.equalize(pil))

def gaussian_noise(img, sigma=25):
    noisy = img + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def geometric_augmentation(img, angle=25, shift=(3, 3)):
    pil = Image.fromarray(img)
    rotated = pil.rotate(angle)
    shifted = Image.new("L", pil.size)
    shifted.paste(rotated, shift)
    return np.array(shifted)

# ---------- Apply Techniques ----------

techniques = {
    "Normalization": lambda img: (normalize(img) * 255).astype(np.uint8),
    "Standardization": lambda img: (
        (standardize(img) - standardize(img).min())
        / (standardize(img).max() - standardize(img).min() + 1e-6)
        * 255
    ).astype(np.uint8),
    "Histogram Equalization": histogram_equalization,
    "Gaussian Noise": gaussian_noise,
    "Geometric Augmentation": geometric_augmentation
}

# ---------- Save Before/After Comparisons ----------

for name, func in techniques.items():
    before_imgs = images
    after_imgs = np.array([func(img) for img in images])

    plt.figure(figsize=(10, 4))
    for i in range(5):
        # before
        plt.subplot(2, 5, i + 1)
        plt.imshow(before_imgs[i], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # after
        plt.subplot(2, 5, i + 6)
        plt.imshow(after_imgs[i], cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.suptitle(name, fontsize=14)
    plt.savefig(f"results/{name}.png")
    plt.close()

print("All preprocessing technique images saved in /results folder.")
print("Upload this script & results folder to your GitHub repository.")
