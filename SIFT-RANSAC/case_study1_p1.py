import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration
# Paths of input images (must match files in your folder)
IMAGE_FILES = [
    "./images/app1_bag1.jpg",
    "./images/app1_bag2.jpg",
    "./images/app1_keyboard.jpg",
]

# Create SIFT detector - SIFT parameters (Lowe defaults)
sift = cv2.SIFT_create(
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6,
)


def process_and_get_keypoints(image, title):
    """
    Detect SIFT keypoints, keep the top 30% strongest responses,
    and return an image with visualized keypoints.
    """
    # 1. Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if not keypoints:
        print(f"[{title}] No keypoints found.")
        return image

    # 2. Filter weak extrema (keep top 30%)
    responses = np.array([kp.response for kp in keypoints])

    if responses.size > 0:
        threshold = np.percentile(responses, 70)
        strong_keypoints = [
            kp for kp in keypoints if kp.response >= threshold
        ]
    else:
        strong_keypoints = []

    print(
        f"[{title}] Total keypoints: {len(keypoints)} | "
        f"Strong retained: {len(strong_keypoints)}"
    )

    # 3. Draw selected keypoints
    img_kp = cv2.drawKeypoints(
        image,
        strong_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return img_kp


# Main
rows = len(IMAGE_FILES)
cols = 3  # Original, Blur, Equalized

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

# Ensure axes is always 2D
if rows == 1:
    axes = np.array([axes])

for idx, filename in enumerate(IMAGE_FILES):
    # Read image in grayscale
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: {filename} not found. Skipping.")
        continue

    # Original image
    vis_original = process_and_get_keypoints(
        img, f"{filename} - Original"
    )
    axes[idx, 0].imshow(vis_original)
    axes[idx, 0].set_title(f"{filename}\nOriginal")
    axes[idx, 0].axis("off")

    # Blurred image - gaussian blur
    img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    vis_blur = process_and_get_keypoints(
        img_blur, f"{filename} - Blur"
    )
    axes[idx, 1].imshow(vis_blur)
    axes[idx, 1].set_title("Gaussian Blur\n(Less small details)")
    axes[idx, 1].axis("off")

    # Histogram equalization
    img_eq = cv2.equalizeHist(img)
    vis_eq = process_and_get_keypoints(
        img_eq, f"{filename} - Equalized"
    )
    axes[idx, 2].imshow(vis_eq)
    axes[idx, 2].set_title("Histogram Equalization\n(High Contrast)")
    axes[idx, 2].axis("off")

print("Processing complete. Displaying results grid...")

plt.tight_layout()
plt.show()