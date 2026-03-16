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


def process_and_get_keypoints(image, title, return_kps=False):
    """
    Detect SIFT keypoints, keep the top 30% strongest responses,
    and optionally return the retained keypoints.
    """
    # 1. Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if not keypoints:
        return (image, []) if return_kps else image

    # 2. Filter for top 30% strongest keypoints
    responses = np.array([kp.response for kp in keypoints])
    threshold = np.percentile(responses, 70)

    strong_keypoints = [
        kp for kp in keypoints if kp.response >= threshold
    ]

    print(
        f"[{title}] Total detected: {len(keypoints)} | "
        f"Strong (Top 30%): {len(strong_keypoints)}"
    )
    # 3. Draw selected keypoints
    img_kp = cv2.drawKeypoints(
        image,
        strong_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    if return_kps:
        return img_kp, strong_keypoints

    return img_kp


# PART 1: Standard Analysis Grid

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
    axes[idx, 0].imshow(
        process_and_get_keypoints(img, f"{filename} - Original")
    )
    axes[idx, 0].set_title(f"{filename}\nOriginal")

    # Blurred image - gaussian blur
    img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    axes[idx, 1].imshow(
        process_and_get_keypoints(img_blur, f"{filename} - Blur")
    )
    axes[idx, 1].set_title("Gaussian Blur")

    # Histogram equalization
    img_eq = cv2.equalizeHist(img)
    axes[idx, 2].imshow(
        process_and_get_keypoints(img_eq, f"{filename} - Equalized")
    )
    axes[idx, 2].set_title("Histogram Equalization")

    for ax in axes[idx]:
        ax.axis("off")

plt.tight_layout()
plt.show()

# PART 2: Real-World Robustness Check (Bag 1 vs Bag 2)

print("\n" + "=" * 50)
print("ROBUSTNESS CHECK: Viewpoint Invariance (Bag 1 vs Bag 2)")
print("=" * 50)

img1 = cv2.imread("app1_bag1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("app1_bag2.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is not None and img2 is not None:
    vis1, kps1 = process_and_get_keypoints(
        img1, "Bag View 1", return_kps=True
    )
    vis2, kps2 = process_and_get_keypoints(
        img2, "Bag View 2", return_kps=True
    )

    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 7))

    ax2[0].imshow(vis1)
    ax2[0].set_title(
        f"Bag 1 (Original View)\nStrong Keypoints: {len(kps1)}"
    )

    ax2[1].imshow(vis2)
    ax2[1].set_title(
        f"Bag 2 (New Angle)\nStrong Keypoints: {len(kps2)}"
    )

    for ax in ax2:
        ax.axis("off")

    plt.tight_layout()
    plt.show()