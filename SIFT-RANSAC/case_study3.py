import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
random.seed(42)  # For reproducibility

img1_path = "app1_keyboard.jpg"
img2_path = "app3_keyboard1.jpg"

reproj_thresholds = [1, 3, 5, 10]
current_threshold = 5.0
lowe_ratio = 0.75

# --------------------------------------------------
# 1. LOAD IMAGES
# --------------------------------------------------
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise ValueError("Images not found. Check file paths.")

# --------------------------------------------------
# 2. SIFT DETECTION
# --------------------------------------------------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# --------------------------------------------------
# 3. MATCHING (Lowe's Ratio Test)
# --------------------------------------------------
bf = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf.knnMatch(des1, des2, k=2)

print(f"Total matches BEFORE Lowe's Ratio (KNN): {len(knn_matches)}")

good_matches = []
for m, n in knn_matches:
    if m.distance < lowe_ratio * n.distance:
        good_matches.append(m)

print(f"Total matches AFTER Lowe's Ratio: {len(good_matches)}")

# --------------------------------------------------
# 4. RANSAC FUNCTION
# --------------------------------------------------
def apply_ransac(matches_list, threshold):
    if len(matches_list) < 4:
        return None, [], 0

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches_list]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches_list]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, threshold
    )

    if H is None:
        return None, [], 0

    mask = mask.ravel()
    inliers = [m for i, m in enumerate(matches_list) if mask[i]]

    return H, inliers, len(inliers)

# --------------------------------------------------
# 5. TASK 2 – RANSAC THRESHOLD ANALYSIS
# --------------------------------------------------
print("\n--- Task 2: Influence of Reprojection Threshold ---")
for th in reproj_thresholds:
    _, _, n_inliers = apply_ransac(good_matches, th)
    print(f"Threshold {th}: {n_inliers} inliers")

# --------------------------------------------------
# 6. TASK 3 – INLIER RATIO
# --------------------------------------------------
H, inliers, num_inliers = apply_ransac(good_matches, current_threshold)
inlier_ratio = num_inliers / len(good_matches) if good_matches else 0

print("\n--- Task 3: Inlier Ratio ---")
print(f"Inlier Ratio: {inlier_ratio:.3f} ({num_inliers}/{len(good_matches)})")

# --------------------------------------------------
# 7. TASK 6 – ARTIFICIAL MATCH REDUCTION
# --------------------------------------------------
print("\n--- Task 6: Match Reduction Stability Test ---")
percentages = [0.30, 0.10]

for p in percentages:
    sample_size = int(len(good_matches) * p)

    if sample_size < 4:
        print(f"Keep {int(p*100)}%: Too few matches ({sample_size})")
        continue

    reduced_matches = random.sample(good_matches, sample_size)
    _, _, sub_inliers = apply_ransac(reduced_matches, current_threshold)
    ratio = sub_inliers / sample_size

    print(f"Keep {int(p*100)}% ({sample_size} matches): "
          f"{sub_inliers} inliers (ratio = {ratio:.3f})")

# --------------------------------------------------
# 8. VISUALIZATION
# --------------------------------------------------
img_lowe = cv2.drawMatches(
    img1, kp1, img2, kp2,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

img_ransac = cv2.drawMatches(
    img1, kp1, img2, kp2,
    inliers, None,
    matchColor=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(20, 10))

plt.subplot(2, 1, 1)
plt.imshow(img_lowe, cmap='gray')
plt.title(f"Before RANSAC (Lowe Only): {len(good_matches)} Matches")
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(img_ransac, cmap='gray')
plt.title(f"After RANSAC (Inliers): {len(inliers)} Matches")
plt.axis('off')

plt.tight_layout()
plt.show()
