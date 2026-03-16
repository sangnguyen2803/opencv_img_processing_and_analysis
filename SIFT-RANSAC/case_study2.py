import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load images
# --------------------------------------------------
# Test cases: 
# Case A: Same object, different view (watch1 vs watch2)
# Resources: watch1.jpg, watch2.jpg
# Case B: Repetitive pattern (glasses vs glasses_rotated or same)
# Resources: glasses.jpg
# Case C: Different objects (watch vs glasses)

img1_path = "./images/window1.jpg"
img2_path = "./images/window2.jpg"

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

assert img1 is not None and img2 is not None, "Verify image paths!"

# Part 1: AUTOMATED SENSITIVITY TEST
print("\n--- PARAMETER SENSITIVITY REPORT ---")

# Test 1: adjust contrast threshold (default 0.04)
contrasts = [0.01, 0.04, 0.10] 
for c in contrasts:
    s = cv2.SIFT_create(contrastThreshold=c) 
    k = s.detect(img1, None)
    print(f"Contrast {c}: {len(k)} keypoints detected.")

# Test 2: adjust edge threshold (default 10)
edges = [5, 10, 20]
for e in edges:
    s = cv2.SIFT_create(edgeThreshold=e)
    k = s.detect(img1, None)
    print(f"EdgeThresh {e}: {len(k)} keypoints detected.")

# Test 3: adjust sigma (default 1.6)
sigmas = [1.0, 1.6, 3.0]
for sig in sigmas:
    s = cv2.SIFT_create(sigma=sig)
    k = s.detect(img1, None)
    print(f"Sigma {sig}: {len(k)} keypoints detected.")
print("------------------------------------\n")

# SIFT configuration
sift = cv2.SIFT_create(
    contrastThreshold=0.04, 
    edgeThreshold=10, 
    sigma=1.6
)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print(f"Keypoints in Image 1: {len(kp1)}")
print(f"Keypoints in Image 2: {len(kp2)}")

# Matcher setup
# SIFT descriptors are L2-norm based. 
bf = cv2.BFMatcher(cv2.NORM_L2)





# Part 2: NAÏVE MATCHING (k=1)
naive_matches = bf.match(des1, des2)
naive_matches = sorted(naive_matches, key=lambda x: x.distance)

img_naive = cv2.drawMatches(img1, kp1, img2, kp2, naive_matches[:50], None, 
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# ==================================================
# NEW REPORT: NAÏVE MATCHING ANALYSIS
# ==================================================
print("\n--- NAÏVE MATCHING ANALYSIS REPORT ---")

# 1. Number of matches
print(f"• Number of matches found: {len(naive_matches)}")

# 2. Visual Coherence
# We calculate the average distance of the matches to give a numeric hint about coherence (lower distance = generally more similar features).
avg_dist = np.mean([m.distance for m in naive_matches])
print(f"• Visual Coherence Metric (Avg Distance): {avg_dist:.2f}")

# 3. Presence of clearly incorrect correspondences
print("• Presence of clearly incorrect correspondences:")
print(f"  - Best Match Distance: {naive_matches[0].distance:.2f}")
print(f"  - Worst Match Distance: {naive_matches[-1].distance:.2f}")
print("------------------------------------------\n")



# 5. TASK 2: LOWE'S RATIO TEST (k=2)

# We find the 2 nearest neighbors for each descriptor
raw_matches = bf.knnMatch(des1, des2, k=2)

def apply_ratio_test(matches, ratio_threshold):
    good_matches = []
    for m, n in matches:
        # m is the closest, n is the second closest
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return good_matches

# Testing different ratios as per instructions
ratio_values = [0.6, 0.75, 0.9]
ratio_results = {}

for r in ratio_values:
    good = apply_ratio_test(raw_matches, r)
    ratio_results[r] = good
    print(f"Ratio {r}: {len(good)} matches retained.")

# Visualize the standard 0.75 ratio
img_ratio = cv2.drawMatches(img1, kp1, img2, kp2, ratio_results[0.75][:50], None, 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 6. VISUALIZATION
plt.figure(figsize=(20, 10))

plt.subplot(2, 1, 1)
plt.imshow(img_naive)
plt.title(f"Naïve Matching (k=1) - Top 50 matches")
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(img_ratio)
plt.title(f"Lowe's Ratio Test (Ratio=0.75) - Top 50 matches")
plt.axis('off')

plt.tight_layout()
plt.show()