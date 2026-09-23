import cv2
import numpy as np
import os


# ============================================================
# Configuration
# ============================================================

IMAGE_1_PATH = "../datasets/hpatches-sequences-release/v_woman/1.ppm"
IMAGE_2_PATH = "../datasets/hpatches-sequences-release/v_woman/2.ppm"
HOMOGRAPHY_PATH = "../datasets/hpatches-sequences-release/v_woman/H_1_2"

RESULTS_DIR = "results"

RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 3.0


# ============================================================
# Helper function
# ============================================================

def ensure_results_directory():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Load images and ground-truth homography
# ============================================================

image1 = cv2.imread(IMAGE_1_PATH)
image2 = cv2.imread(IMAGE_2_PATH)

if image1 is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_1_PATH}")

if image2 is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_2_PATH}")

H_ground_truth = np.loadtxt(HOMOGRAPHY_PATH)

print("\nGround-truth homography:")
print(H_ground_truth)


# ============================================================
# SIFT feature detection and description
# ============================================================

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

print(f"\nImage 1 keypoints: {len(keypoints1)}")
print(f"Image 2 keypoints: {len(keypoints2)}")

print(
    f"Descriptor shape - Image 1: "
    f"{descriptors1.shape}"
)

print(
    f"Descriptor shape - Image 2: "
    f"{descriptors2.shape}"
)


# ============================================================
# Descriptor matching
# ============================================================

bf = cv2.BFMatcher(cv2.NORM_L2)

knn_matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)

print(f"\nTotal KNN matches: {len(knn_matches)}")


# ============================================================
# Lowe ratio test
# ============================================================

good_matches = []

for match_pair in knn_matches:

    if len(match_pair) < 2:
        continue

    m, n = match_pair

    if m.distance < RATIO_THRESHOLD * n.distance:
        good_matches.append(m)


print(
    f"Ratio-test threshold: "
    f"{RATIO_THRESHOLD}"
)

print(
    f"Good matches after ratio test: "
    f"{len(good_matches)}"
)


# ============================================================
# Check if enough matches exist
# ============================================================

if len(good_matches) < 4:
    raise RuntimeError(
        "Not enough matches for homography estimation."
    )


# ============================================================
# Extract matched point coordinates
# ============================================================

points1 = np.float32(
    [keypoints1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

points2 = np.float32(
    [keypoints2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)


# ============================================================
# RANSAC homography estimation
# ============================================================

H_ransac, ransac_mask = cv2.findHomography(
    points1,
    points2,
    cv2.RANSAC,
    RANSAC_THRESHOLD
)

if H_ransac is None or ransac_mask is None:
    raise RuntimeError(
        "RANSAC homography estimation failed."
    )


ransac_mask = ransac_mask.ravel().astype(bool)


# ============================================================
# Separate inliers and outliers
# ============================================================

inlier_matches = [
    match
    for match, is_inlier in zip(
        good_matches,
        ransac_mask
    )
    if is_inlier
]

outlier_matches = [
    match
    for match, is_inlier in zip(
        good_matches,
        ransac_mask
    )
    if not is_inlier
]


num_good_matches = len(good_matches)
num_inliers = len(inlier_matches)
num_outliers = len(outlier_matches)

inlier_ratio = (
    num_inliers / num_good_matches * 100
)


# ============================================================
# Ground-truth evaluation of RANSAC inliers
# ============================================================

points1_all = np.float32(
    [keypoints1[m.queryIdx].pt for m in good_matches]
)

points2_all = np.float32(
    [keypoints2[m.trainIdx].pt for m in good_matches]
)


points1_h = np.concatenate(
    [
        points1_all,
        np.ones(
            (points1_all.shape[0], 1),
            dtype=np.float32
        )
    ],
    axis=1
)


projected_points = (
    H_ground_truth @ points1_h.T
).T


projected_points = (
    projected_points[:, :2]
    /
    projected_points[:, 2:3]
)


ground_truth_errors = np.linalg.norm(
    projected_points - points2_all,
    axis=1
)


# ============================================================
# Evaluate RANSAC classification
# ============================================================

GROUND_TRUTH_THRESHOLD = 3.0

ground_truth_correct = (
    ground_truth_errors <= GROUND_TRUTH_THRESHOLD
)

ransac_predicted_inlier = ransac_mask

true_positive = np.sum(
    ransac_predicted_inlier
    & ground_truth_correct
)

false_positive = np.sum(
    ransac_predicted_inlier
    & ~ground_truth_correct
)

true_negative = np.sum(
    ~ransac_predicted_inlier
    & ~ground_truth_correct
)

false_negative = np.sum(
    ~ransac_predicted_inlier
    & ground_truth_correct
)


# ============================================================
# Precision / recall
# ============================================================

precision = (
    true_positive /
    (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0.0
)

recall = (
    true_positive /
    (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0.0
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 60)
print("RANSAC GEOMETRIC VERIFICATION RESULTS")
print("=" * 60)

print(f"\nCandidate matches: {num_good_matches}")
print(f"RANSAC inliers: {num_inliers}")
print(f"RANSAC outliers: {num_outliers}")

print(
    f"RANSAC inlier ratio: "
    f"{inlier_ratio:.2f}%"
)

print(
    f"\nGround-truth threshold: "
    f"{GROUND_TRUTH_THRESHOLD:.1f} pixels"
)

print("\nRANSAC classification against HPatches ground truth:")

print(f"True positives : {true_positive}")
print(f"False positives: {false_positive}")
print(f"True negatives : {true_negative}")
print(f"False negatives: {false_negative}")

print(
    f"\nPrecision: {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall   : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print("\nEstimated homography:")
print(H_ransac)


# ============================================================
# Visualize RANSAC inliers
# ============================================================

inlier_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    os.path.join(
        RESULTS_DIR,
        "task_04_ransac_inliers.png"
    ),
    inlier_image
)


# ============================================================
# Visualize RANSAC outliers
# ============================================================

outlier_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    outlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    os.path.join(
        RESULTS_DIR,
        "task_04_ransac_outliers.png"
    ),
    outlier_image
)


# ============================================================
# Save numerical results
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "task_04_ransac_results.txt"
)

with open(results_path, "w") as file:

    file.write(
        "TASK 4 - RANSAC GEOMETRIC VERIFICATION\n"
    )
    file.write("=" * 60 + "\n\n")

    file.write(
        f"Ratio-test threshold: "
        f"{RATIO_THRESHOLD}\n"
    )

    file.write(
        f"RANSAC threshold: "
        f"{RANSAC_THRESHOLD} pixels\n\n"
    )

    file.write(
        f"Candidate matches: "
        f"{num_good_matches}\n"
    )

    file.write(
        f"RANSAC inliers: "
        f"{num_inliers}\n"
    )

    file.write(
        f"RANSAC outliers: "
        f"{num_outliers}\n"
    )

    file.write(
        f"RANSAC inlier ratio: "
        f"{inlier_ratio:.2f}%\n\n"
    )

    file.write(
        "Ground-truth classification\n"
    )
    file.write("-" * 40 + "\n")

    file.write(
        f"True positives: "
        f"{true_positive}\n"
    )

    file.write(
        f"False positives: "
        f"{false_positive}\n"
    )

    file.write(
        f"True negatives: "
        f"{true_negative}\n"
    )

    file.write(
        f"False negatives: "
        f"{false_negative}\n\n"
    )

    file.write(
        f"Precision: "
        f"{precision:.4f}\n"
    )

    file.write(
        f"Recall: "
        f"{recall:.4f}\n\n"
    )

    file.write(
        "Estimated homography:\n"
    )

    file.write(
        np.array2string(H_ransac)
    )


print("\nResults saved to:")
print(
    os.path.join(
        RESULTS_DIR,
        "task_04_ransac_results.txt"
    )
)

print("\nVisualizations saved to:")
print(
    os.path.join(
        RESULTS_DIR,
        "task_04_ransac_inliers.png"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "task_04_ransac_outliers.png"
    )
)