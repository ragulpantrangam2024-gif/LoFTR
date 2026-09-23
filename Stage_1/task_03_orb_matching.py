from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "hpatches-sequences-release"
    / "v_woman"
)

IMAGE1_PATH = DATASET_DIR / "1.ppm"
IMAGE2_PATH = DATASET_DIR / "2.ppm"
H_PATH = DATASET_DIR / "H_1_2"

RESULTS_DIR = PROJECT_ROOT / "Stage_1" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD IMAGES AND GROUND-TRUTH HOMOGRAPHY
# ============================================================

image1 = cv2.imread(
    str(IMAGE1_PATH),
    cv2.IMREAD_GRAYSCALE
)

image2 = cv2.imread(
    str(IMAGE2_PATH),
    cv2.IMREAD_GRAYSCALE
)

if image1 is None:
    raise FileNotFoundError(
        f"Could not load: {IMAGE1_PATH}"
    )

if image2 is None:
    raise FileNotFoundError(
        f"Could not load: {IMAGE2_PATH}"
    )

H = np.loadtxt(H_PATH)


print("=" * 60)
print("HPATCHES ORB MATCHING")
print("=" * 60)

print("\nImage 1:")
print(f"  Path: {IMAGE1_PATH}")
print(
    f"  Size: "
    f"{image1.shape[1]} x {image1.shape[0]}"
)

print("\nImage 2:")
print(f"  Path: {IMAGE2_PATH}")
print(
    f"  Size: "
    f"{image2.shape[1]} x {image2.shape[0]}"
)

print("\nGround-truth homography H_1_2:")
print(H)


# ============================================================
# 3. ORB FEATURE DETECTION AND DESCRIPTION
# ============================================================

# ORB uses FAST-based keypoint detection and
# a rotated binary BRIEF-style descriptor.

orb = cv2.ORB_create(
    nfeatures=5000
)

keypoints1, descriptors1 = orb.detectAndCompute(
    image1,
    None
)

keypoints2, descriptors2 = orb.detectAndCompute(
    image2,
    None
)


print("\n" + "=" * 60)
print("ORB FEATURES")
print("=" * 60)

print(
    f"Image 1 keypoints: {len(keypoints1)}"
)

print(
    f"Image 2 keypoints: {len(keypoints2)}"
)

print(
    f"Descriptor shape - Image 1: "
    f"{descriptors1.shape if descriptors1 is not None else None}"
)

print(
    f"Descriptor shape - Image 2: "
    f"{descriptors2.shape if descriptors2 is not None else None}"
)


# ============================================================
# 4. VISUALIZE ORB KEYPOINTS
# ============================================================

keypoints_image1 = cv2.drawKeypoints(
    image1,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoints_image2 = cv2.drawKeypoints(
    image2,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite(
    str(
        RESULTS_DIR
        / "task_03_orb_keypoints_image1.png"
    ),
    keypoints_image1
)

cv2.imwrite(
    str(
        RESULTS_DIR
        / "task_03_orb_keypoints_image2.png"
    ),
    keypoints_image2
)


# ============================================================
# 5. DESCRIPTOR MATCHING
# ============================================================

# ORB descriptors are binary.
#
# Therefore we use Hamming distance rather than
# the L2 distance used for SIFT.

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

knn_matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


print("\n" + "=" * 60)
print("DESCRIPTOR MATCHING")
print("=" * 60)

print(
    f"Total KNN matches: "
    f"{len(knn_matches)}"
)


# ============================================================
# 6. LOWE RATIO TEST
# ============================================================

ratio_threshold = 0.75

good_matches = []

for match_pair in knn_matches:

    if len(match_pair) < 2:
        continue

    best_match, second_best_match = match_pair

    if (
        best_match.distance
        < ratio_threshold * second_best_match.distance
    ):
        good_matches.append(best_match)


print(
    f"Ratio-test threshold: "
    f"{ratio_threshold}"
)

print(
    f"Good matches after ratio test: "
    f"{len(good_matches)}"
)


# ============================================================
# 7. VISUALIZE GOOD ORB MATCHES
# ============================================================

good_match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(
        RESULTS_DIR
        / "task_03_orb_good_matches.png"
    ),
    good_match_image
)


# ============================================================
# 8. EXTRACT MATCHED POINTS
# ============================================================

if len(good_matches) == 0:
    raise RuntimeError(
        "No good ORB matches found."
    )

points1 = np.float32(
    [
        keypoints1[m.queryIdx].pt
        for m in good_matches
    ]
)

points2 = np.float32(
    [
        keypoints2[m.trainIdx].pt
        for m in good_matches
    ]
)


# ============================================================
# 9. PROJECT IMAGE-1 POINTS USING H_1_2
# ============================================================

# Convert Image-1 points into homogeneous coordinates:
#
# [x, y] -> [x, y, 1]
#
# Then:
#
# p2 = H_1_2 * p1
#
# Finally normalize by the third homogeneous coordinate.

points1_h = np.hstack(
    [
        points1,
        np.ones(
            (len(points1), 1),
            dtype=np.float32
        )
    ]
)

projected_points = (
    H @ points1_h.T
).T

projected_points = (
    projected_points[:, :2]
    / projected_points[:, 2:3]
)


# ============================================================
# 10. COMPUTE GEOMETRIC ERROR
# ============================================================

# Compare:
#
# ORB matched point in Image 2
#
#              VS
#
# Ground-truth point predicted using H_1_2

errors = np.linalg.norm(
    points2 - projected_points,
    axis=1
)


# ============================================================
# 11. ERROR STATISTICS
# ============================================================

mean_error = np.mean(errors)
median_error = np.median(errors)
min_error = np.min(errors)
max_error = np.max(errors)


# Ground-truth consistency threshold

error_threshold = 3.0

inlier_mask = errors <= error_threshold

num_inliers = np.sum(inlier_mask)

inlier_percentage = (
    100.0
    * num_inliers
    / len(errors)
)


# ============================================================
# 12. ERROR DISTRIBUTION
# ============================================================

range_0_1 = np.sum(
    (errors >= 0)
    & (errors < 1)
)

range_1_3 = np.sum(
    (errors >= 1)
    & (errors < 3)
)

range_3_5 = np.sum(
    (errors >= 3)
    & (errors < 5)
)

range_5_10 = np.sum(
    (errors >= 5)
    & (errors < 10)
)

range_over_10 = np.sum(
    errors >= 10
)


# ============================================================
# 13. SEPARATE CONSISTENT MATCHES AND OUTLIERS
# ============================================================

inlier_matches = [
    match
    for match, is_inlier in zip(
        good_matches,
        inlier_mask
    )
    if is_inlier
]

outlier_matches = [
    match
    for match, is_inlier in zip(
        good_matches,
        inlier_mask
    )
    if not is_inlier
]


# ============================================================
# 14. VISUALIZE GROUND-TRUTH-CONSISTENT MATCHES
# ============================================================

inlier_match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(
        RESULTS_DIR
        / "task_03_orb_ground_truth_consistent_matches.png"
    ),
    inlier_match_image
)


# ============================================================
# 15. VISUALIZE GEOMETRIC OUTLIERS
# ============================================================

outlier_match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    outlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(
        RESULTS_DIR
        / "task_03_orb_geometric_outliers.png"
    ),
    outlier_match_image
)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ORB GEOMETRIC EVALUATION")
print("=" * 60)

print(
    f"Number of good matches : "
    f"{len(good_matches)}"
)

print("\nGeometric error statistics:")

print(
    f"Mean error             : "
    f"{mean_error:.4f} pixels"
)

print(
    f"Median error           : "
    f"{median_error:.4f} pixels"
)

print(
    f"Minimum error          : "
    f"{min_error:.4f} pixels"
)

print(
    f"Maximum error          : "
    f"{max_error:.4f} pixels"
)

print("\nGround-truth consistency:")

print(
    f"Matches within "
    f"{error_threshold:.1f} pixels: "
    f"{num_inliers}/{len(good_matches)}"
)

print(
    f"Ground-truth-consistent "
    f"match rate: "
    f"{inlier_percentage:.2f}%"
)

print("\nError distribution:")

print(
    f"0 - 1 px     : "
    f"{range_0_1}"
)

print(
    f"1 - 3 px     : "
    f"{range_1_3}"
)

print(
    f"3 - 5 px     : "
    f"{range_3_5}"
)

print(
    f"5 - 10 px    : "
    f"{range_5_10}"
)

print(
    f"> 10 px      : "
    f"{range_over_10}"
)


# ============================================================
# 17. SAVE TEXT RESULTS
# ============================================================

results_file = (
    RESULTS_DIR
    / "task_03_orb_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ORB MATCHING - HPATCHES v_woman\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write("Dataset\n")
    file.write("-" * 50 + "\n")

    file.write(
        "Sequence: v_woman\n"
    )

    file.write(
        "Image 1: 1.ppm\n"
    )

    file.write(
        "Image 2: 2.ppm\n"
    )

    file.write(
        "Ground truth: H_1_2\n\n"
    )

    file.write("ORB Features\n")
    file.write("-" * 50 + "\n")

    file.write(
        f"Image 1 keypoints: "
        f"{len(keypoints1)}\n"
    )

    file.write(
        f"Image 2 keypoints: "
        f"{len(keypoints2)}\n"
    )

    file.write(
        f"Image 1 descriptor shape: "
        f"{descriptors1.shape}\n"
    )

    file.write(
        f"Image 2 descriptor shape: "
        f"{descriptors2.shape}\n\n"
    )

    file.write("Matching\n")
    file.write("-" * 50 + "\n")

    file.write(
        f"KNN matches: "
        f"{len(knn_matches)}\n"
    )

    file.write(
        f"Lowe ratio threshold: "
        f"{ratio_threshold}\n"
    )

    file.write(
        f"Good matches: "
        f"{len(good_matches)}\n\n"
    )

    file.write(
        "Ground-Truth Evaluation\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    file.write(
        "Evaluation formula:\n"
    )

    file.write(
        "e_i = ||p2_i - H_1_2 p1_i||_2\n\n"
    )

    file.write(
        f"Mean error: "
        f"{mean_error:.4f} pixels\n"
    )

    file.write(
        f"Median error: "
        f"{median_error:.4f} pixels\n"
    )

    file.write(
        f"Minimum error: "
        f"{min_error:.4f} pixels\n"
    )

    file.write(
        f"Maximum error: "
        f"{max_error:.4f} pixels\n\n"
    )

    file.write(
        f"Evaluation threshold: "
        f"{error_threshold:.1f} pixels\n"
    )

    file.write(
        f"Ground-truth-consistent matches: "
        f"{num_inliers}\n"
    )

    file.write(
        f"Ground-truth-inconsistent matches: "
        f"{len(outlier_matches)}\n"
    )

    file.write(
        f"Ground-truth-consistent match rate: "
        f"{inlier_percentage:.2f}%\n\n"
    )

    file.write(
        "Error Distribution\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    file.write(
        f"0 - 1 px: "
        f"{range_0_1}\n"
    )

    file.write(
        f"1 - 3 px: "
        f"{range_1_3}\n"
    )

    file.write(
        f"3 - 5 px: "
        f"{range_3_5}\n"
    )

    file.write(
        f"5 - 10 px: "
        f"{range_5_10}\n"
    )

    file.write(
        f"> 10 px: "
        f"{range_over_10}\n"
    )


# ============================================================
# 18. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)

print(
    "Results directory:"
)

print(RESULTS_DIR)

print("\nGenerated files:")

print(
    "  task_03_orb_keypoints_image1.png"
)

print(
    "  task_03_orb_keypoints_image2.png"
)

print(
    "  task_03_orb_good_matches.png"
)

print(
    "  task_03_orb_ground_truth_consistent_matches.png"
)

print(
    "  task_03_orb_geometric_outliers.png"
)

print(
    "  task_03_orb_results.txt"
)

print("\nTask 3 completed.")