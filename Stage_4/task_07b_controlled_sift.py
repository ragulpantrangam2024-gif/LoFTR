import os
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release"
)

SEQUENCE_DIR = os.path.join(
    DATASET_DIR,
    "v_soldiers"
)

IMAGE1_PATH = os.path.join(
    SEQUENCE_DIR,
    "1.ppm"
)

IMAGE2_PATH = os.path.join(
    SEQUENCE_DIR,
    "2.ppm"
)

H_PATH = os.path.join(
    SEQUENCE_DIR,
    "H_1_2"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_07b"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

RATIO_THRESHOLD = 0.75

ERROR_THRESHOLDS = [1.0, 3.0, 5.0, 10.0]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TASK 7B - CONTROLLED SIFT BASELINE")
print("=" * 70)

print("\nSequence: v_soldiers")
print("Image pair: 1.ppm -> 2.ppm")
print("Ground truth: H_1_2")

img1 = cv2.imread(IMAGE1_PATH, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(IMAGE2_PATH, cv2.IMREAD_GRAYSCALE)

if img1 is None:
    raise FileNotFoundError(
        f"Could not load image 1: {IMAGE1_PATH}"
    )

if img2 is None:
    raise FileNotFoundError(
        f"Could not load image 2: {IMAGE2_PATH}"
    )

H = np.loadtxt(H_PATH)

print("\nImage 1 shape:", img1.shape)
print("Image 2 shape:", img2.shape)

print("\nGround-truth homography:")
print(H)


# ============================================================
# SIFT DETECTION AND DESCRIPTION
# ============================================================

print("\n" + "-" * 70)
print("SIFT FEATURE DETECTION")
print("-" * 70)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(
    img1,
    None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    img2,
    None
)

print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))

if descriptors1 is None or descriptors2 is None:
    raise RuntimeError(
        "SIFT descriptors could not be computed."
    )

print("Image 1 descriptor shape:", descriptors1.shape)
print("Image 2 descriptor shape:", descriptors2.shape)


# ============================================================
# KNN MATCHING
# ============================================================

print("\n" + "-" * 70)
print("SIFT KNN MATCHING")
print("-" * 70)

bf = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False
)

knn_matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)

print("Total KNN matches:", len(knn_matches))


# ============================================================
# LOWE RATIO TEST
# ============================================================

good_matches = []

for pair in knn_matches:

    if len(pair) < 2:
        continue

    m, n = pair

    if m.distance < RATIO_THRESHOLD * n.distance:
        good_matches.append(m)


print(
    f"Good matches after Lowe ratio test "
    f"({RATIO_THRESHOLD}): {len(good_matches)}"
)


if len(good_matches) == 0:
    raise RuntimeError(
        "No good matches found after ratio test."
    )


# ============================================================
# EXTRACT MATCH POINTS
# ============================================================

pts1 = np.float32([
    keypoints1[m.queryIdx].pt
    for m in good_matches
])

pts2 = np.float32([
    keypoints2[m.trainIdx].pt
    for m in good_matches
])


# ============================================================
# APPLY GROUND-TRUTH HOMOGRAPHY
# ============================================================

pts1_h = cv2.convertPointsToHomogeneous(
    pts1
).reshape(-1, 3)

projected_pts2_h = (
    H @ pts1_h.T
).T

projected_pts2 = (
    projected_pts2_h[:, :2]
    /
    projected_pts2_h[:, 2:3]
)


# ============================================================
# GEOMETRIC ERROR
# ============================================================

errors = np.linalg.norm(
    projected_pts2 - pts2,
    axis=1
)

print("\n" + "-" * 70)
print("GEOMETRIC EVALUATION")
print("-" * 70)

print(f"Mean error:   {np.mean(errors):.4f} px")
print(f"Median error: {np.median(errors):.4f} px")
print(f"Minimum error:{np.min(errors):.4f} px")
print(f"Maximum error:{np.max(errors):.4f} px")


# ============================================================
# THRESHOLD STATISTICS
# ============================================================

threshold_counts = {}

for threshold in ERROR_THRESHOLDS:

    count = int(
        np.sum(errors <= threshold)
    )

    percentage = (
        count / len(errors)
    ) * 100.0

    threshold_counts[threshold] = (
        count,
        percentage
    )

    print(
        f"Within {threshold:.0f} px: "
        f"{count}/{len(errors)} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SAVE MATCH VISUALIZATION
# ============================================================

print("\n" + "-" * 70)
print("SAVING RESULTS")
print("-" * 70)

match_visualization = cv2.drawMatches(
    img1,
    keypoints1,
    img2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

visualization_path = os.path.join(
    RESULTS_DIR,
    "sift_matches_v_soldiers.png"
)

cv2.imwrite(
    visualization_path,
    match_visualization
)


# ============================================================
# SAVE ERROR CSV
# ============================================================

error_csv_path = os.path.join(
    RESULTS_DIR,
    "sift_errors.csv"
)

with open(
    error_csv_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "match_index,"
        "image1_x,"
        "image1_y,"
        "image2_x,"
        "image2_y,"
        "projected_x,"
        "projected_y,"
        "geometric_error_px\n"
    )

    for i in range(len(errors)):

        f.write(
            f"{i},"
            f"{pts1[i, 0]:.6f},"
            f"{pts1[i, 1]:.6f},"
            f"{pts2[i, 0]:.6f},"
            f"{pts2[i, 1]:.6f},"
            f"{projected_pts2[i, 0]:.6f},"
            f"{projected_pts2[i, 1]:.6f},"
            f"{errors[i]:.6f}\n"
        )


# ============================================================
# SAVE NUMPY DATA
# ============================================================

npz_path = os.path.join(
    RESULTS_DIR,
    "sift_results.npz"
)

np.savez(
    npz_path,
    pts1=pts1,
    pts2=pts2,
    projected_pts2=projected_pts2,
    errors=errors
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "sift_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 4 TASK 7B - CONTROLLED SIFT BASELINE\n"
    )
    f.write("=" * 60 + "\n\n")

    f.write("Dataset: HPatches\n")
    f.write("Sequence: v_soldiers\n")
    f.write("Image pair: 1.ppm -> 2.ppm\n")
    f.write("Ground truth: H_1_2\n\n")

    f.write(
        f"Image 1 shape: {img1.shape}\n"
    )
    f.write(
        f"Image 2 shape: {img2.shape}\n\n"
    )

    f.write(
        f"Image 1 keypoints: {len(keypoints1)}\n"
    )
    f.write(
        f"Image 2 keypoints: {len(keypoints2)}\n"
    )

    f.write(
        f"Image 1 descriptors: {descriptors1.shape}\n"
    )
    f.write(
        f"Image 2 descriptors: {descriptors2.shape}\n\n"
    )

    f.write(
        f"Total KNN matches: {len(knn_matches)}\n"
    )

    f.write(
        f"Lowe ratio threshold: {RATIO_THRESHOLD}\n"
    )

    f.write(
        f"Good matches: {len(good_matches)}\n\n"
    )

    f.write(
        f"Mean geometric error: "
        f"{np.mean(errors):.4f} px\n"
    )

    f.write(
        f"Median geometric error: "
        f"{np.median(errors):.4f} px\n"
    )

    f.write(
        f"Minimum geometric error: "
        f"{np.min(errors):.4f} px\n"
    )

    f.write(
        f"Maximum geometric error: "
        f"{np.max(errors):.4f} px\n\n"
    )

    for threshold, (count, percentage) in threshold_counts.items():

        f.write(
            f"Matches within {threshold:.0f} px: "
            f"{count}/{len(errors)} "
            f"({percentage:.2f}%)\n"
        )


print("\nResults saved to:")
print(RESULTS_DIR)

print("\nGenerated files:")
print("  - sift_summary.txt")
print("  - sift_errors.csv")
print("  - sift_results.npz")
print("  - sift_matches_v_soldiers.png")

print("\n" + "=" * 70)
print("TASK 7B COMPLETED")
print("=" * 70)