import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STAGE 1 - TASK 1
# IMAGE CORRESPONDENCES USING HPATCHES
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

# task_01_image_correspondences.py
#        │
#        └── Stage_1
#               │
#               └── LoFTR
#
# parents[1] = LoFTR project root

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "hpatches-sequences-release"
)

SEQUENCE_DIR = DATASET_DIR / "v_woman"

IMAGE_1_PATH = SEQUENCE_DIR / "1.ppm"
IMAGE_2_PATH = SEQUENCE_DIR / "2.ppm"

H_PATH = SEQUENCE_DIR / "H_1_2"

RESULTS_DIR = Path(__file__).resolve().parent / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. Check required files
# ------------------------------------------------------------

required_files = [
    IMAGE_1_PATH,
    IMAGE_2_PATH,
    H_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ------------------------------------------------------------
# 3. Load images
# ------------------------------------------------------------

image_1 = cv2.imread(
    str(IMAGE_1_PATH),
    cv2.IMREAD_COLOR
)

image_2 = cv2.imread(
    str(IMAGE_2_PATH),
    cv2.IMREAD_COLOR
)


if image_1 is None:

    raise RuntimeError(
        f"Could not load image:\n{IMAGE_1_PATH}"
    )


if image_2 is None:

    raise RuntimeError(
        f"Could not load image:\n{IMAGE_2_PATH}"
    )


# ------------------------------------------------------------
# 4. Load ground-truth homography
# ------------------------------------------------------------

H = np.loadtxt(H_PATH)


# Check homography shape
if H.shape != (3, 3):

    raise ValueError(
        f"Expected a 3x3 homography matrix, "
        f"but received shape {H.shape}"
    )


# ------------------------------------------------------------
# 5. Print dataset information
# ------------------------------------------------------------

print()
print("=" * 70)
print("STAGE 1 - TASK 1")
print("HPATCHES IMAGE CORRESPONDENCES")
print("=" * 70)

print()
print("Sequence:")
print("v_woman")

print()
print("Reference image:")
print(IMAGE_1_PATH)

print()
print("Target image:")
print(IMAGE_2_PATH)


# ------------------------------------------------------------
# 6. Print image dimensions
# ------------------------------------------------------------

height_1, width_1 = image_1.shape[:2]

height_2, width_2 = image_2.shape[:2]


print()
print("Image 1 dimensions:")
print(f"Width  : {width_1}")
print(f"Height : {height_1}")

print()
print("Image 2 dimensions:")
print(f"Width  : {width_2}")
print(f"Height : {height_2}")


# ------------------------------------------------------------
# 7. Print homography
# ------------------------------------------------------------

print()
print("Ground-truth homography H_1_2:")
print(H)


# ============================================================
# PART B
# TRANSFORM POINTS USING THE HOMOGRAPHY
# ============================================================


# ------------------------------------------------------------
# 8. Define points manually in Image 1
# ------------------------------------------------------------

points_image_1 = np.array(
    [
        [250, 300],
        [500, 300],
        [700, 500],
        [400, 600],
        [800, 200]
    ],
    dtype=np.float32
)


# ------------------------------------------------------------
# 9. Convert points to homogeneous coordinates
# ------------------------------------------------------------

# From:
#
# (x, y)
#
# to:
#
# (x, y, 1)

points_homogeneous = np.hstack(
    [
        points_image_1,
        np.ones(
            (len(points_image_1), 1),
            dtype=np.float32
        )
    ]
)


# ------------------------------------------------------------
# 10. Apply homography
# ------------------------------------------------------------

# p' = H @ p

transformed_points = (
    H @ points_homogeneous.T
).T


# ------------------------------------------------------------
# 11. Normalize homogeneous coordinates
# ------------------------------------------------------------

# Homogeneous point:
#
# (x', y', w')
#
# becomes:
#
# X = x' / w'
# Y = y' / w'

transformed_points = (
    transformed_points[:, :2]
    /
    transformed_points[:, 2:3]
)


# ============================================================
# PRINT GROUND-TRUTH CORRESPONDENCES
# ============================================================

print()
print("=" * 70)
print("GROUND-TRUTH CORRESPONDENCES")
print("=" * 70)


for i, (point_1, point_2) in enumerate(
    zip(
        points_image_1,
        transformed_points
    )
):

    print()
    print(f"Point {i + 1}")

    print(
        f"Image 1 : "
        f"({point_1[0]:.2f}, "
        f"{point_1[1]:.2f})"
    )

    print(
        f"Image 2 : "
        f"({point_2[0]:.2f}, "
        f"{point_2[1]:.2f})"
    )


# ============================================================
# VISUALIZATION
# ============================================================


# ------------------------------------------------------------
# 12. Create figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7)
)


# ------------------------------------------------------------
# 13. Display Image 1
# ------------------------------------------------------------

axes[0].imshow(
    cv2.cvtColor(
        image_1,
        cv2.COLOR_BGR2RGB
    )
)


# Plot reference points

axes[0].scatter(
    points_image_1[:, 0],
    points_image_1[:, 1],
    s=100,
    marker="x"
)


axes[0].set_title(
    "Image 1 - Reference Points"
)

axes[0].axis("off")


# ------------------------------------------------------------
# 14. Display Image 2
# ------------------------------------------------------------

axes[1].imshow(
    cv2.cvtColor(
        image_2,
        cv2.COLOR_BGR2RGB
    )
)


# Plot transformed points

axes[1].scatter(
    transformed_points[:, 0],
    transformed_points[:, 1],
    s=100,
    marker="x"
)


axes[1].set_title(
    "Image 2 - Ground-Truth Corresponding Points"
)

axes[1].axis("off")


# ------------------------------------------------------------
# 15. Add point labels
# ------------------------------------------------------------

for i, point in enumerate(points_image_1):

    axes[0].text(
        point[0] + 10,
        point[1],
        f"P{i + 1}",
        fontsize=12
    )


for i, point in enumerate(transformed_points):

    axes[1].text(
        point[0] + 10,
        point[1],
        f"P{i + 1}'",
        fontsize=12
    )


# ------------------------------------------------------------
# 16. Improve layout
# ------------------------------------------------------------

plt.tight_layout()


# ------------------------------------------------------------
# 17. Save visualization
# ------------------------------------------------------------

output_path = (
    RESULTS_DIR
    / "task_01_ground_truth_correspondences.png"
)


plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


# ------------------------------------------------------------
# 18. Display visualization
# ------------------------------------------------------------

plt.show()


# ------------------------------------------------------------
# 19. Final message
# ------------------------------------------------------------

print()
print("=" * 70)
print("TASK 1 PART B COMPLETED")
print("=" * 70)

print()
print("Ground-truth correspondence visualization saved to:")

print(output_path)

print()

# ============================================================
# PART C
# HOMOGRAPHY ROUND-TRIP VERIFICATION
# ============================================================


# ------------------------------------------------------------
# 20. Calculate inverse homography
# ------------------------------------------------------------

H_inverse = np.linalg.inv(H)


# ------------------------------------------------------------
# 21. Convert transformed points back to homogeneous form
# ------------------------------------------------------------

transformed_points_homogeneous = np.hstack(
    [
        transformed_points,
        np.ones(
            (len(transformed_points), 1),
            dtype=np.float32
        )
    ]
)


# ------------------------------------------------------------
# 22. Transform Image 2 points back to Image 1
# ------------------------------------------------------------

recovered_points = (
    H_inverse
    @ transformed_points_homogeneous.T
).T


# ------------------------------------------------------------
# 23. Normalize recovered points
# ------------------------------------------------------------

recovered_points = (
    recovered_points[:, :2]
    /
    recovered_points[:, 2:3]
)


# ------------------------------------------------------------
# 24. Calculate round-trip error
# ------------------------------------------------------------

errors = np.linalg.norm(
    points_image_1 - recovered_points,
    axis=1
)


# ------------------------------------------------------------
# 25. Print verification results
# ------------------------------------------------------------

print()
print("=" * 70)
print("HOMOGRAPHY ROUND-TRIP VERIFICATION")
print("=" * 70)


for i, (
    original,
    recovered,
    error
) in enumerate(
    zip(
        points_image_1,
        recovered_points,
        errors
    )
):

    print()
    print(f"Point {i + 1}")

    print(
        f"Original : "
        f"({original[0]:.6f}, "
        f"{original[1]:.6f})"
    )

    print(
        f"Recovered: "
        f"({recovered[0]:.6f}, "
        f"{recovered[1]:.6f})"
    )

    print(
        f"Round-trip error: "
        f"{error:.10f} pixels"
    )


# ------------------------------------------------------------
# 26. Calculate overall error statistics
# ------------------------------------------------------------

mean_error = np.mean(errors)
max_error = np.max(errors)


print()
print("-" * 70)

print(
    f"Mean round-trip error: "
    f"{mean_error:.10f} pixels"
)

print(
    f"Maximum round-trip error: "
    f"{max_error:.10f} pixels"
)

print("-" * 70)