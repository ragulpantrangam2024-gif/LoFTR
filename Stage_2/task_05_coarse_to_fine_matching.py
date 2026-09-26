import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE1_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_woman",
    "1.ppm"
)

IMAGE2_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_woman",
    "2.ppm"
)

H_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_woman",
    "H_1_2"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_05"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 256

COARSE_PATCH_SIZE = 16
FINE_PATCH_SIZE = 4

EMBED_DIM = 64

LOCAL_RADIUS = 2

NUM_COARSE_ROWS = IMAGE_SIZE // COARSE_PATCH_SIZE
NUM_COARSE_COLS = IMAGE_SIZE // COARSE_PATCH_SIZE

NUM_FINE_ROWS = IMAGE_SIZE // FINE_PATCH_SIZE
NUM_FINE_COLS = IMAGE_SIZE // FINE_PATCH_SIZE


# ============================================================
# LOAD IMAGES
# ============================================================

image1 = cv2.imread(
    IMAGE1_PATH,
    cv2.IMREAD_GRAYSCALE
)

image2 = cv2.imread(
    IMAGE2_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image1 is None:
    raise FileNotFoundError(
        f"Could not load Image 1:\n{IMAGE1_PATH}"
    )

if image2 is None:
    raise FileNotFoundError(
        f"Could not load Image 2:\n{IMAGE2_PATH}"
    )


original_shape1 = image1.shape
original_shape2 = image2.shape


# ============================================================
# RESIZE IMAGES
# ============================================================

image1 = cv2.resize(
    image1,
    (IMAGE_SIZE, IMAGE_SIZE)
)

image2 = cv2.resize(
    image2,
    (IMAGE_SIZE, IMAGE_SIZE)
)


# ============================================================
# LOAD AND SCALE GROUND-TRUTH HOMOGRAPHY
# ============================================================

H_original = np.loadtxt(H_PATH)

h1, w1 = original_shape1
h2, w2 = original_shape2


# Transformation from original Image 1 coordinates
# to resized Image 1 coordinates
S1 = np.array(
    [
        [IMAGE_SIZE / w1, 0, 0],
        [0, IMAGE_SIZE / h1, 0],
        [0, 0, 1]
    ],
    dtype=np.float64
)


# Transformation from original Image 2 coordinates
# to resized Image 2 coordinates
S2 = np.array(
    [
        [IMAGE_SIZE / w2, 0, 0],
        [0, IMAGE_SIZE / h2, 0],
        [0, 0, 1]
    ],
    dtype=np.float64
)


# Original:
#
# p2 = H_original * p1
#
# After resizing:
#
# p2_resized = S2 * H_original * S1^-1 * p1_resized
#
H_resized = (
    S2
    @ H_original
    @ np.linalg.inv(S1)
)

H_resized = (
    H_resized
    / H_resized[2, 2]
)


# ============================================================
# PATCH EXTRACTION
# ============================================================

def extract_patches(image, patch_size):
    """
    Divide an image into non-overlapping square patches.

    Returns:
        patches:
            [number_of_patches,
             patch_size * patch_size]
    """

    h, w = image.shape

    patches = []

    for y in range(0, h, patch_size):

        for x in range(0, w, patch_size):

            patch = image[
                y:y + patch_size,
                x:x + patch_size
            ]

            patches.append(
                patch.astype(
                    np.float32
                ).reshape(-1)
            )

    return np.stack(patches)


# ============================================================
# PATCH NORMALIZATION
# ============================================================

def normalize_patches(patches):
    """
    Convert pixel values from [0, 255]
    to approximately [0, 1].
    """

    return patches / 255.0


# ============================================================
# PATCH CENTER
# ============================================================

def patch_center(
    index,
    patch_size,
    image_size
):
    """
    Return the center coordinate of a patch.

    Coordinate convention:
        x = horizontal
        y = vertical
    """

    cols = image_size // patch_size

    row = index // cols
    col = index % cols

    x = (
        col * patch_size
        + patch_size / 2
    )

    y = (
        row * patch_size
        + patch_size / 2
    )

    return np.array(
        [x, y],
        dtype=np.float32
    )


# ============================================================
# FINE PATCH INDEX
# ============================================================

def fine_patch_index(
    x,
    y,
    patch_size,
    image_size
):
    """
    Convert an image coordinate into
    a fine-patch index.
    """

    cols = image_size // patch_size

    col = int(
        np.clip(
            x // patch_size,
            0,
            cols - 1
        )
    )

    row = int(
        np.clip(
            y // patch_size,
            0,
            cols - 1
        )
    )

    return row * cols + col


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(42)


# ============================================================
# EMBEDDING LAYERS
# ============================================================

coarse_embedding = nn.Linear(
    COARSE_PATCH_SIZE * COARSE_PATCH_SIZE,
    EMBED_DIM
)

fine_embedding = nn.Linear(
    FINE_PATCH_SIZE * FINE_PATCH_SIZE,
    EMBED_DIM
)


# ============================================================
# COARSE FEATURES
# ============================================================

coarse_patches1 = extract_patches(
    image1,
    COARSE_PATCH_SIZE
)

coarse_patches2 = extract_patches(
    image2,
    COARSE_PATCH_SIZE
)


coarse_patches1 = normalize_patches(
    coarse_patches1
)

coarse_patches2 = normalize_patches(
    coarse_patches2
)


coarse_patches1 = torch.tensor(
    coarse_patches1,
    dtype=torch.float32
)

coarse_patches2 = torch.tensor(
    coarse_patches2,
    dtype=torch.float32
)


coarse_features1 = coarse_embedding(
    coarse_patches1
)

coarse_features2 = coarse_embedding(
    coarse_patches2
)


# ============================================================
# L2 NORMALIZATION
# ============================================================

coarse_features1 = nn.functional.normalize(
    coarse_features1,
    p=2,
    dim=1
)

coarse_features2 = nn.functional.normalize(
    coarse_features2,
    p=2,
    dim=1
)


# ============================================================
# COARSE SIMILARITY MATRIX
# ============================================================

similarity = torch.matmul(
    coarse_features1,
    coarse_features2.T
)

similarity_np = (
    similarity
    .detach()
    .numpy()
)


# ============================================================
# COARSE MUTUAL NEAREST-NEIGHBOR MATCHING
# ============================================================

best_image2 = torch.argmax(
    similarity,
    dim=1
)

best_image1 = torch.argmax(
    similarity,
    dim=0
)


coarse_matches = []


for i in range(
    similarity.shape[0]
):

    j = best_image2[i].item()

    # Mutual nearest-neighbor condition
    if best_image1[j].item() == i:

        score = similarity[
            i,
            j
        ].item()

        coarse_matches.append(
            (
                i,
                j,
                score
            )
        )


# Sort from highest similarity
# to lowest similarity
coarse_matches = sorted(
    coarse_matches,
    key=lambda x: x[2],
    reverse=True
)


# ============================================================
# PREPARE COARSE MATCHES FOR EVALUATION
# ============================================================

coarse_matches_for_eval = []


for i, j, score in coarse_matches:

    p1 = patch_center(
        i,
        COARSE_PATCH_SIZE,
        IMAGE_SIZE
    )

    p2 = patch_center(
        j,
        COARSE_PATCH_SIZE,
        IMAGE_SIZE
    )

    coarse_matches_for_eval.append(
        (
            p1,
            p2,
            score
        )
    )


# ============================================================
# FINE FEATURES
# ============================================================

fine_patches1 = extract_patches(
    image1,
    FINE_PATCH_SIZE
)

fine_patches2 = extract_patches(
    image2,
    FINE_PATCH_SIZE
)


fine_patches1 = normalize_patches(
    fine_patches1
)

fine_patches2 = normalize_patches(
    fine_patches2
)


fine_patches1 = torch.tensor(
    fine_patches1,
    dtype=torch.float32
)

fine_patches2 = torch.tensor(
    fine_patches2,
    dtype=torch.float32
)


fine_features1 = fine_embedding(
    fine_patches1
)

fine_features2 = fine_embedding(
    fine_patches2
)


# ============================================================
# FINE FEATURE NORMALIZATION
# ============================================================

fine_features1 = nn.functional.normalize(
    fine_features1,
    p=2,
    dim=1
)

fine_features2 = nn.functional.normalize(
    fine_features2,
    p=2,
    dim=1
)


# ============================================================
# FINE LOCAL MATCHING
# ============================================================

fine_matches = []


# Only refine the top 100 coarse matches
# This is sufficient for this educational experiment.
for coarse_i, coarse_j, coarse_score in coarse_matches[:100]:

    center1 = patch_center(
        coarse_i,
        COARSE_PATCH_SIZE,
        IMAGE_SIZE
    )

    center2 = patch_center(
        coarse_j,
        COARSE_PATCH_SIZE,
        IMAGE_SIZE
    )


    # Find corresponding fine patch
    # around the coarse location.
    fine_index1 = fine_patch_index(
        center1[0],
        center1[1],
        FINE_PATCH_SIZE,
        IMAGE_SIZE
    )

    fine_index2 = fine_patch_index(
        center2[0],
        center2[1],
        FINE_PATCH_SIZE,
        IMAGE_SIZE
    )


    cols = (
        IMAGE_SIZE
        // FINE_PATCH_SIZE
    )


    row2 = fine_index2 // cols
    col2 = fine_index2 % cols


    # --------------------------------------------------------
    # Create local candidate neighborhood
    # --------------------------------------------------------

    candidate_indices = []


    for dy in range(
        -LOCAL_RADIUS,
        LOCAL_RADIUS + 1
    ):

        for dx in range(
            -LOCAL_RADIUS,
            LOCAL_RADIUS + 1
        ):

            r = row2 + dy
            c = col2 + dx


            if (
                0 <= r < NUM_FINE_ROWS
                and
                0 <= c < NUM_FINE_COLS
            ):

                idx = (
                    r * cols
                    + c
                )

                candidate_indices.append(
                    idx
                )


    # --------------------------------------------------------
    # Query feature
    # --------------------------------------------------------

    query_feature = fine_features1[
        fine_index1
    ].unsqueeze(0)


    # --------------------------------------------------------
    # Candidate features
    # --------------------------------------------------------

    candidate_features = fine_features2[
        candidate_indices
    ]


    # --------------------------------------------------------
    # Local similarity
    # --------------------------------------------------------

    scores = torch.matmul(
        query_feature,
        candidate_features.T
    ).squeeze(0)


    # --------------------------------------------------------
    # Best local candidate
    # --------------------------------------------------------

    best_local = torch.argmax(
        scores
    ).item()


    best_index2 = (
        candidate_indices[
            best_local
        ]
    )


    best_score = (
        scores[
            best_local
        ].item()
    )


    # --------------------------------------------------------
    # Refined coordinates
    # --------------------------------------------------------

    refined_center1 = patch_center(
        fine_index1,
        FINE_PATCH_SIZE,
        IMAGE_SIZE
    )

    refined_center2 = patch_center(
        best_index2,
        FINE_PATCH_SIZE,
        IMAGE_SIZE
    )


    fine_matches.append(
        (
            refined_center1,
            refined_center2,
            best_score
        )
    )


# ============================================================
# GROUND-TRUTH EVALUATION FUNCTION
# ============================================================

def evaluate_matches(
    matches,
    H
):
    """
    Evaluate predicted correspondences
    against a ground-truth homography.

    Error:
        Euclidean distance between the
        predicted point in Image 2 and
        the homography-transformed point.
    """

    if len(matches) == 0:

        return {
            "count": 0,
            "mean_error": float("nan"),
            "median_error": float("nan"),
            "min_error": float("nan"),
            "max_error": float("nan"),
            "within_1px": 0,
            "within_3px": 0,
            "within_5px": 0,
            "within_10px": 0
        }


    predicted_points1 = []
    predicted_points2 = []


    for match in matches:

        p1 = match[0]
        p2 = match[1]

        predicted_points1.append(
            p1
        )

        predicted_points2.append(
            p2
        )


    predicted_points1 = np.array(
        predicted_points1,
        dtype=np.float32
    )

    predicted_points2 = np.array(
        predicted_points2,
        dtype=np.float32
    )


    # Transform Image 1 points
    # into Image 2 using ground truth.
    gt_points2 = cv2.perspectiveTransform(
        predicted_points1.reshape(
            -1,
            1,
            2
        ),
        H
    ).reshape(
        -1,
        2
    )


    # Euclidean geometric error
    errors = np.linalg.norm(
        predicted_points2
        - gt_points2,
        axis=1
    )


    return {
        "count": len(errors),

        "mean_error": float(
            np.mean(errors)
        ),

        "median_error": float(
            np.median(errors)
        ),

        "min_error": float(
            np.min(errors)
        ),

        "max_error": float(
            np.max(errors)
        ),

        "within_1px": int(
            np.sum(
                errors <= 1.0
            )
        ),

        "within_3px": int(
            np.sum(
                errors <= 3.0
            )
        ),

        "within_5px": int(
            np.sum(
                errors <= 5.0
            )
        ),

        "within_10px": int(
            np.sum(
                errors <= 10.0
            )
        )
    }


# ============================================================
# EVALUATE COARSE AND FINE MATCHES
# ============================================================

coarse_metrics = evaluate_matches(
    coarse_matches_for_eval,
    H_resized
)

fine_metrics = evaluate_matches(
    fine_matches,
    H_resized
)


# ============================================================
# VISUALIZATION 1 — IMAGE PAIR
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


axes[0].imshow(
    image1,
    cmap="gray"
)

axes[0].set_title(
    "Image 1"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    image2,
    cmap="gray"
)

axes[1].set_title(
    "Image 2"
)

axes[1].axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "image_pair.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZATION 2 — COARSE SIMILARITY MATRIX
# ============================================================

plt.figure(
    figsize=(8, 7)
)


plt.imshow(
    similarity_np,
    cmap="viridis",
    aspect="auto"
)


plt.colorbar(
    label="Cosine Similarity"
)


plt.xlabel(
    "Image 2 coarse patches"
)

plt.ylabel(
    "Image 1 coarse patches"
)


plt.title(
    "Coarse Feature Similarity Matrix"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "coarse_similarity_matrix.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZATION 3 — COARSE MATCHES
# ============================================================

canvas_coarse = np.concatenate(
    [
        image1,
        image2
    ],
    axis=1
)


plt.figure(
    figsize=(14, 7)
)


plt.imshow(
    canvas_coarse,
    cmap="gray"
)


for i, j, score in coarse_matches[:50]:

    p1 = patch_center(
        i,
        COARSE_PATCH_SIZE,
        IMAGE_SIZE
    )

    p2 = patch_center(
        j,
        COARSE_PATCH_SIZE,
        IMAGE_SIZE
    )


    # Shift Image 2 point
    # to the right side.
    p2_plot = p2.copy()

    p2_plot[0] += IMAGE_SIZE


    plt.plot(
        [
            p1[0],
            p2_plot[0]
        ],
        [
            p1[1],
            p2_plot[1]
        ],
        linewidth=0.7
    )


    plt.scatter(
        p1[0],
        p1[1],
        s=10
    )


    plt.scatter(
        p2_plot[0],
        p2_plot[1],
        s=10
    )


plt.axvline(
    IMAGE_SIZE,
    linewidth=2
)


plt.title(
    "Top Coarse Mutual Matches "
    f"({min(50, len(coarse_matches))})"
)


plt.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "coarse_matches.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZATION 4 — FINE MATCHES
# ============================================================

canvas_fine = np.concatenate(
    [
        image1,
        image2
    ],
    axis=1
)


plt.figure(
    figsize=(14, 7)
)


plt.imshow(
    canvas_fine,
    cmap="gray"
)


for p1, p2, score in fine_matches[:50]:

    p2_plot = p2.copy()

    p2_plot[0] += IMAGE_SIZE


    plt.plot(
        [
            p1[0],
            p2_plot[0]
        ],
        [
            p1[1],
            p2_plot[1]
        ],
        linewidth=0.8
    )


    plt.scatter(
        p1[0],
        p1[1],
        s=12
    )


    plt.scatter(
        p2_plot[0],
        p2_plot[1],
        s=12
    )


plt.axvline(
    IMAGE_SIZE,
    linewidth=2
)


plt.title(
    "Fine-Refined Matches "
    f"({min(50, len(fine_matches))})"
)


plt.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "fine_matches.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "task_05_results.txt"
)


with open(
    results_path,
    "w"
) as f:

    f.write(
        "Stage 2 Task 5 — Coarse-to-Fine Matching\n"
    )

    f.write(
        "=========================================\n\n"
    )


    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    f.write(
        "CONFIGURATION\n"
    )

    f.write(
        "-------------\n"
    )

    f.write(
        f"Original Image 1 shape: "
        f"{original_shape1}\n"
    )

    f.write(
        f"Original Image 2 shape: "
        f"{original_shape2}\n"
    )

    f.write(
        f"Resized image size: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}\n"
    )

    f.write(
        f"Coarse patch size: "
        f"{COARSE_PATCH_SIZE} x "
        f"{COARSE_PATCH_SIZE}\n"
    )

    f.write(
        f"Fine patch size: "
        f"{FINE_PATCH_SIZE} x "
        f"{FINE_PATCH_SIZE}\n"
    )

    f.write(
        f"Coarse tokens Image 1: "
        f"{len(coarse_features1)}\n"
    )

    f.write(
        f"Coarse tokens Image 2: "
        f"{len(coarse_features2)}\n"
    )

    f.write(
        f"Fine tokens Image 1: "
        f"{len(fine_features1)}\n"
    )

    f.write(
        f"Fine tokens Image 2: "
        f"{len(fine_features2)}\n"
    )

    f.write(
        f"Feature dimension: "
        f"{EMBED_DIM}\n"
    )

    f.write(
        f"Local fine search radius: "
        f"{LOCAL_RADIUS}\n"
    )

    f.write(
        f"Similarity matrix shape: "
        f"{tuple(similarity.shape)}\n"
    )


    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    f.write(
        "\n\nMATCHING RESULTS\n"
    )

    f.write(
        "================\n"
    )

    f.write(
        f"Mutual coarse matches: "
        f"{len(coarse_matches)}\n"
    )

    f.write(
        f"Fine-refined matches: "
        f"{len(fine_matches)}\n"
    )


    # --------------------------------------------------------
    # Top coarse matches
    # --------------------------------------------------------

    f.write(
        "\n\nTOP COARSE MATCHES\n"
    )

    f.write(
        "==================\n"
    )


    for i, j, score in coarse_matches[:20]:

        f.write(
            f"Image1 patch {i} -> "
            f"Image2 patch {j}, "
            f"similarity = {score:.6f}\n"
        )


    # --------------------------------------------------------
    # Ground-truth evaluation
    # --------------------------------------------------------

    f.write(
        "\n\nGROUND-TRUTH EVALUATION\n"
    )

    f.write(
        "=======================\n"
    )


    f.write(
        "\nCOARSE MATCHING\n"
    )

    f.write(
        "---------------\n"
    )

    f.write(
        f"Matches: "
        f"{coarse_metrics['count']}\n"
    )

    f.write(
        f"Mean error: "
        f"{coarse_metrics['mean_error']:.4f} px\n"
    )

    f.write(
        f"Median error: "
        f"{coarse_metrics['median_error']:.4f} px\n"
    )

    f.write(
        f"Min error: "
        f"{coarse_metrics['min_error']:.4f} px\n"
    )

    f.write(
        f"Max error: "
        f"{coarse_metrics['max_error']:.4f} px\n"
    )

    f.write(
        f"Within 1 px: "
        f"{coarse_metrics['within_1px']}\n"
    )

    f.write(
        f"Within 3 px: "
        f"{coarse_metrics['within_3px']}\n"
    )

    f.write(
        f"Within 5 px: "
        f"{coarse_metrics['within_5px']}\n"
    )

    f.write(
        f"Within 10 px: "
        f"{coarse_metrics['within_10px']}\n"
    )


    f.write(
        "\nFINE MATCHING\n"
    )

    f.write(
        "-------------\n"
    )

    f.write(
        f"Matches: "
        f"{fine_metrics['count']}\n"
    )

    f.write(
        f"Mean error: "
        f"{fine_metrics['mean_error']:.4f} px\n"
    )

    f.write(
        f"Median error: "
        f"{fine_metrics['median_error']:.4f} px\n"
    )

    f.write(
        f"Min error: "
        f"{fine_metrics['min_error']:.4f} px\n"
    )

    f.write(
        f"Max error: "
        f"{fine_metrics['max_error']:.4f} px\n"
    )

    f.write(
        f"Within 1 px: "
        f"{fine_metrics['within_1px']}\n"
    )

    f.write(
        f"Within 3 px: "
        f"{fine_metrics['within_3px']}\n"
    )

    f.write(
        f"Within 5 px: "
        f"{fine_metrics['within_5px']}\n"
    )

    f.write(
        f"Within 10 px: "
        f"{fine_metrics['within_10px']}\n"
    )


    # --------------------------------------------------------
    # Interpretation note
    # --------------------------------------------------------

    f.write(
        "\n\nNOTE\n"
    )

    f.write(
        "----\n"
    )

    f.write(
        "This is an educational coarse-to-fine "
        "matching implementation inspired by "
        "the LoFTR pipeline.\n"
    )

    f.write(
        "The feature projection layers are randomly "
        "initialized and are not trained LoFTR features.\n"
    )

    f.write(
        "Therefore, the experiment demonstrates "
        "the matching mechanism rather than reproducing "
        "the performance of the original LoFTR model.\n"
    )

    f.write(
        "Ground-truth errors are calculated using "
        "the HPatches H_1_2 homography after scaling "
        "it to the 256x256 resized images.\n"
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print(
    "=============================================="
)

print(
    "Stage 2 Task 5 completed"
)

print(
    "=============================================="
)


print(
    f"Original Image 1: "
    f"{original_shape1}"
)

print(
    f"Original Image 2: "
    f"{original_shape2}"
)

print(
    f"Resized: "
    f"{IMAGE_SIZE} x {IMAGE_SIZE}"
)

print(
    f"Coarse tokens: "
    f"{len(coarse_features1)}"
)

print(
    f"Fine tokens: "
    f"{len(fine_features1)}"
)

print(
    f"Embedding dimension: "
    f"{EMBED_DIM}"
)

print(
    f"Similarity matrix: "
    f"{tuple(similarity.shape)}"
)

print(
    f"Mutual coarse matches: "
    f"{len(coarse_matches)}"
)

print(
    f"Fine refined matches: "
    f"{len(fine_matches)}"
)


print()
print(
    "=============================================="
)

print(
    "GROUND-TRUTH EVALUATION"
)

print(
    "=============================================="
)


print()
print(
    "COARSE MATCHING"
)

print(
    "----------------------------------------------"
)

print(
    f"Matches:        "
    f"{coarse_metrics['count']}"
)

print(
    f"Mean error:     "
    f"{coarse_metrics['mean_error']:.4f} px"
)

print(
    f"Median error:   "
    f"{coarse_metrics['median_error']:.4f} px"
)

print(
    f"Min error:      "
    f"{coarse_metrics['min_error']:.4f} px"
)

print(
    f"Max error:      "
    f"{coarse_metrics['max_error']:.4f} px"
)

print(
    f"Within 1 px:    "
    f"{coarse_metrics['within_1px']}"
)

print(
    f"Within 3 px:    "
    f"{coarse_metrics['within_3px']}"
)

print(
    f"Within 5 px:    "
    f"{coarse_metrics['within_5px']}"
)

print(
    f"Within 10 px:   "
    f"{coarse_metrics['within_10px']}"
)


print()
print(
    "FINE MATCHING"
)

print(
    "----------------------------------------------"
)

print(
    f"Matches:        "
    f"{fine_metrics['count']}"
)

print(
    f"Mean error:     "
    f"{fine_metrics['mean_error']:.4f} px"
)

print(
    f"Median error:   "
    f"{fine_metrics['median_error']:.4f} px"
)

print(
    f"Min error:      "
    f"{fine_metrics['min_error']:.4f} px"
)

print(
    f"Max error:      "
    f"{fine_metrics['max_error']:.4f} px"
)

print(
    f"Within 1 px:    "
    f"{fine_metrics['within_1px']}"
)

print(
    f"Within 3 px:    "
    f"{fine_metrics['within_3px']}"
)

print(
    f"Within 5 px:    "
    f"{fine_metrics['within_5px']}"
)

print(
    f"Within 10 px:   "
    f"{fine_metrics['within_10px']}"
)


print()
print(
    "Results saved to:"
)

print(
    RESULTS_DIR
)

print(
    "=============================================="
)