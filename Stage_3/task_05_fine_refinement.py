import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_soldiers"
)

IMAGE1_PATH = os.path.join(
    DATASET_DIR,
    "1.ppm"
)

IMAGE2_PATH = os.path.join(
    DATASET_DIR,
    "2.ppm"
)

HOMOGRAPHY_PATH = os.path.join(
    DATASET_DIR,
    "H_1_2"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_05"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256

FEATURE_SIZE = 128

EMBED_DIM = 64

COARSE_SIZE = 16

FINE_SEARCH_RADIUS = 4

NUM_SAMPLES = 50

torch.manual_seed(42)


# ============================================================
# CNN Feature Extractor
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# Image loading
# ============================================================

def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n{path}"
        )

    original_shape = image.shape

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image_float = (
        image.astype(np.float32) / 255.0
    )

    tensor = torch.from_numpy(
        image_float
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return (
        image,
        tensor,
        original_shape
    )


# ============================================================
# Load homography
# ============================================================

def load_homography():

    H = np.loadtxt(
        HOMOGRAPHY_PATH
    )

    if H.shape != (3, 3):

        raise ValueError(
            f"Unexpected homography shape: {H.shape}"
        )

    return H


# ============================================================
# Scale homography for 256x256 images
# ============================================================

def scale_homography(
    H_original,
    shape1_original,
    shape2_original
):

    h1, w1 = shape1_original

    h2, w2 = shape2_original

    S1 = np.array([
        [
            IMAGE_SIZE / w1,
            0,
            0
        ],
        [
            0,
            IMAGE_SIZE / h1,
            0
        ],
        [
            0,
            0,
            1
        ]
    ])

    S2 = np.array([
        [
            IMAGE_SIZE / w2,
            0,
            0
        ],
        [
            0,
            IMAGE_SIZE / h2,
            0
        ],
        [
            0,
            0,
            1
        ]
    ])

    H_resized = (
        S2
        @ H_original
        @ np.linalg.inv(S1)
    )

    return (
        H_resized
        / H_resized[2, 2]
    )


# ============================================================
# Apply homography
# ============================================================

def apply_homography(
    H,
    x,
    y
):

    point = np.array([
        x,
        y,
        1.0
    ])

    transformed = H @ point

    transformed = (
        transformed
        / transformed[2]
    )

    return (
        transformed[0],
        transformed[1]
    )


# ============================================================
# Convert feature coordinates to image coordinates
# ============================================================

def feature_to_image(
    x,
    y
):

    scale = IMAGE_SIZE / FEATURE_SIZE

    return (
        (x + 0.5) * scale,
        (y + 0.5) * scale
    )


# ============================================================
# Convert image coordinates to feature coordinates
# ============================================================

def image_to_feature(
    x,
    y
):

    scale = FEATURE_SIZE / IMAGE_SIZE

    return (
        x * scale - 0.5,
        y * scale - 0.5
    )


# ============================================================
# Generate valid coarse points
# ============================================================

def generate_coarse_points(
    H,
    num_samples
):

    points = []

    for row in range(
        COARSE_SIZE
    ):

        for col in range(
            COARSE_SIZE
        ):

            # Center of coarse cell

            x1 = (
                col + 0.5
            ) * (
                IMAGE_SIZE
                / COARSE_SIZE
            )

            y1 = (
                row + 0.5
            ) * (
                IMAGE_SIZE
                / COARSE_SIZE
            )

            x2, y2 = apply_homography(
                H,
                x1,
                y1
            )

            # Keep only points inside image

            if (
                0 <= x2 < IMAGE_SIZE
                and
                0 <= y2 < IMAGE_SIZE
            ):

                points.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

    # Deterministic subset

    points = points[
        :min(
            num_samples,
            len(points)
        )
    ]

    return points


# ============================================================
# Fine-level matching
# ============================================================

def fine_match(
    feature_map_1,
    feature_map_2,
    point
):

    (
        x1_image,
        y1_image,
        x2_true_image,
        y2_true_image
    ) = point

    # Convert Image 1 coordinate
    # to feature coordinates

    x1_feature, y1_feature = (
        image_to_feature(
            x1_image,
            y1_image
        )
    )

    x1_feature_index = int(
        round(x1_feature)
    )

    y1_feature_index = int(
        round(y1_feature)
    )

    # Extract Image 1 descriptor

    descriptor_1 = feature_map_1[
        0,
        :,
        y1_feature_index,
        x1_feature_index
    ]

    descriptor_1 = F.normalize(
        descriptor_1,
        p=2,
        dim=0
    )

    # True Image 2 location in feature coordinates

    x2_true_feature, y2_true_feature = (
        image_to_feature(
            x2_true_image,
            y2_true_image
        )
    )

    center_x = int(
        round(x2_true_feature)
    )

    center_y = int(
        round(y2_true_feature)
    )

    # Search local window

    candidates = []

    for dy in range(
        -FINE_SEARCH_RADIUS,
        FINE_SEARCH_RADIUS + 1
    ):

        for dx in range(
            -FINE_SEARCH_RADIUS,
            FINE_SEARCH_RADIUS + 1
        ):

            x = center_x + dx
            y = center_y + dy

            if (
                x < 0
                or
                x >= FEATURE_SIZE
                or
                y < 0
                or
                y >= FEATURE_SIZE
            ):

                continue

            descriptor_2 = feature_map_2[
                0,
                :,
                y,
                x
            ]

            descriptor_2 = F.normalize(
                descriptor_2,
                p=2,
                dim=0
            )

            similarity = torch.dot(
                descriptor_1,
                descriptor_2
            ).item()

            candidates.append(
                (
                    x,
                    y,
                    similarity
                )
            )

    if len(candidates) == 0:

        return None

    # Highest similarity

    best = max(
        candidates,
        key=lambda item: item[2]
    )

    best_x_feature = best[0]
    best_y_feature = best[1]
    best_similarity = best[2]

    best_x_image, best_y_image = (
        feature_to_image(
            best_x_feature,
            best_y_feature
        )
    )

    error = np.linalg.norm(
        np.array([
            best_x_image,
            best_y_image
        ])
        -
        np.array([
            x2_true_image,
            y2_true_image
        ])
    )

    return {
        "source_x": x1_image,
        "source_y": y1_image,
        "true_x": x2_true_image,
        "true_y": y2_true_image,
        "pred_x": best_x_image,
        "pred_y": best_y_image,
        "similarity": best_similarity,
        "error": float(error)
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Stage 3 - Task 5"
    )

    print(
        "Fine-Level Refinement"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    image1, tensor1, shape1 = load_image(
        IMAGE1_PATH
    )

    image2, tensor2, shape2 = load_image(
        IMAGE2_PATH
    )

    print(
        "\nImage 1 original shape:",
        shape1
    )

    print(
        "Image 2 original shape:",
        shape2
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn = SimpleCNN()

    cnn.eval()

    with torch.no_grad():

        feature_map_1 = cnn(
            tensor1
        )

        feature_map_2 = cnn(
            tensor2
        )

    print(
        "\nFine feature map image 1:",
        tuple(feature_map_1.shape)
    )

    print(
        "Fine feature map image 2:",
        tuple(feature_map_2.shape)
    )

    # --------------------------------------------------------
    # Ground truth homography
    # --------------------------------------------------------

    H_original = load_homography()

    H = scale_homography(
        H_original,
        shape1,
        shape2
    )

    print(
        "\nResized homography:"
    )

    print(
        H
    )

    # --------------------------------------------------------
    # Generate ground-truth-guided coarse points
    # --------------------------------------------------------

    coarse_points = generate_coarse_points(
        H,
        NUM_SAMPLES
    )

    print(
        "\nGround-truth-guided coarse points:",
        len(coarse_points)
    )

    print(
        "Fine search radius:",
        FINE_SEARCH_RADIUS
    )

    print(
        "Fine search window:",
        (
            2 * FINE_SEARCH_RADIUS + 1
        ),
        "x",
        (
            2 * FINE_SEARCH_RADIUS + 1
        )
    )

    # --------------------------------------------------------
    # Fine matching
    # --------------------------------------------------------

    results = []

    for point in coarse_points:

        result = fine_match(
            feature_map_1,
            feature_map_2,
            point
        )

        if result is not None:

            results.append(
                result
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    errors = np.array([
        result["error"]
        for result in results
    ])

    similarities = np.array([
        result["similarity"]
        for result in results
    ])

    if len(errors) > 0:

        mean_error = float(
            np.mean(errors)
        )

        median_error = float(
            np.median(errors)
        )

        within_3 = int(
            np.sum(errors <= 3.0)
        )

        within_5 = int(
            np.sum(errors <= 5.0)
        )

        within_10 = int(
            np.sum(errors <= 10.0)
        )

        accuracy_10 = (
            within_10
            / len(errors)
            * 100.0
        )

        print(
            "\nFine matching evaluation:"
        )

        print(
            "Number of refined matches:",
            len(results)
        )

        print(
            "Mean error:",
            mean_error,
            "px"
        )

        print(
            "Median error:",
            median_error,
            "px"
        )

        print(
            "Within 3 px:",
            within_3,
            "/",
            len(results)
        )

        print(
            "Within 5 px:",
            within_5,
            "/",
            len(results)
        )

        print(
            "Within 10 px:",
            within_10,
            "/",
            len(results)
        )

        print(
            "Accuracy within 10 px:",
            accuracy_10,
            "%"
        )

        print(
            "Mean local similarity:",
            float(
                np.mean(similarities)
            )
        )

    else:

        print(
            "\nNo fine matches generated."
        )

        return

    # --------------------------------------------------------
    # Error distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        errors,
        bins=15
    )

    plt.xlabel(
        "Fine Matching Error (pixels)"
    )

    plt.ylabel(
        "Number of Matches"
    )

    plt.title(
        "Fine-Level Matching Error Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "fine_error_distribution.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Similarity distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        similarities,
        bins=15
    )

    plt.xlabel(
        "Local Cosine Similarity"
    )

    plt.ylabel(
        "Number of Matches"
    )

    plt.title(
        "Fine-Level Similarity Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "fine_similarity_distribution.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Visualize refined points
    # --------------------------------------------------------

    canvas = np.hstack(
        [
            image1,
            image2
        ]
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.imshow(
        canvas,
        cmap="gray"
    )

    for result in results:

        x1 = result["source_x"]
        y1 = result["source_y"]

        x2 = (
            result["pred_x"]
            + IMAGE_SIZE
        )

        y2 = result["pred_y"]

        plt.plot(
            [x1, x2],
            [y1, y2],
            linewidth=0.7
        )

        plt.scatter(
            [x1, x2],
            [y1, y2],
            s=8
        )

    plt.title(
        f"Fine-Level Refinement: {len(results)} points"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "fine_refined_matches.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        RESULTS_DIR,
        "task_05_results.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 3 - Task 5: Fine-Level Refinement\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "Dataset: HPatches v_soldiers\n"
        )

        f.write(
            "Images: 1.ppm and 2.ppm\n\n"
        )

        f.write(
            f"Fine feature map image 1: "
            f"{tuple(feature_map_1.shape)}\n"
        )

        f.write(
            f"Fine feature map image 2: "
            f"{tuple(feature_map_2.shape)}\n"
        )

        f.write(
            f"Fine search radius: "
            f"{FINE_SEARCH_RADIUS}\n"
        )

        f.write(
            f"Fine search window: "
            f"{2 * FINE_SEARCH_RADIUS + 1}x"
            f"{2 * FINE_SEARCH_RADIUS + 1}\n"
        )

        f.write(
            f"Ground-truth-guided coarse points: "
            f"{len(coarse_points)}\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Coarse locations were generated using the "
            "HPatches ground-truth homography.\n"
        )

        f.write(
            "Therefore this experiment evaluates the "
            "fine-refinement mechanism and is not an "
            "end-to-end coarse matching evaluation.\n\n"
        )

        f.write(
            f"Refined matches: "
            f"{len(results)}\n"
        )

        f.write(
            f"Mean error: "
            f"{mean_error:.6f} px\n"
        )

        f.write(
            f"Median error: "
            f"{median_error:.6f} px\n"
        )

        f.write(
            f"Within 3 px: "
            f"{within_3} / {len(results)}\n"
        )

        f.write(
            f"Within 5 px: "
            f"{within_5} / {len(results)}\n"
        )

        f.write(
            f"Within 10 px: "
            f"{within_10} / {len(results)}\n"
        )

        f.write(
            f"Accuracy within 10 px: "
            f"{accuracy_10:.4f}%\n"
        )

        f.write(
            f"Mean local similarity: "
            f"{np.mean(similarities):.6f}\n"
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nTask 5 completed."
    )


if __name__ == "__main__":
    main()