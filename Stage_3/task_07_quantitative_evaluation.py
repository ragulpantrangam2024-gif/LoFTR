"""
Stage 3 - Task 7
Quantitative Evaluation of the Simplified LoFTR Pipeline

Dataset:
    HPatches v_soldiers
    Images: 1.ppm and 2.ppm

This task evaluates:
    1. CNN feature extraction
    2. Coarse token representation
    3. Transformer contextualization
    4. Coarse matching
    5. Fine local refinement
    6. Geometric accuracy using the HPatches homography
    7. Matching-threshold sensitivity

IMPORTANT:
The CNN and Transformer are randomly initialized and untrained.
Therefore, this experiment evaluates the implemented pipeline,
not the performance of a trained LoFTR model.
"""

import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

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

IMAGE1_PATH = os.path.join(DATASET_DIR, "1.ppm")
IMAGE2_PATH = os.path.join(DATASET_DIR, "2.ppm")
HOMOGRAPHY_PATH = os.path.join(DATASET_DIR, "H_1_2")

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_07"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = 256

COARSE_GRID = 16
FINE_GRID = 128

EMBED_DIM = 64
NUM_HEADS = 8

FINE_RADIUS = 4

SIMILARITY_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80
]

GEOMETRIC_THRESHOLD = 10.0


# ============================================================
# Utility functions
# ============================================================

def load_image(path):
    """
    Load grayscale image and resize to 256x256.
    """

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image:\n{path}"
        )

    original_shape = image.shape

    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    tensor = (
        torch.from_numpy(resized)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )

    return image, resized, tensor, original_shape


def load_homography(path):
    """
    Load HPatches homography.
    """

    H = np.loadtxt(path)

    if H.shape != (3, 3):
        raise ValueError(
            f"Unexpected homography shape: {H.shape}"
        )

    return H.astype(np.float64)


def resize_homography(H, source_shape, target_shape):
    """
    Convert original-image homography into the coordinate
    system of the resized 256x256 images.

    H_resized = S2 @ H_original @ inv(S1)
    """

    h1, w1 = source_shape
    h2, w2 = target_shape

    S1 = np.array(
        [
            [IMAGE_SIZE / w1, 0, 0],
            [0, IMAGE_SIZE / h1, 0],
            [0, 0, 1]
        ],
        dtype=np.float64
    )

    S2 = np.array(
        [
            [IMAGE_SIZE / w2, 0, 0],
            [0, IMAGE_SIZE / h2, 0],
            [0, 0, 1]
        ],
        dtype=np.float64
    )

    H_resized = S2 @ H @ np.linalg.inv(S1)

    H_resized = H_resized / H_resized[2, 2]

    return H_resized


def apply_homography(points, H):
    """
    Apply homography to Nx2 points.
    """

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    points_h = np.concatenate(
        [
            points,
            np.ones((len(points), 1))
        ],
        axis=1
    )

    transformed = (H @ points_h.T).T

    transformed = (
        transformed[:, :2]
        / transformed[:, 2:3]
    )

    return transformed


def create_2d_sinusoidal_encoding(
    height,
    width,
    embed_dim
):
    """
    Create 2D sinusoidal positional encoding.

    Half of the channels encode row position.
    Half encode column position.
    """

    if embed_dim % 4 != 0:
        raise ValueError(
            "embed_dim must be divisible by 4."
        )

    encoding = torch.zeros(
        height,
        width,
        embed_dim
    )

    quarter_dim = embed_dim // 4

    y = torch.arange(height).float()
    x = torch.arange(width).float()

    div_term = torch.exp(
        torch.arange(
            0,
            quarter_dim,
            dtype=torch.float32
        )
        * -(np.log(10000.0) / quarter_dim)
    )

    for i in range(quarter_dim):

        encoding[:, :, 2 * i] = torch.sin(
            y[:, None] * div_term[i]
        )

        encoding[:, :, 2 * i + 1] = torch.cos(
            y[:, None] * div_term[i]
        )

        offset = embed_dim // 2

        encoding[:, :, offset + 2 * i] = torch.sin(
            x[None, :] * div_term[i]
        )

        encoding[:, :, offset + 2 * i + 1] = torch.cos(
            x[None, :] * div_term[i]
        )

    return encoding


# ============================================================
# CNN Feature Extractor
# ============================================================

class DenseCNN(nn.Module):
    """
    Simplified CNN used in Stage 3 Tasks 1-6.
    """

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
# Transformer Matching
# ============================================================

class TransformerMatching(nn.Module):
    """
    Simplified Transformer matching module.

    Performs:
        Self-attention image 1
        Self-attention image 2
        Cross-attention image 1 -> image 2
        Cross-attention image 2 -> image 1
    """

    def __init__(
        self,
        embed_dim=64,
        num_heads=8
    ):
        super().__init__()

        self.self_attention_1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.self_attention_2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens1, tokens2):

        # ----------------------------------------------------
        # Self-attention
        # ----------------------------------------------------

        self1, _ = self.self_attention_1(
            tokens1,
            tokens1,
            tokens1
        )

        self2, _ = self.self_attention_2(
            tokens2,
            tokens2,
            tokens2
        )

        tokens1 = self.norm1(
            tokens1 + self1
        )

        tokens2 = self.norm2(
            tokens2 + self2
        )

        # ----------------------------------------------------
        # Cross-attention
        # ----------------------------------------------------

        cross1, _ = self.cross_attention_1(
            tokens1,
            tokens2,
            tokens2
        )

        cross2, _ = self.cross_attention_2(
            tokens2,
            tokens1,
            tokens1
        )

        tokens1 = self.norm1(
            tokens1 + cross1
        )

        tokens2 = self.norm2(
            tokens2 + cross2
        )

        return tokens1, tokens2


# ============================================================
# Feature preparation
# ============================================================

def extract_features(model, image_tensor):
    """
    Extract dense CNN feature map.
    """

    with torch.no_grad():
        feature_map = model(image_tensor)

    return feature_map


def create_coarse_tokens(feature_map):
    """
    Convert 128x128 CNN feature map into 16x16 coarse tokens.

    Output:
        [1, 256, 64]
    """

    pooled = nn.functional.adaptive_avg_pool2d(
        feature_map,
        (COARSE_GRID, COARSE_GRID)
    )

    # [1, 64, 16, 16]
    tokens = pooled.permute(
        0,
        2,
        3,
        1
    )

    # [1, 256, 64]
    tokens = tokens.reshape(
        1,
        COARSE_GRID * COARSE_GRID,
        EMBED_DIM
    )

    positional_encoding = create_2d_sinusoidal_encoding(
        COARSE_GRID,
        COARSE_GRID,
        EMBED_DIM
    )

    positional_encoding = positional_encoding.reshape(
        1,
        COARSE_GRID * COARSE_GRID,
        EMBED_DIM
    )

    tokens = tokens + positional_encoding

    return tokens


# ============================================================
# Coarse matching
# ============================================================

def cosine_similarity_matrix(features1, features2):
    """
    Calculate pairwise cosine similarity.
    """

    features1 = nn.functional.normalize(
        features1,
        p=2,
        dim=-1
    )

    features2 = nn.functional.normalize(
        features2,
        p=2,
        dim=-1
    )

    similarity = torch.matmul(
        features1,
        features2.transpose(1, 2)
    )

    return similarity[0]


def mutual_nearest_matches(
    similarity,
    threshold
):
    """
    Mutual nearest-neighbor matching with
    similarity threshold.
    """

    best12 = torch.argmax(
        similarity,
        dim=1
    )

    best21 = torch.argmax(
        similarity,
        dim=0
    )

    matches = []

    for i in range(similarity.shape[0]):

        j = best12[i].item()

        mutual = (
            best21[j].item() == i
        )

        score = similarity[i, j].item()

        if mutual and score >= threshold:

            matches.append(
                (
                    i,
                    j,
                    score
                )
            )

    return matches


# ============================================================
# Coarse token coordinates
# ============================================================

def coarse_token_to_image_coordinate(index):
    """
    Convert 16x16 coarse-token index into
    256x256 image coordinates.

    Each token represents an 16x16 image region.
    """

    row = index // COARSE_GRID
    col = index % COARSE_GRID

    cell_size = IMAGE_SIZE / COARSE_GRID

    x = (col + 0.5) * cell_size
    y = (row + 0.5) * cell_size

    return np.array(
        [x, y],
        dtype=np.float64
    )


def coarse_index_to_feature_coordinate(index):
    """
    Convert coarse token index into coordinates
    on the 128x128 fine feature map.
    """

    row = index // COARSE_GRID
    col = index % COARSE_GRID

    scale = FINE_GRID / COARSE_GRID

    x = (col + 0.5) * scale
    y = (row + 0.5) * scale

    return x, y


# ============================================================
# Fine refinement
# ============================================================

def fine_refinement(
    feature_map1,
    feature_map2,
    matches
):
    """
    Perform local fine matching around the
    predicted coarse target position.

    IMPORTANT:
    The target search center comes from the
    predicted coarse match, not the ground truth.
    """

    feature1 = feature_map1[0]
    feature2 = feature_map2[0]

    _, height, width = feature1.shape

    refined_matches = []

    for source_idx, target_idx, coarse_score in matches:

        source_x, source_y = (
            coarse_index_to_feature_coordinate(
                source_idx
            )
        )

        target_x, target_y = (
            coarse_index_to_feature_coordinate(
                target_idx
            )
        )

        source_x = int(round(source_x))
        source_y = int(round(source_y))

        target_x = int(round(target_x))
        target_y = int(round(target_y))

        # Keep source location inside feature map
        source_x = np.clip(
            source_x,
            0,
            width - 1
        )

        source_y = np.clip(
            source_y,
            0,
            height - 1
        )

        source_feature = feature1[
            :,
            source_y,
            source_x
        ]

        source_feature = (
            source_feature
            / (
                torch.norm(
                    source_feature
                ) + 1e-8
            )
        )

        best_score = -float("inf")
        best_x = target_x
        best_y = target_y

        for dy in range(
            -FINE_RADIUS,
            FINE_RADIUS + 1
        ):

            for dx in range(
                -FINE_RADIUS,
                FINE_RADIUS + 1
            ):

                candidate_x = target_x + dx
                candidate_y = target_y + dy

                if (
                    candidate_x < 0
                    or candidate_x >= width
                    or candidate_y < 0
                    or candidate_y >= height
                ):
                    continue

                candidate = feature2[
                    :,
                    candidate_y,
                    candidate_x
                ]

                candidate = (
                    candidate
                    / (
                        torch.norm(candidate)
                        + 1e-8
                    )
                )

                score = torch.dot(
                    source_feature,
                    candidate
                ).item()

                if score > best_score:

                    best_score = score
                    best_x = candidate_x
                    best_y = candidate_y

        refined_matches.append(
            {
                "source_idx": source_idx,
                "target_idx": target_idx,
                "coarse_score": coarse_score,
                "source_feature_x": source_x,
                "source_feature_y": source_y,
                "target_feature_x": best_x,
                "target_feature_y": best_y,
                "fine_score": best_score
            }
        )

    return refined_matches


# ============================================================
# Geometric evaluation
# ============================================================

def evaluate_coarse_matches(
    matches,
    H
):
    """
    Evaluate coarse matches against homography.
    """

    errors = []

    records = []

    for source_idx, target_idx, score in matches:

        source_point = coarse_token_to_image_coordinate(
            source_idx
        )

        target_point = coarse_token_to_image_coordinate(
            target_idx
        )

        projected_point = apply_homography(
            source_point.reshape(1, 2),
            H
        )[0]

        error = np.linalg.norm(
            projected_point - target_point
        )

        errors.append(error)

        records.append(
            {
                "source_idx": source_idx,
                "target_idx": target_idx,
                "score": score,
                "error": error
            }
        )

    return records


def evaluate_fine_matches(
    refined_matches,
    H
):
    """
    Evaluate fine matches against homography.

    Feature-map coordinates are converted to
    256x256 image coordinates.
    """

    errors = []

    records = []

    scale = IMAGE_SIZE / FINE_GRID

    for match in refined_matches:

        source_x = (
            match["source_feature_x"]
            * scale
        )

        source_y = (
            match["source_feature_y"]
            * scale
        )

        target_x = (
            match["target_feature_x"]
            * scale
        )

        target_y = (
            match["target_feature_y"]
            * scale
        )

        source_point = np.array(
            [
                source_x,
                source_y
            ],
            dtype=np.float64
        )

        target_point = np.array(
            [
                target_x,
                target_y
            ],
            dtype=np.float64
        )

        projected_point = apply_homography(
            source_point.reshape(1, 2),
            H
        )[0]

        error = np.linalg.norm(
            projected_point - target_point
        )

        errors.append(error)

        record = match.copy()

        record["source_x_256"] = source_x
        record["source_y_256"] = source_y
        record["target_x_256"] = target_x
        record["target_y_256"] = target_y
        record["error"] = error

        records.append(record)

    return records


# ============================================================
# Valid ground-truth coarse correspondences
# ============================================================

def count_valid_source_tokens(H):
    """
    Count coarse source-token centers whose
    homography-projected point lies inside
    the target image.

    This provides the denominator for the
    coarse geometric recall calculation.
    """

    valid_count = 0

    for index in range(
        COARSE_GRID * COARSE_GRID
    ):

        source_point = (
            coarse_token_to_image_coordinate(index)
        )

        projected = apply_homography(
            source_point.reshape(1, 2),
            H
        )[0]

        x, y = projected

        if (
            0 <= x < IMAGE_SIZE
            and 0 <= y < IMAGE_SIZE
        ):
            valid_count += 1

    return valid_count


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    records,
    valid_source_count
):
    """
    Calculate precision, recall and error metrics.
    """

    count = len(records)

    if count == 0:

        return {
            "matches": 0,
            "correct": 0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_error": np.nan,
            "median_error": np.nan,
            "within_3": 0,
            "within_5": 0,
            "within_10": 0
        }

    errors = np.array(
        [
            r["error"]
            for r in records
        ]
    )

    correct = int(
        np.sum(
            errors <= GEOMETRIC_THRESHOLD
        )
    )

    precision = (
        correct / count
        if count > 0
        else 0.0
    )

    recall = (
        correct / valid_source_count
        if valid_source_count > 0
        else 0.0
    )

    return {
        "matches": count,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "mean_error": float(
            np.mean(errors)
        ),
        "median_error": float(
            np.median(errors)
        ),
        "within_3": int(
            np.sum(errors <= 3.0)
        ),
        "within_5": int(
            np.sum(errors <= 5.0)
        ),
        "within_10": int(
            np.sum(errors <= 10.0)
        )
    }


# ============================================================
# Plotting
# ============================================================

def save_threshold_plot(results):
    """
    Plot threshold versus number of matches,
    precision and recall.
    """

    thresholds = [
        r["threshold"]
        for r in results
    ]

    matches = [
        r["coarse_matches"]
        for r in results
    ]

    precision = [
        r["coarse_precision"] * 100
        for r in results
    ]

    recall = [
        r["coarse_recall"] * 100
        for r in results
    ]

    fig, ax1 = plt.subplots()

    ax1.plot(
        thresholds,
        matches,
        marker="o",
        label="Coarse matches"
    )

    ax1.set_xlabel(
        "Similarity threshold"
    )

    ax1.set_ylabel(
        "Number of coarse matches"
    )

    ax1.grid(True)

    ax2 = ax1.twinx()

    ax2.plot(
        thresholds,
        precision,
        marker="s",
        label="Precision (%)"
    )

    ax2.plot(
        thresholds,
        recall,
        marker="^",
        label="Recall (%)"
    )

    ax2.set_ylabel(
        "Percentage"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best"
    )

    plt.title(
        "Coarse Matching Threshold Sensitivity"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "threshold_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


def save_error_distribution(
    coarse_records,
    fine_records
):
    """
    Save coarse and fine error distributions.
    """

    if len(coarse_records) > 0:

        coarse_errors = [
            r["error"]
            for r in coarse_records
        ]

        plt.figure()

        plt.hist(
            coarse_errors,
            bins=20
        )

        plt.xlabel(
            "Geometric error (pixels)"
        )

        plt.ylabel(
            "Number of matches"
        )

        plt.title(
            "Coarse Matching Error Distribution"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                "coarse_error_distribution.png"
            ),
            dpi=200
        )

        plt.close()

    if len(fine_records) > 0:

        fine_errors = [
            r["error"]
            for r in fine_records
        ]

        plt.figure()

        plt.hist(
            fine_errors,
            bins=20
        )

        plt.xlabel(
            "Geometric error (pixels)"
        )

        plt.ylabel(
            "Number of matches"
        )

        plt.title(
            "Fine Matching Error Distribution"
        )

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                "fine_error_distribution.png"
            ),
            dpi=200
        )

        plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Stage 3 - Task 7")
    print("Quantitative Evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    image1, resized1, tensor1, shape1 = load_image(
        IMAGE1_PATH
    )

    image2, resized2, tensor2, shape2 = load_image(
        IMAGE2_PATH
    )

    print()
    print("Image 1 original shape:", shape1)
    print("Image 2 original shape:", shape2)
    print("Resized image shape:", resized1.shape)

    # --------------------------------------------------------
    # Load homography
    # --------------------------------------------------------

    H_original = load_homography(
        HOMOGRAPHY_PATH
    )

    H_resized = resize_homography(
        H_original,
        shape1,
        shape2
    )

    print()
    print("Original homography:")
    print(H_original)

    print()
    print("Resized homography:")
    print(H_resized)

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    cnn = DenseCNN()

    transformer = TransformerMatching(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS
    )

    cnn.eval()
    transformer.eval()

    # --------------------------------------------------------
    # CNN feature extraction
    # --------------------------------------------------------

    feature_map1 = extract_features(
        cnn,
        tensor1
    )

    feature_map2 = extract_features(
        cnn,
        tensor2
    )

    print()
    print("CNN feature map image 1:",
          tuple(feature_map1.shape))

    print("CNN feature map image 2:",
          tuple(feature_map2.shape))

    # --------------------------------------------------------
    # Coarse tokens
    # --------------------------------------------------------

    tokens1 = create_coarse_tokens(
        feature_map1
    )

    tokens2 = create_coarse_tokens(
        feature_map2
    )

    print()
    print("Coarse tokens image 1:",
          tuple(tokens1.shape))

    print("Coarse tokens image 2:",
          tuple(tokens2.shape))

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    with torch.no_grad():

        contextual1, contextual2 = transformer(
            tokens1,
            tokens2
        )

    print()
    print("Contextualized features image 1:",
          tuple(contextual1.shape))

    print("Contextualized features image 2:",
          tuple(contextual2.shape))

    # --------------------------------------------------------
    # Similarity matrix
    # --------------------------------------------------------

    similarity = cosine_similarity_matrix(
        contextual1,
        contextual2
    )

    print()
    print("Similarity matrix:",
          tuple(similarity.shape))

    print(
        "Similarity minimum:",
        similarity.min().item()
    )

    print(
        "Similarity maximum:",
        similarity.max().item()
    )

    print(
        "Similarity mean:",
        similarity.mean().item()
    )

    # --------------------------------------------------------
    # Valid GT source-token count
    # --------------------------------------------------------

    valid_source_count = count_valid_source_tokens(
        H_resized
    )

    print()
    print(
        "Ground-truth valid source tokens:",
        valid_source_count
    )

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    evaluation_results = []

    all_coarse_records = []

    selected_fine_records = []

    for threshold in SIMILARITY_THRESHOLDS:

        print()
        print("-" * 60)
        print(
            f"Similarity threshold: {threshold:.2f}"
        )

        matches = mutual_nearest_matches(
            similarity,
            threshold
        )

        print(
            "Predicted coarse matches:",
            len(matches)
        )

        coarse_records = evaluate_coarse_matches(
            matches,
            H_resized
        )

        coarse_metrics = calculate_metrics(
            coarse_records,
            valid_source_count
        )

        print(
            "Correct coarse matches:",
            coarse_metrics["correct"]
        )

        print(
            "Coarse precision:",
            f"{coarse_metrics['precision'] * 100:.2f}%"
        )

        print(
            "Coarse recall:",
            f"{coarse_metrics['recall'] * 100:.2f}%"
        )

        if not np.isnan(
            coarse_metrics["mean_error"]
        ):

            print(
                "Coarse mean error:",
                f"{coarse_metrics['mean_error']:.4f} px"
            )

            print(
                "Coarse median error:",
                f"{coarse_metrics['median_error']:.4f} px"
            )

        # ----------------------------------------------------
        # Fine refinement using predicted coarse matches
        # ----------------------------------------------------

        refined_matches = fine_refinement(
            feature_map1,
            feature_map2,
            matches
        )

        fine_records = evaluate_fine_matches(
            refined_matches,
            H_resized
        )

        fine_metrics = calculate_metrics(
            fine_records,
            valid_source_count
        )

        print(
            "Fine refined matches:",
            fine_metrics["matches"]
        )

        if fine_metrics["matches"] > 0:

            print(
                "Fine correct matches:",
                fine_metrics["correct"]
            )

            print(
                "Fine precision:",
                f"{fine_metrics['precision'] * 100:.2f}%"
            )

            print(
                "Fine mean error:",
                f"{fine_metrics['mean_error']:.4f} px"
            )

            print(
                "Fine median error:",
                f"{fine_metrics['median_error']:.4f} px"
            )

            print(
                "Fine within 3 px:",
                fine_metrics["within_3"]
            )

            print(
                "Fine within 5 px:",
                fine_metrics["within_5"]
            )

            print(
                "Fine within 10 px:",
                fine_metrics["within_10"]
            )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = {

            "threshold": threshold,

            "coarse_matches":
                coarse_metrics["matches"],

            "coarse_correct":
                coarse_metrics["correct"],

            "coarse_precision":
                coarse_metrics["precision"],

            "coarse_recall":
                coarse_metrics["recall"],

            "coarse_mean_error":
                coarse_metrics["mean_error"],

            "coarse_median_error":
                coarse_metrics["median_error"],

            "coarse_within_3":
                coarse_metrics["within_3"],

            "coarse_within_5":
                coarse_metrics["within_5"],

            "coarse_within_10":
                coarse_metrics["within_10"],

            "fine_matches":
                fine_metrics["matches"],

            "fine_correct":
                fine_metrics["correct"],

            "fine_precision":
                fine_metrics["precision"],

            "fine_recall":
                fine_metrics["recall"],

            "fine_mean_error":
                fine_metrics["mean_error"],

            "fine_median_error":
                fine_metrics["median_error"],

            "fine_within_3":
                fine_metrics["within_3"],

            "fine_within_5":
                fine_metrics["within_5"],

            "fine_within_10":
                fine_metrics["within_10"]
        }

        evaluation_results.append(result)

        if threshold == 0.50:

            all_coarse_records = coarse_records

            selected_fine_records = fine_records

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        "evaluation_results.csv"
    )

    fieldnames = list(
        evaluation_results[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            evaluation_results
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        RESULTS_DIR,
        "evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        file.write(
            "Stage 3 - Task 7\n"
        )

        file.write(
            "Quantitative Evaluation\n"
        )

        file.write(
            "==============================\n\n"
        )

        file.write(
            "Dataset: HPatches v_soldiers\n"
        )

        file.write(
            "Images: 1.ppm and 2.ppm\n"
        )

        file.write(
            f"Original image 1 shape: {shape1}\n"
        )

        file.write(
            f"Original image 2 shape: {shape2}\n"
        )

        file.write(
            "Resized shape: (256, 256)\n\n"
        )

        file.write(
            f"Coarse grid: {COARSE_GRID}x{COARSE_GRID}\n"
        )

        file.write(
            f"Fine feature grid: {FINE_GRID}x{FINE_GRID}\n"
        )

        file.write(
            f"Embedding dimension: {EMBED_DIM}\n"
        )

        file.write(
            f"Attention heads: {NUM_HEADS}\n"
        )

        file.write(
            f"Fine search radius: {FINE_RADIUS}\n"
        )

        file.write(
            f"Geometric correctness threshold: "
            f"{GEOMETRIC_THRESHOLD} px\n\n"
        )

        file.write(
            f"Valid GT source tokens: "
            f"{valid_source_count}\n\n"
        )

        file.write(
            "Threshold Results\n"
        )

        file.write(
            "------------------------------\n"
        )

        for result in evaluation_results:

            file.write(
                f"\nThreshold: "
                f"{result['threshold']:.2f}\n"
            )

            file.write(
                f"Coarse matches: "
                f"{result['coarse_matches']}\n"
            )

            file.write(
                f"Coarse correct: "
                f"{result['coarse_correct']}\n"
            )

            file.write(
                f"Coarse precision: "
                f"{result['coarse_precision'] * 100:.2f}%\n"
            )

            file.write(
                f"Coarse recall: "
                f"{result['coarse_recall'] * 100:.2f}%\n"
            )

            file.write(
                f"Coarse mean error: "
                f"{result['coarse_mean_error']}\n"
            )

            file.write(
                f"Coarse median error: "
                f"{result['coarse_median_error']}\n"
            )

            file.write(
                f"Fine matches: "
                f"{result['fine_matches']}\n"
            )

            file.write(
                f"Fine correct: "
                f"{result['fine_correct']}\n"
            )

            file.write(
                f"Fine precision: "
                f"{result['fine_precision'] * 100:.2f}%\n"
            )

            file.write(
                f"Fine recall: "
                f"{result['fine_recall'] * 100:.2f}%\n"
            )

            file.write(
                f"Fine mean error: "
                f"{result['fine_mean_error']}\n"
            )

            file.write(
                f"Fine median error: "
                f"{result['fine_median_error']}\n"
            )

        file.write(
            "\n\nInterpretation\n"
        )

        file.write(
            "------------------------------\n"
        )

        file.write(
            "The CNN and Transformer are randomly "
            "initialized and untrained.\n"
        )

        file.write(
            "Therefore the measurements evaluate "
            "the implemented pipeline rather than "
            "a trained LoFTR model.\n"
        )

        file.write(
            "Fine refinement is performed around "
            "the predicted coarse target location, "
            "not the ground-truth location.\n"
        )

        file.write(
            "The geometric recall denominator is "
            "the number of source coarse-token "
            "centers whose homography projection "
            "lies inside the target image.\n"
        )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    save_threshold_plot(
        evaluation_results
    )

    save_error_distribution(
        all_coarse_records,
        selected_fine_records
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Task 7 completed.")
    print("=" * 60)

    print()
    print("Results saved to:")
    print(RESULTS_DIR)

    print()
    print("Generated files:")
    print("  evaluation_results.csv")
    print("  evaluation_summary.txt")
    print("  threshold_analysis.png")
    print("  coarse_error_distribution.png")
    print("  fine_error_distribution.png")


if __name__ == "__main__":
    main()