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
    "task_06"
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

NUM_HEADS = 8

SIMILARITY_THRESHOLD = 0.80

FINE_SEARCH_RADIUS = 4

NUM_ORACLE_POINTS = 50

GEOMETRIC_THRESHOLD = 10.0

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
# Positional encoding
# ============================================================

def create_2d_positional_encoding(
    height,
    width,
    dim
):

    if dim % 4 != 0:

        raise ValueError(
            "Embedding dimension must be divisible by 4."
        )

    pe = torch.zeros(
        height,
        width,
        dim
    )

    quarter_dim = dim // 4

    position_y = torch.arange(
        height,
        dtype=torch.float32
    ).unsqueeze(1)

    position_x = torch.arange(
        width,
        dtype=torch.float32
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            quarter_dim,
            dtype=torch.float32
        )
        * -(
            np.log(10000.0)
            / quarter_dim
        )
    )

    pe[
        :,
        :,
        0:quarter_dim
    ] = (
        torch.sin(
            position_y * div_term
        )
        .unsqueeze(1)
    )

    pe[
        :,
        :,
        quarter_dim:2 * quarter_dim
    ] = (
        torch.cos(
            position_y * div_term
        )
        .unsqueeze(1)
    )

    pe[
        :,
        :,
        2 * quarter_dim:3 * quarter_dim
    ] = (
        torch.sin(
            position_x * div_term
        )
        .unsqueeze(0)
    )

    pe[
        :,
        :,
        3 * quarter_dim:4 * quarter_dim
    ] = (
        torch.cos(
            position_x * div_term
        )
        .unsqueeze(0)
    )

    return pe


# ============================================================
# Create coarse tokens
# ============================================================

def create_coarse_tokens(
    feature_map
):

    pooled = F.adaptive_avg_pool2d(
        feature_map,
        (
            COARSE_SIZE,
            COARSE_SIZE
        )
    )

    batch, channels, height, width = (
        pooled.shape
    )

    features = pooled.permute(
        0,
        2,
        3,
        1
    )

    tokens = features.reshape(
        batch,
        height * width,
        channels
    )

    positional_encoding = (
        create_2d_positional_encoding(
            height,
            width,
            channels
        )
    )

    positional_encoding = (
        positional_encoding.reshape(
            height * width,
            channels
        )
        .unsqueeze(0)
    )

    tokens = (
        tokens
        + positional_encoding
    )

    return tokens


# ============================================================
# Transformer Matching Module
# ============================================================

class TransformerMatchingModule(
    nn.Module
):

    def __init__(
        self,
        embed_dim,
        num_heads
    ):

        super().__init__()

        self.self_attention_1 = (
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
        )

        self.self_attention_2 = (
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
        )

        self.cross_attention_1 = (
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
        )

        self.cross_attention_2 = (
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
        )

        self.norm_1 = nn.LayerNorm(
            embed_dim
        )

        self.norm_2 = nn.LayerNorm(
            embed_dim
        )

        self.norm_3 = nn.LayerNorm(
            embed_dim
        )

        self.norm_4 = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        features_1,
        features_2
    ):

        self_1, _ = (
            self.self_attention_1(
                features_1,
                features_1,
                features_1,
                need_weights=False
            )
        )

        features_1 = self.norm_1(
            features_1 + self_1
        )

        self_2, _ = (
            self.self_attention_2(
                features_2,
                features_2,
                features_2,
                need_weights=False
            )
        )

        features_2 = self.norm_2(
            features_2 + self_2
        )

        cross_1, _ = (
            self.cross_attention_1(
                features_1,
                features_2,
                features_2,
                need_weights=False
            )
        )

        features_1 = self.norm_3(
            features_1 + cross_1
        )

        cross_2, _ = (
            self.cross_attention_2(
                features_2,
                features_1,
                features_1,
                need_weights=False
            )
        )

        features_2 = self.norm_4(
            features_2 + cross_2
        )

        return (
            features_1,
            features_2
        )


# ============================================================
# Similarity
# ============================================================

def compute_similarity(
    features_1,
    features_2
):

    features_1 = F.normalize(
        features_1,
        p=2,
        dim=-1
    )

    features_2 = F.normalize(
        features_2,
        p=2,
        dim=-1
    )

    similarity = torch.matmul(
        features_1,
        features_2.transpose(
            1,
            2
        )
    )

    return similarity


# ============================================================
# Mutual nearest-neighbor matching
# ============================================================

def mutual_matches(
    similarity,
    threshold
):

    similarity = similarity[0]

    best_2 = torch.argmax(
        similarity,
        dim=1
    )

    best_scores = torch.max(
        similarity,
        dim=1
    ).values

    best_1 = torch.argmax(
        similarity,
        dim=0
    )

    matches = []

    for i in range(
        similarity.shape[0]
    ):

        j = best_2[i]

        score = best_scores[i]

        if best_1[j] != i:

            continue

        if score < threshold:

            continue

        matches.append(
            (
                i,
                int(j.item()),
                float(score.item())
            )
        )

    return matches


# ============================================================
# Token → image coordinate
# ============================================================

def token_to_pixel(
    token_index
):

    row = token_index // COARSE_SIZE

    col = token_index % COARSE_SIZE

    cell_size = (
        IMAGE_SIZE / COARSE_SIZE
    )

    x = (
        col + 0.5
    ) * cell_size

    y = (
        row + 0.5
    ) * cell_size

    return (
        x,
        y
    )


# ============================================================
# Homography
# ============================================================

def load_homography():

    H = np.loadtxt(
        HOMOGRAPHY_PATH
    )

    return H


def scale_homography(
    H_original,
    shape1,
    shape2
):

    h1, w1 = shape1
    h2, w2 = shape2

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

    H = (
        S2
        @ H_original
        @ np.linalg.inv(S1)
    )

    return H / H[2, 2]


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
# Evaluate predicted coarse matches
# ============================================================

def evaluate_coarse_matches(
    matches,
    H
):

    results = []

    for (
        token1,
        token2,
        score
    ) in matches:

        x1, y1 = token_to_pixel(
            token1
        )

        x2, y2 = token_to_pixel(
            token2
        )

        gt_x, gt_y = apply_homography(
            H,
            x1,
            y1
        )

        error = np.linalg.norm(
            np.array([
                gt_x,
                gt_y
            ])
            -
            np.array([
                x2,
                y2
            ])
        )

        results.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "score": score,
                "error": float(error)
            }
        )

    return results


# ============================================================
# Generate oracle points
# ============================================================

def generate_oracle_points(
    H,
    num_points
):

    points = []

    for row in range(
        COARSE_SIZE
    ):

        for col in range(
            COARSE_SIZE
        ):

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

    return points[:num_points]


# ============================================================
# Image → fine feature coordinates
# ============================================================

def image_to_feature(
    x,
    y
):

    scale = (
        FEATURE_SIZE
        / IMAGE_SIZE
    )

    return (
        x * scale - 0.5,
        y * scale - 0.5
    )


def feature_to_image(
    x,
    y
):

    scale = (
        IMAGE_SIZE
        / FEATURE_SIZE
    )

    return (
        (x + 0.5) * scale,
        (y + 0.5) * scale
    )


# ============================================================
# Fine refinement
# ============================================================

def refine_point(
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

    x1_feature, y1_feature = (
        image_to_feature(
            x1_image,
            y1_image
        )
    )

    x1_index = int(
        round(x1_feature)
    )

    y1_index = int(
        round(y1_feature)
    )

    descriptor_1 = feature_map_1[
        0,
        :,
        y1_index,
        x1_index
    ]

    descriptor_1 = F.normalize(
        descriptor_1,
        p=2,
        dim=0
    )

    x2_feature, y2_feature = (
        image_to_feature(
            x2_true_image,
            y2_true_image
        )
    )

    center_x = int(
        round(x2_feature)
    )

    center_y = int(
        round(y2_feature)
    )

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
                or x >= FEATURE_SIZE
                or y < 0
                or y >= FEATURE_SIZE
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

    best = max(
        candidates,
        key=lambda item: item[2]
    )

    pred_x, pred_y = (
        feature_to_image(
            best[0],
            best[1]
        )
    )

    error = np.linalg.norm(
        np.array([
            pred_x,
            pred_y
        ])
        -
        np.array([
            x2_true_image,
            y2_true_image
        ])
    )

    return {
        "x1": x1_image,
        "y1": y1_image,
        "true_x": x2_true_image,
        "true_y": y2_true_image,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "similarity": best[2],
        "error": float(error)
    }


# ============================================================
# Draw predicted coarse matches
# ============================================================

def draw_coarse_matches(
    image1,
    image2,
    results
):

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

        plt.plot(
            [
                result["x1"],
                result["x2"] + IMAGE_SIZE
            ],
            [
                result["y1"],
                result["y2"]
            ],
            linewidth=0.8
        )

    plt.title(
        f"Predicted Coarse Matches: {len(results)}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "predicted_coarse_matches.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# Draw oracle fine refinement
# ============================================================

def draw_oracle_refinement(
    image1,
    image2,
    results
):

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

        plt.plot(
            [
                result["x1"],
                result["pred_x"] + IMAGE_SIZE
            ],
            [
                result["y1"],
                result["pred_y"]
            ],
            linewidth=0.7
        )

    plt.title(
        f"Oracle-Guided Fine Refinement: "
        f"{len(results)} points"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "oracle_fine_refinement.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Stage 3 - Task 6"
    )

    print(
        "End-to-End Simplified LoFTR"
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
        "\nCNN feature map:",
        tuple(feature_map_1.shape)
    )

    # --------------------------------------------------------
    # Coarse representation
    # --------------------------------------------------------

    tokens_1 = create_coarse_tokens(
        feature_map_1
    )

    tokens_2 = create_coarse_tokens(
        feature_map_2
    )

    print(
        "Coarse tokens image 1:",
        tuple(tokens_1.shape)
    )

    print(
        "Coarse tokens image 2:",
        tuple(tokens_2.shape)
    )

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    transformer = TransformerMatchingModule(
        EMBED_DIM,
        NUM_HEADS
    )

    transformer.eval()

    with torch.no_grad():

        contextual_1, contextual_2 = (
            transformer(
                tokens_1,
                tokens_2
            )
        )

    print(
        "\nContextualized features:",
        tuple(contextual_1.shape)
    )

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    with torch.no_grad():

        similarity = compute_similarity(
            contextual_1,
            contextual_2
        )

    print(
        "Similarity matrix:",
        tuple(similarity.shape)
    )

    print(
        "Similarity min:",
        similarity.min().item()
    )

    print(
        "Similarity max:",
        similarity.max().item()
    )

    print(
        "Similarity mean:",
        similarity.mean().item()
    )

    # --------------------------------------------------------
    # Predicted coarse matching
    # --------------------------------------------------------

    predicted_matches = mutual_matches(
        similarity,
        SIMILARITY_THRESHOLD
    )

    print(
        "\nSimilarity threshold:",
        SIMILARITY_THRESHOLD
    )

    print(
        "Predicted coarse matches:",
        len(predicted_matches)
    )

    # --------------------------------------------------------
    # Homography
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
    # Evaluate predicted coarse matches
    # --------------------------------------------------------

    predicted_results = (
        evaluate_coarse_matches(
            predicted_matches,
            H
        )
    )

    if len(predicted_results) > 0:

        predicted_errors = np.array([
            result["error"]
            for result in predicted_results
        ])

        print(
            "\nPredicted coarse evaluation:"
        )

        print(
            "Mean error:",
            predicted_errors.mean()
        )

        print(
            "Median error:",
            np.median(
                predicted_errors
            )
        )

        print(
            "Within 10 px:",
            np.sum(
                predicted_errors <= 10
            ),
            "/",
            len(predicted_errors)
        )

        draw_coarse_matches(
            image1,
            image2,
            predicted_results
        )

    else:

        print(
            "\nNo confident predicted coarse matches."
        )

    # --------------------------------------------------------
    # Oracle-guided fine stage
    # --------------------------------------------------------

    oracle_points = generate_oracle_points(
        H,
        NUM_ORACLE_POINTS
    )

    oracle_results = []

    for point in oracle_points:

        result = refine_point(
            feature_map_1,
            feature_map_2,
            point
        )

        if result is not None:

            oracle_results.append(
                result
            )

    print(
        "\nOracle-guided fine stage:"
    )

    print(
        "Oracle coarse points:",
        len(oracle_points)
    )

    print(
        "Fine refined points:",
        len(oracle_results)
    )

    if len(oracle_results) > 0:

        fine_errors = np.array([
            result["error"]
            for result in oracle_results
        ])

        fine_similarities = np.array([
            result["similarity"]
            for result in oracle_results
        ])

        fine_mean = fine_errors.mean()
        fine_median = np.median(
            fine_errors
        )

        fine_3 = np.sum(
            fine_errors <= 3
        )

        fine_5 = np.sum(
            fine_errors <= 5
        )

        fine_10 = np.sum(
            fine_errors <= 10
        )

        print(
            "Fine mean error:",
            fine_mean
        )

        print(
            "Fine median error:",
            fine_median
        )

        print(
            "Fine within 3 px:",
            fine_3,
            "/",
            len(fine_errors)
        )

        print(
            "Fine within 5 px:",
            fine_5,
            "/",
            len(fine_errors)
        )

        print(
            "Fine within 10 px:",
            fine_10,
            "/",
            len(fine_errors)
        )

        print(
            "Fine accuracy within 10 px:",
            (
                fine_10
                / len(fine_errors)
                * 100
            ),
            "%"
        )

        print(
            "Mean fine similarity:",
            fine_similarities.mean()
        )

        draw_oracle_refinement(
            image1,
            image2,
            oracle_results
        )

    # --------------------------------------------------------
    # Similarity matrix visualization
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        similarity[0]
        .detach()
        .cpu()
        .numpy()
    )

    plt.title(
        "End-to-End Coarse Similarity Matrix"
    )

    plt.xlabel(
        "Image 2 token"
    )

    plt.ylabel(
        "Image 1 token"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "similarity_matrix.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        RESULTS_DIR,
        "task_06_results.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 3 - Task 6: End-to-End Simplified LoFTR\n"
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
            "IMPORTANT:\n"
        )

        f.write(
            "This is a simplified educational/research "
            "implementation inspired by LoFTR.\n"
        )

        f.write(
            "It is not the official LoFTR architecture "
            "or pretrained model.\n\n"
        )

        f.write(
            f"CNN feature map: "
            f"{tuple(feature_map_1.shape)}\n"
        )

        f.write(
            f"Coarse tokens: "
            f"{tuple(tokens_1.shape)}\n"
        )

        f.write(
            f"Transformer output: "
            f"{tuple(contextual_1.shape)}\n"
        )

        f.write(
            f"Similarity matrix: "
            f"{tuple(similarity.shape)}\n\n"
        )

        f.write(
            f"Similarity threshold: "
            f"{SIMILARITY_THRESHOLD}\n"
        )

        f.write(
            f"Predicted coarse matches: "
            f"{len(predicted_matches)}\n\n"
        )

        if len(predicted_results) > 0:

            f.write(
                f"Predicted coarse mean error: "
                f"{np.mean([r['error'] for r in predicted_results]):.6f} px\n"
            )

            f.write(
                f"Predicted coarse median error: "
                f"{np.median([r['error'] for r in predicted_results]):.6f} px\n"
            )

        else:

            f.write(
                "No confident predicted coarse matches.\n"
            )

        f.write(
            "\nOracle-guided fine experiment:\n"
        )

        f.write(
            f"Oracle coarse points: "
            f"{len(oracle_points)}\n"
        )

        f.write(
            f"Fine refined points: "
            f"{len(oracle_results)}\n"
        )

        f.write(
            "\nThe oracle fine experiment uses "
            "ground-truth homography to select the "
            "coarse search locations.\n"
        )

        f.write(
            "Therefore its results should not be "
            "interpreted as end-to-end prediction accuracy.\n"
        )

        if len(oracle_results) > 0:

            errors = np.array([
                r["error"]
                for r in oracle_results
            ])

            similarities = np.array([
                r["similarity"]
                for r in oracle_results
            ])

            f.write(
                f"\nFine mean error: "
                f"{errors.mean():.6f} px\n"
            )

            f.write(
                f"Fine median error: "
                f"{np.median(errors):.6f} px\n"
            )

            f.write(
                f"Fine within 3 px: "
                f"{np.sum(errors <= 3)} / "
                f"{len(errors)}\n"
            )

            f.write(
                f"Fine within 5 px: "
                f"{np.sum(errors <= 5)} / "
                f"{len(errors)}\n"
            )

            f.write(
                f"Fine within 10 px: "
                f"{np.sum(errors <= 10)} / "
                f"{len(errors)}\n"
            )

            f.write(
                f"Fine accuracy within 10 px: "
                f"{np.mean(errors <= 10) * 100:.4f}%\n"
            )

            f.write(
                f"Mean fine similarity: "
                f"{similarities.mean():.6f}\n"
            )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nTask 6 completed."
    )


if __name__ == "__main__":
    main()