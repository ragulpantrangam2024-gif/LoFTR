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
    "task_04"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256

EMBED_DIM = 64

COARSE_SIZE = 16

NUM_HEADS = 8

SIMILARITY_THRESHOLD = 0.80

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
# Load image
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
# 2D positional encoding
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
# Similarity matrix
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

def mutual_nearest_neighbor_matching(
    similarity,
    threshold
):

    similarity = similarity[0]

    # Best Image 2 token for every Image 1 token

    best_2 = torch.argmax(
        similarity,
        dim=1
    )

    best_scores_1 = torch.max(
        similarity,
        dim=1
    ).values

    # Best Image 1 token for every Image 2 token

    best_1 = torch.argmax(
        similarity,
        dim=0
    )

    matches = []

    for i in range(
        similarity.shape[0]
    ):

        j = best_2[i]

        score = best_scores_1[i]

        # Mutual condition

        if best_1[j] != i:

            continue

        # Confidence threshold

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
# Token index → pixel coordinate
# ============================================================

def token_to_pixel(
    token_index
):

    row = token_index // COARSE_SIZE

    col = token_index % COARSE_SIZE

    cell_width = IMAGE_SIZE / COARSE_SIZE

    cell_height = IMAGE_SIZE / COARSE_SIZE

    x = (
        col + 0.5
    ) * cell_width

    y = (
        row + 0.5
    ) * cell_height

    return (
        x,
        y
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
# Scale homography to resized images
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

    return H_resized / H_resized[2, 2]


# ============================================================
# Apply homography
# ============================================================

def apply_homography(
    H,
    point
):

    x, y = point

    p = np.array([
        x,
        y,
        1.0
    ])

    transformed = H @ p

    transformed = (
        transformed
        / transformed[2]
    )

    return (
        transformed[0],
        transformed[1]
    )


# ============================================================
# Evaluate matches
# ============================================================

def evaluate_matches(
    matches,
    H
):

    errors = []

    valid_matches = []

    for (
        token1,
        token2,
        score
    ) in matches:

        point1 = token_to_pixel(
            token1
        )

        point2 = token_to_pixel(
            token2
        )

        projected = apply_homography(
            H,
            point1
        )

        error = np.linalg.norm(
            np.array(projected)
            - np.array(point2)
        )

        errors.append(
            float(error)
        )

        valid_matches.append(
            (
                point1,
                point2,
                score,
                float(error)
            )
        )

    if len(errors) == 0:

        return {
            "errors": [],
            "valid_matches": [],
            "mean": None,
            "median": None,
            "within_3": 0,
            "within_5": 0,
            "within_10": 0,
            "accuracy": None
        }

    errors = np.array(
        errors
    )

    return {
        "errors": errors,
        "valid_matches": valid_matches,
        "mean": float(
            np.mean(errors)
        ),
        "median": float(
            np.median(errors)
        ),
        "within_3": int(
            np.sum(errors <= 3.0)
        ),
        "within_5": int(
            np.sum(errors <= 5.0)
        ),
        "within_10": int(
            np.sum(
                errors <= GEOMETRIC_THRESHOLD
            )
        ),
        "accuracy": float(
            np.mean(
                errors <= GEOMETRIC_THRESHOLD
            ) * 100.0
        )
    }


# ============================================================
# Visualize matches
# ============================================================

def draw_matches(
    image1,
    image2,
    matches
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

    offset = IMAGE_SIZE

    for (
        point1,
        point2,
        score,
        error
    ) in matches:

        x1, y1 = point1
        x2, y2 = point2

        x2_display = (
            x2 + offset
        )

        plt.plot(
            [x1, x2_display],
            [y1, y2],
            linewidth=0.7
        )

        plt.scatter(
            [x1, x2_display],
            [y1, y2],
            s=8
        )

    plt.title(
        f"Coarse Matches: {len(matches)}"
    )

    plt.axis("off")

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
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Stage 3 - Task 4"
    )

    print(
        "Coarse Matching"
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
    # Coarse tokens
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
        "\nContextualized features image 1:",
        tuple(contextual_1.shape)
    )

    print(
        "Contextualized features image 2:",
        tuple(contextual_2.shape)
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
        "\nSimilarity matrix:",
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
    # Visualize similarity matrix
    # --------------------------------------------------------

    similarity_image = (
        similarity[0]
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        similarity_image,
        aspect="auto"
    )

    plt.title(
        "Coarse Feature Similarity"
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
    # Matching
    # --------------------------------------------------------

    matches = mutual_nearest_neighbor_matching(
        similarity,
        SIMILARITY_THRESHOLD
    )

    print(
        "\nSimilarity threshold:",
        SIMILARITY_THRESHOLD
    )

    print(
        "Mutual coarse matches:",
        len(matches)
    )

    # --------------------------------------------------------
    # Ground-truth homography
    # --------------------------------------------------------

    H_original = load_homography()

    H_resized = scale_homography(
        H_original,
        shape1,
        shape2
    )

    print(
        "\nOriginal homography:"
    )

    print(
        H_original
    )

    print(
        "\nResized homography:"
    )

    print(
        H_resized
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluation = evaluate_matches(
        matches,
        H_resized
    )

    if evaluation["mean"] is not None:

        print(
            "\nCoarse matching evaluation:"
        )

        print(
            "Mean error:",
            evaluation["mean"]
        )

        print(
            "Median error:",
            evaluation["median"]
        )

        print(
            "Within 3 px:",
            evaluation["within_3"],
            "/",
            len(matches)
        )

        print(
            "Within 5 px:",
            evaluation["within_5"],
            "/",
            len(matches)
        )

        print(
            "Within 10 px:",
            evaluation["within_10"],
            "/",
            len(matches)
        )

        print(
            "Geometric accuracy:",
            evaluation["accuracy"],
            "%"
        )

    else:

        print(
            "\nNo matches passed the threshold."
        )

    # --------------------------------------------------------
    # Draw matches
    # --------------------------------------------------------

    draw_matches(
        image1,
        image2,
        evaluation["valid_matches"]
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    summary_path = os.path.join(
        RESULTS_DIR,
        "task_04_results.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 3 - Task 4: Coarse Matching\n"
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
            f"Image 1 original shape: {shape1}\n"
        )

        f.write(
            f"Image 2 original shape: {shape2}\n\n"
        )

        f.write(
            f"Coarse tokens image 1: "
            f"{tuple(tokens_1.shape)}\n"
        )

        f.write(
            f"Coarse tokens image 2: "
            f"{tuple(tokens_2.shape)}\n"
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
            f"Mutual coarse matches: "
            f"{len(matches)}\n\n"
        )

        if evaluation["mean"] is not None:

            f.write(
                f"Mean error: "
                f"{evaluation['mean']:.6f} px\n"
            )

            f.write(
                f"Median error: "
                f"{evaluation['median']:.6f} px\n"
            )

            f.write(
                f"Within 3 px: "
                f"{evaluation['within_3']} / "
                f"{len(matches)}\n"
            )

            f.write(
                f"Within 5 px: "
                f"{evaluation['within_5']} / "
                f"{len(matches)}\n"
            )

            f.write(
                f"Within 10 px: "
                f"{evaluation['within_10']} / "
                f"{len(matches)}\n"
            )

            f.write(
                f"Geometric accuracy: "
                f"{evaluation['accuracy']:.4f}%\n"
            )

        else:

            f.write(
                "No matches passed the similarity threshold.\n"
            )

        f.write(
            "\nResized homography:\n"
        )

        f.write(
            np.array2string(
                H_resized,
                precision=8
            )
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nTask 4 completed."
    )


if __name__ == "__main__":
    main()