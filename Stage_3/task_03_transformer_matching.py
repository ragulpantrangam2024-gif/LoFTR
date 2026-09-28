import os
import cv2
import numpy as np
import torch
import torch.nn as nn
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

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_03"
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

torch.manual_seed(42)


# ============================================================
# CNN Feature Extractor
# Same architecture as Task 1
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
        image_float,
        tensor,
        original_shape
    )


# ============================================================
# 2D Sinusoidal Positional Encoding
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

    # Y position

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

    # X position

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
# Convert feature map to coarse tokens
# ============================================================

def create_coarse_tokens(
    feature_map
):

    # Original:
    #
    # [1,64,128,128]
    #
    # Pool to:
    #
    # [1,64,16,16]

    pooled = nn.functional.adaptive_avg_pool2d(
        feature_map,
        (
            COARSE_SIZE,
            COARSE_SIZE
        )
    )

    batch, channels, height, width = pooled.shape

    # [1,16,16,64]

    features = pooled.permute(
        0,
        2,
        3,
        1
    )

    # [1,256,64]

    tokens = features.reshape(
        batch,
        height * width,
        channels
    )

    # Create positional encoding

    positional_encoding = create_2d_positional_encoding(
        height,
        width,
        channels
    )

    positional_encoding = positional_encoding.reshape(
        height * width,
        channels
    )

    positional_encoding = positional_encoding.unsqueeze(0)

    # Add position

    tokens_with_position = (
        tokens
        + positional_encoding
    )

    return (
        pooled,
        tokens,
        positional_encoding,
        tokens_with_position
    )


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

        # Self-attention

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

        # Cross-attention

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

        # Normalization

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

        # ====================================================
        # Self-attention: Image 1
        # ====================================================

        self_1, weights_self_1 = (
            self.self_attention_1(
                features_1,
                features_1,
                features_1,
                need_weights=True,
                average_attn_weights=False
            )
        )

        features_1 = self.norm_1(
            features_1 + self_1
        )

        # ====================================================
        # Self-attention: Image 2
        # ====================================================

        self_2, weights_self_2 = (
            self.self_attention_2(
                features_2,
                features_2,
                features_2,
                need_weights=True,
                average_attn_weights=False
            )
        )

        features_2 = self.norm_2(
            features_2 + self_2
        )

        # ====================================================
        # Cross-attention
        #
        # Image 1 queries Image 2
        # ====================================================

        cross_1, weights_cross_1 = (
            self.cross_attention_1(
                features_1,
                features_2,
                features_2,
                need_weights=True,
                average_attn_weights=False
            )
        )

        features_1 = self.norm_3(
            features_1 + cross_1
        )

        # ====================================================
        # Cross-attention
        #
        # Image 2 queries Image 1
        # ====================================================

        cross_2, weights_cross_2 = (
            self.cross_attention_2(
                features_2,
                features_1,
                features_1,
                need_weights=True,
                average_attn_weights=False
            )
        )

        features_2 = self.norm_4(
            features_2 + cross_2
        )

        return (
            features_1,
            features_2,
            weights_self_1,
            weights_self_2,
            weights_cross_1,
            weights_cross_2
        )


# ============================================================
# Visualize attention
# ============================================================

def save_attention_visualization(
    attention,
    filename,
    title
):

    # attention:
    #
    # [batch, heads, query, key]

    attention = attention[
        0
    ]

    # Average across heads

    attention_mean = attention.mean(
        dim=0
    ).detach().cpu().numpy()

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        attention_mean
    )

    plt.title(title)

    plt.xlabel(
        "Key token"
    )

    plt.ylabel(
        "Query token"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        dpi=150
    )

    plt.close()

    return attention_mean


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)

    print(
        "Stage 3 - Task 3"
    )

    print(
        "Transformer Matching Module"
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
        "\nCNN feature map image 1:",
        tuple(feature_map_1.shape)
    )

    print(
        "CNN feature map image 2:",
        tuple(feature_map_2.shape)
    )

    # --------------------------------------------------------
    # Create coarse tokens
    # --------------------------------------------------------

    (
        pooled_1,
        tokens_1,
        pos_1,
        tokens_pos_1
    ) = create_coarse_tokens(
        feature_map_1
    )

    (
        pooled_2,
        tokens_2,
        pos_2,
        tokens_pos_2
    ) = create_coarse_tokens(
        feature_map_2
    )

    print(
        "\nCoarse feature map:",
        tuple(pooled_1.shape)
    )

    print(
        "Image 1 coarse tokens:",
        tuple(tokens_pos_1.shape)
    )

    print(
        "Image 2 coarse tokens:",
        tuple(tokens_pos_2.shape)
    )

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    transformer = TransformerMatchingModule(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS
    )

    transformer.eval()

    with torch.no_grad():

        (
            output_1,
            output_2,
            self_weights_1,
            self_weights_2,
            cross_weights_1,
            cross_weights_2
        ) = transformer(
            tokens_pos_1,
            tokens_pos_2
        )

    # --------------------------------------------------------
    # Output shapes
    # --------------------------------------------------------

    print(
        "\nTransformer output image 1:",
        tuple(output_1.shape)
    )

    print(
        "Transformer output image 2:",
        tuple(output_2.shape)
    )

    print(
        "\nSelf-attention image 1:",
        tuple(self_weights_1.shape)
    )

    print(
        "Self-attention image 2:",
        tuple(self_weights_2.shape)
    )

    print(
        "Cross-attention image 1 -> image 2:",
        tuple(cross_weights_1.shape)
    )

    print(
        "Cross-attention image 2 -> image 1:",
        tuple(cross_weights_2.shape)
    )

    # --------------------------------------------------------
    # Check attention row sums
    # --------------------------------------------------------

    self_row_sum = (
        self_weights_1.sum(
            dim=-1
        )
    )

    cross_row_sum = (
        cross_weights_1.sum(
            dim=-1
        )
    )

    print(
        "\nSelf-attention row sum:"
    )

    print(
        "  Min:",
        self_row_sum.min().item()
    )

    print(
        "  Max:",
        self_row_sum.max().item()
    )

    print(
        "\nCross-attention row sum:"
    )

    print(
        "  Min:",
        cross_row_sum.min().item()
    )

    print(
        "  Max:",
        cross_row_sum.max().item()
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    self_mean = save_attention_visualization(
        self_weights_1,
        "self_attention_image1.png",
        "Image 1 Self-Attention"
    )

    save_attention_visualization(
        self_weights_2,
        "self_attention_image2.png",
        "Image 2 Self-Attention"
    )

    cross_mean = save_attention_visualization(
        cross_weights_1,
        "cross_attention_image1_to_image2.png",
        "Cross-Attention: Image 1 → Image 2"
    )

    save_attention_visualization(
        cross_weights_2,
        "cross_attention_image2_to_image1.png",
        "Cross-Attention: Image 2 → Image 1"
    )

    # --------------------------------------------------------
    # Save feature norm comparison
    # --------------------------------------------------------

    input_norm_1 = torch.norm(
        tokens_pos_1,
        dim=2
    ).mean().item()

    input_norm_2 = torch.norm(
        tokens_pos_2,
        dim=2
    ).mean().item()

    output_norm_1 = torch.norm(
        output_1,
        dim=2
    ).mean().item()

    output_norm_2 = torch.norm(
        output_2,
        dim=2
    ).mean().item()

    # --------------------------------------------------------
    # Save numerical summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        RESULTS_DIR,
        "task_03_results.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 3 - Task 3: Transformer Matching Module\n"
        )

        f.write("=" * 60 + "\n\n")

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
            f"CNN feature map: "
            f"{tuple(feature_map_1.shape)}\n"
        )

        f.write(
            f"Coarse feature map: "
            f"{tuple(pooled_1.shape)}\n"
        )

        f.write(
            f"Coarse tokens: "
            f"{tuple(tokens_pos_1.shape)}\n\n"
        )

        f.write(
            f"Embedding dimension: {EMBED_DIM}\n"
        )

        f.write(
            f"Attention heads: {NUM_HEADS}\n\n"
        )

        f.write(
            f"Transformer output image 1: "
            f"{tuple(output_1.shape)}\n"
        )

        f.write(
            f"Transformer output image 2: "
            f"{tuple(output_2.shape)}\n\n"
        )

        f.write(
            f"Self-attention shape: "
            f"{tuple(self_weights_1.shape)}\n"
        )

        f.write(
            f"Cross-attention shape: "
            f"{tuple(cross_weights_1.shape)}\n\n"
        )

        f.write(
            f"Self-attention row sum min: "
            f"{self_row_sum.min().item():.9f}\n"
        )

        f.write(
            f"Self-attention row sum max: "
            f"{self_row_sum.max().item():.9f}\n"
        )

        f.write(
            f"Cross-attention row sum min: "
            f"{cross_row_sum.min().item():.9f}\n"
        )

        f.write(
            f"Cross-attention row sum max: "
            f"{cross_row_sum.max().item():.9f}\n\n"
        )

        f.write(
            f"Mean input token norm image 1: "
            f"{input_norm_1:.6f}\n"
        )

        f.write(
            f"Mean input token norm image 2: "
            f"{input_norm_2:.6f}\n"
        )

        f.write(
            f"Mean output token norm image 1: "
            f"{output_norm_1:.6f}\n"
        )

        f.write(
            f"Mean output token norm image 2: "
            f"{output_norm_2:.6f}\n"
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nTask 3 completed."
    )


if __name__ == "__main__":
    main()