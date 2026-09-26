import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_woman",
    "1.ppm"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_04"
)

IMAGE_SIZE = 256
PATCH_SIZE = 16

EMBED_DIM = 64
NUM_HEADS = 8
FFN_DIM = 128

SEED = 42


# ============================================================
# SETUP
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

original_shape = image.shape

image = cv2.resize(
    image,
    (IMAGE_SIZE, IMAGE_SIZE)
)


# ============================================================
# IMAGE → PATCHES
# ============================================================

def image_to_patches(image, patch_size):
    """
    Convert grayscale image into flattened patches.
    """

    tensor = torch.from_numpy(
        image
    ).float()

    patches = (
        tensor
        .unfold(0, patch_size, patch_size)
        .unfold(1, patch_size, patch_size)
    )

    rows = patches.shape[0]
    cols = patches.shape[1]

    patches = patches.contiguous().view(
        rows * cols,
        patch_size * patch_size
    )

    return patches


patches = image_to_patches(
    image,
    PATCH_SIZE
)

patches = patches / 255.0


# ============================================================
# PATCH EMBEDDING
# ============================================================

input_dim = PATCH_SIZE * PATCH_SIZE

patch_embedding = nn.Linear(
    input_dim,
    EMBED_DIM
)

tokens = patch_embedding(
    patches
)


# ============================================================
# 2D SINUSOIDAL POSITIONAL ENCODING
# ============================================================

def get_1d_sinusoidal_encoding(
    positions,
    dimension
):
    encoding = torch.zeros(
        positions,
        dimension
    )

    position = torch.arange(
        positions,
        dtype=torch.float32
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            dimension,
            2,
            dtype=torch.float32
        )
        *
        (
            -np.log(10000.0)
            / dimension
        )
    )

    encoding[:, 0::2] = torch.sin(
        position * div_term
    )

    encoding[:, 1::2] = torch.cos(
        position * div_term
    )

    return encoding


def get_2d_sinusoidal_encoding(
    rows,
    cols,
    embed_dim
):
    """
    Half of the embedding represents row position.
    Half represents column position.
    """

    if embed_dim % 4 != 0:
        raise ValueError(
            "Embedding dimension must be divisible by 4."
        )

    half_dim = embed_dim // 2

    row_encoding = get_1d_sinusoidal_encoding(
        rows,
        half_dim
    )

    col_encoding = get_1d_sinusoidal_encoding(
        cols,
        half_dim
    )

    positional_encoding = torch.zeros(
        rows,
        cols,
        embed_dim
    )

    for r in range(rows):
        for c in range(cols):

            positional_encoding[
                r,
                c,
                :half_dim
            ] = row_encoding[r]

            positional_encoding[
                r,
                c,
                half_dim:
            ] = col_encoding[c]

    return positional_encoding


num_rows = IMAGE_SIZE // PATCH_SIZE
num_cols = IMAGE_SIZE // PATCH_SIZE

positional_encoding = get_2d_sinusoidal_encoding(
    num_rows,
    num_cols,
    EMBED_DIM
)

positional_encoding = positional_encoding.view(
    num_rows * num_cols,
    EMBED_DIM
)


# ============================================================
# ADD POSITIONAL INFORMATION
# ============================================================

tokens_with_position = (
    tokens
    +
    positional_encoding
)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

# Transformer expects:
#
# batch × sequence × embedding
#
# Current:
#
# sequence × embedding
#
# Therefore:

transformer_input = (
    tokens_with_position
    .unsqueeze(0)
)


# ============================================================
# TRANSFORMER ENCODER BLOCK
# ============================================================

class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_dim
    ):

        super().__init__()

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # First LayerNorm
        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(
                embed_dim,
                ffn_dim
            ),

            nn.ReLU(),

            nn.Linear(
                ffn_dim,
                embed_dim
            )
        )

        # Second LayerNorm
        self.norm2 = nn.LayerNorm(
            embed_dim
        )

    def forward(self, x):

        # ----------------------------------------------------
        # SELF-ATTENTION
        # ----------------------------------------------------

        attention_output, attention_weights = (
            self.self_attention(
                x,
                x,
                x,
                need_weights=True,
                average_attn_weights=False
            )
        )

        # ----------------------------------------------------
        # RESIDUAL CONNECTION + NORMALIZATION
        # ----------------------------------------------------

        x_attention = self.norm1(
            x + attention_output
        )

        # ----------------------------------------------------
        # FEED-FORWARD NETWORK
        # ----------------------------------------------------

        ffn_output = self.ffn(
            x_attention
        )

        # ----------------------------------------------------
        # SECOND RESIDUAL CONNECTION + NORMALIZATION
        # ----------------------------------------------------

        output = self.norm2(
            x_attention + ffn_output
        )

        return (
            output,
            attention_weights
        )


# ============================================================
# CREATE ENCODER
# ============================================================

encoder = TransformerEncoderBlock(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM
)


# ============================================================
# FORWARD PASS
# ============================================================

encoder_output, attention_weights = encoder(
    transformer_input
)


# ============================================================
# REMOVE BATCH DIMENSION
# ============================================================

encoder_output = encoder_output.squeeze(0)


# ============================================================
# ATTENTION INFORMATION
# ============================================================

# attention_weights shape:
#
# batch × heads × query × key
#
# = 1 × 8 × 256 × 256

attention_weights = attention_weights.squeeze(0)


# Average over all heads

mean_attention = attention_weights.mean(
    dim=0
)


# ============================================================
# SELECT ONE HEAD
# ============================================================

selected_head = 0

head_attention = attention_weights[
    selected_head
]


# ============================================================
# VISUALIZE ORIGINAL IMAGE
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    image,
    cmap="gray"
)

plt.title(
    "Input Image"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "input_image.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZE MEAN MULTI-HEAD ATTENTION
# ============================================================

plt.figure(
    figsize=(9, 8)
)

plt.imshow(
    mean_attention.detach().numpy(),
    aspect="auto"
)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel("Key Patch")
plt.ylabel("Query Patch")

plt.title(
    "Mean Attention Across 8 Heads"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "mean_multihead_attention.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZE ONE ATTENTION HEAD
# ============================================================

plt.figure(
    figsize=(9, 8)
)

plt.imshow(
    head_attention.detach().numpy(),
    aspect="auto"
)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel("Key Patch")
plt.ylabel("Query Patch")

plt.title(
    f"Transformer Self-Attention — Head {selected_head}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "attention_head_0.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PATCH 128 ATTENTION MAP
# ============================================================

selected_patch = 128

patch_attention = (
    mean_attention[selected_patch]
)

patch_attention_map = patch_attention.view(
    num_rows,
    num_cols
)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    patch_attention_map.detach().numpy()
)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel("Patch Column")
plt.ylabel("Patch Row")

plt.title(
    "Mean Attention from Patch 128"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "patch_128_attention.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# INPUT VS OUTPUT
# ============================================================

input_norm = torch.norm(
    transformer_input.squeeze(0),
    dim=1
)

output_norm = torch.norm(
    encoder_output,
    dim=1
)


# ============================================================
# OUTPUT EMBEDDING VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.imshow(
    encoder_output.detach().numpy(),
    aspect="auto"
)

plt.colorbar(
    label="Embedding Value"
)

plt.xlabel("Embedding Dimension")
plt.ylabel("Patch Token")

plt.title(
    "Transformer Encoder Output"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "encoder_output.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("IMAGE INFORMATION")
print("=" * 60)

print(
    f"Original image shape: "
    f"{original_shape}"
)

print(
    f"Resized image shape: "
    f"{image.shape}"
)


print("\n" + "=" * 60)
print("TOKEN INFORMATION")
print("=" * 60)

print(
    f"Number of tokens: "
    f"{tokens.shape[0]}"
)

print(
    f"Token embedding dimension: "
    f"{tokens.shape[1]}"
)

print(
    f"Input with position: "
    f"{transformer_input.shape}"
)


print("\n" + "=" * 60)
print("TRANSFORMER CONFIGURATION")
print("=" * 60)

print(
    f"Embedding dimension: "
    f"{EMBED_DIM}"
)

print(
    f"Number of attention heads: "
    f"{NUM_HEADS}"
)

print(
    f"Feed-forward dimension: "
    f"{FFN_DIM}"
)


print("\n" + "=" * 60)
print("ATTENTION")
print("=" * 60)

print(
    f"Attention weights shape: "
    f"{attention_weights.shape}"
)

print(
    f"Mean attention shape: "
    f"{mean_attention.shape}"
)

print(
    f"Attention head shape: "
    f"{head_attention.shape}"
)


print("\n" + "=" * 60)
print("ENCODER OUTPUT")
print("=" * 60)

print(
    f"Encoder output shape: "
    f"{encoder_output.shape}"
)

print(
    f"Mean input token norm: "
    f"{input_norm.mean().item():.6f}"
)

print(
    f"Mean output token norm: "
    f"{output_norm.mean().item():.6f}"
)


print("\n" + "=" * 60)
print("ATTENTION ROW VERIFICATION")
print("=" * 60)

row_sums = mean_attention.sum(
    dim=1
)

print(
    f"Minimum row sum: "
    f"{row_sums.min().item():.9f}"
)

print(
    f"Maximum row sum: "
    f"{row_sums.max().item():.9f}"
)


# ============================================================
# TOP ATTENDED PATCHES
# ============================================================

top_values, top_indices = torch.topk(
    mean_attention[selected_patch],
    k=10
)

print("\n" + "=" * 60)
print(
    f"TOP 10 ATTENDED PATCHES "
    f"FOR PATCH {selected_patch}"
)
print("=" * 60)

for rank, (idx, weight) in enumerate(
    zip(
        top_indices.tolist(),
        top_values.tolist()
    ),
    start=1
):

    row = idx // num_cols
    col = idx % num_cols

    print(
        f"{rank:2d}. Patch {idx:3d} "
        f"(row={row:2d}, col={col:2d}) "
        f"weight={weight:.6f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "task_04_results.txt"
)

with open(
    results_file,
    "w"
) as f:

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        "TASK 4 — TRANSFORMER ENCODER RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "IMAGE INFORMATION\n"
    )

    f.write(
        f"Original image shape: "
        f"{original_shape}\n"
    )

    f.write(
        f"Resized image shape: "
        f"{image.shape}\n\n"
    )

    f.write(
        "TOKEN INFORMATION\n"
    )

    f.write(
        f"Number of tokens: "
        f"{tokens.shape[0]}\n"
    )

    f.write(
        f"Embedding dimension: "
        f"{tokens.shape[1]}\n"
    )

    f.write(
        f"Transformer input shape: "
        f"{transformer_input.shape}\n\n"
    )

    f.write(
        "TRANSFORMER CONFIGURATION\n"
    )

    f.write(
        f"Embedding dimension: "
        f"{EMBED_DIM}\n"
    )

    f.write(
        f"Number of heads: "
        f"{NUM_HEADS}\n"
    )

    f.write(
        f"Feed-forward dimension: "
        f"{FFN_DIM}\n\n"
    )

    f.write(
        "ATTENTION\n"
    )

    f.write(
        f"Attention shape: "
        f"{attention_weights.shape}\n"
    )

    f.write(
        f"Mean attention shape: "
        f"{mean_attention.shape}\n"
    )

    f.write(
        f"Head 0 shape: "
        f"{head_attention.shape}\n\n"
    )

    f.write(
        "ENCODER OUTPUT\n"
    )

    f.write(
        f"Encoder output shape: "
        f"{encoder_output.shape}\n"
    )

    f.write(
        f"Mean input norm: "
        f"{input_norm.mean().item():.6f}\n"
    )

    f.write(
        f"Mean output norm: "
        f"{output_norm.mean().item():.6f}\n\n"
    )

    f.write(
        "ATTENTION ROW VERIFICATION\n"
    )

    f.write(
        f"Minimum row sum: "
        f"{row_sums.min().item():.9f}\n"
    )

    f.write(
        f"Maximum row sum: "
        f"{row_sums.max().item():.9f}\n\n"
    )

    f.write(
        f"TOP 10 ATTENDED PATCHES "
        f"FOR PATCH {selected_patch}\n"
    )

    for rank, (idx, weight) in enumerate(
        zip(
            top_indices.tolist(),
            top_values.tolist()
        ),
        start=1
    ):

        row = idx // num_cols
        col = idx % num_cols

        f.write(
            f"{rank:2d}. Patch {idx:3d} "
            f"(row={row:2d}, col={col:2d}) "
            f"weight={weight:.6f}\n"
        )


print("\n" + "=" * 60)
print("TASK 4 COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)