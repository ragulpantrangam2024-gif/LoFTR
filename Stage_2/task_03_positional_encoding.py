import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = (
    "../datasets/hpatches-sequences-release/v_woman/1.ppm"
)

RESULTS_DIR = "results/task_03"

IMAGE_SIZE = 256
PATCH_SIZE = 16
EMBED_DIM = 64

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
    Convert a grayscale image into flattened patches.
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
    """
    Generate standard sinusoidal positional encoding
    for one spatial dimension.
    """

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
    Generate 2D sinusoidal positional encoding.

    Half of the dimensions encode row position.
    Half encode column position.
    """

    if embed_dim % 4 != 0:
        raise ValueError(
            "embed_dim must be divisible by 4 "
            "for this 2D encoding."
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


# ============================================================
# CREATE POSITIONAL ENCODING
# ============================================================

num_rows = IMAGE_SIZE // PATCH_SIZE
num_cols = IMAGE_SIZE // PATCH_SIZE

positional_encoding = get_2d_sinusoidal_encoding(
    num_rows,
    num_cols,
    EMBED_DIM
)


# Flatten spatial grid into token sequence

positional_encoding_tokens = (
    positional_encoding
    .view(
        num_rows * num_cols,
        EMBED_DIM
    )
)


# ============================================================
# ADD POSITION TO IMAGE TOKENS
# ============================================================

tokens_with_position = (
    tokens
    +
    positional_encoding_tokens
)


# ============================================================
# POSITIONAL ENCODING MAGNITUDE
# ============================================================

position_magnitude = torch.norm(
    positional_encoding,
    dim=2
)


# ============================================================
# SAVE ORIGINAL IMAGE
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    image,
    cmap="gray"
)

plt.title(
    "Original Image"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "original_image.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PATCH GRID
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    image,
    cmap="gray"
)

for x in range(
    0,
    IMAGE_SIZE + 1,
    PATCH_SIZE
):

    plt.axvline(
        x,
        linewidth=0.5
    )

for y in range(
    0,
    IMAGE_SIZE + 1,
    PATCH_SIZE
):

    plt.axhline(
        y,
        linewidth=0.5
    )

plt.title(
    "16 × 16 Patch Grid"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "patch_grid.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# POSITIONAL ENCODING CHANNEL
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    positional_encoding[:, :, 0].numpy(),
    cmap="viridis"
)

plt.colorbar(
    label="Encoding Value"
)

plt.title(
    "2D Positional Encoding — Channel 0"
)

plt.xlabel("Patch Column")
plt.ylabel("Patch Row")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "positional_encoding_channel_0.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# POSITIONAL ENCODING MAGNITUDE
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    position_magnitude.numpy(),
    cmap="viridis"
)

plt.colorbar(
    label="Magnitude"
)

plt.title(
    "Positional Encoding Magnitude"
)

plt.xlabel("Patch Column")
plt.ylabel("Patch Row")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "positional_encoding_magnitude.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# TOKEN EMBEDDING BEFORE POSITION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.imshow(
    tokens.detach().numpy(),
    aspect="auto"
)

plt.colorbar(
    label="Embedding Value"
)

plt.xlabel("Embedding Dimension")
plt.ylabel("Patch Token")

plt.title(
    "Patch Embeddings Before Positional Encoding"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "tokens_before_position.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# TOKEN EMBEDDING AFTER POSITION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.imshow(
    tokens_with_position.detach().numpy(),
    aspect="auto"
)

plt.colorbar(
    label="Embedding Value"
)

plt.xlabel("Embedding Dimension")
plt.ylabel("Patch Token")

plt.title(
    "Patch Embeddings After Positional Encoding"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "tokens_after_position.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# DIFFERENCE
# ============================================================

difference = (
    tokens_with_position
    -
    tokens
)


plt.figure(
    figsize=(10, 6)
)

plt.imshow(
    difference.detach().numpy(),
    aspect="auto"
)

plt.colorbar(
    label="Positional Contribution"
)

plt.xlabel("Embedding Dimension")
plt.ylabel("Patch Token")

plt.title(
    "Contribution of Positional Encoding"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "positional_encoding_contribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# NUMERICAL INFORMATION
# ============================================================

position_norms = torch.norm(
    positional_encoding_tokens,
    dim=1
)

token_norms_before = torch.norm(
    tokens,
    dim=1
)

token_norms_after = torch.norm(
    tokens_with_position,
    dim=1
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("IMAGE INFORMATION")
print("=" * 60)

print(
    f"Original image shape: {original_shape}"
)

print(
    f"Resized image shape: {image.shape}"
)


print("\n" + "=" * 60)
print("PATCH INFORMATION")
print("=" * 60)

print(
    f"Patch size: "
    f"{PATCH_SIZE} x {PATCH_SIZE}"
)

print(
    f"Number of patch rows: {num_rows}"
)

print(
    f"Number of patch columns: {num_cols}"
)

print(
    f"Total patches/tokens: "
    f"{num_rows * num_cols}"
)

print(
    f"Patch tensor shape: "
    f"{patches.shape}"
)


print("\n" + "=" * 60)
print("TOKEN EMBEDDING")
print("=" * 60)

print(
    f"Token embedding shape: "
    f"{tokens.shape}"
)


print("\n" + "=" * 60)
print("POSITIONAL ENCODING")
print("=" * 60)

print(
    f"Positional encoding grid shape: "
    f"{positional_encoding.shape}"
)

print(
    f"Positional encoding token shape: "
    f"{positional_encoding_tokens.shape}"
)

print(
    f"Positional encoding dimension: "
    f"{EMBED_DIM}"
)


print("\n" + "=" * 60)
print("TOKEN REPRESENTATION")
print("=" * 60)

print(
    f"Before positional encoding: "
    f"{tokens.shape}"
)

print(
    f"After positional encoding: "
    f"{tokens_with_position.shape}"
)


print("\n" + "=" * 60)
print("POSITIONAL ENCODING NORMS")
print("=" * 60)

print(
    f"Minimum position norm: "
    f"{position_norms.min().item():.6f}"
)

print(
    f"Maximum position norm: "
    f"{position_norms.max().item():.6f}"
)

print(
    f"Mean position norm: "
    f"{position_norms.mean().item():.6f}"
)


print("\n" + "=" * 60)
print("TOKEN NORMS")
print("=" * 60)

print(
    f"Mean token norm before: "
    f"{token_norms_before.mean().item():.6f}"
)

print(
    f"Mean token norm after: "
    f"{token_norms_after.mean().item():.6f}"
)


print("\n" + "=" * 60)
print("EXAMPLE POSITIONAL ENCODINGS")
print("=" * 60)

positions_to_show = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (8, 8),
    (15, 15)
]

for row, col in positions_to_show:

    patch_index = (
        row * num_cols + col
    )

    print(
        f"\nPatch {patch_index} "
        f"(row={row}, col={col})"
    )

    print(
        positional_encoding_tokens[
            patch_index
        ][:8]
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "task_03_results.txt"
)

with open(
    results_file,
    "w"
) as f:

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        "TASK 3 — POSITIONAL ENCODING RESULTS\n"
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
        "PATCH INFORMATION\n"
    )

    f.write(
        f"Patch size: "
        f"{PATCH_SIZE} x {PATCH_SIZE}\n"
    )

    f.write(
        f"Patch rows: {num_rows}\n"
    )

    f.write(
        f"Patch columns: {num_cols}\n"
    )

    f.write(
        f"Total tokens: "
        f"{num_rows * num_cols}\n"
    )

    f.write(
        f"Patch tensor shape: "
        f"{patches.shape}\n\n"
    )

    f.write(
        "TOKEN EMBEDDING\n"
    )

    f.write(
        f"Token embedding shape: "
        f"{tokens.shape}\n\n"
    )

    f.write(
        "POSITIONAL ENCODING\n"
    )

    f.write(
        f"Positional encoding shape: "
        f"{positional_encoding.shape}\n"
    )

    f.write(
        f"Flattened positional encoding: "
        f"{positional_encoding_tokens.shape}\n"
    )

    f.write(
        f"Embedding dimension: "
        f"{EMBED_DIM}\n\n"
    )

    f.write(
        "TOKEN REPRESENTATION\n"
    )

    f.write(
        f"Before position: "
        f"{tokens.shape}\n"
    )

    f.write(
        f"After position: "
        f"{tokens_with_position.shape}\n\n"
    )

    f.write(
        "POSITIONAL ENCODING NORMS\n"
    )

    f.write(
        f"Minimum: "
        f"{position_norms.min().item():.6f}\n"
    )

    f.write(
        f"Maximum: "
        f"{position_norms.max().item():.6f}\n"
    )

    f.write(
        f"Mean: "
        f"{position_norms.mean().item():.6f}\n\n"
    )

    f.write(
        "TOKEN NORMS\n"
    )

    f.write(
        f"Mean before: "
        f"{token_norms_before.mean().item():.6f}\n"
    )

    f.write(
        f"Mean after: "
        f"{token_norms_after.mean().item():.6f}\n\n"
    )

    f.write(
        "EXAMPLE POSITIONAL ENCODINGS\n"
    )

    for row, col in positions_to_show:

        patch_index = (
            row * num_cols + col
        )

        f.write(
            f"\nPatch {patch_index} "
            f"(row={row}, col={col})\n"
        )

        f.write(
            str(
                positional_encoding_tokens[
                    patch_index
                ][:8]
            )
        )

        f.write("\n")


print("\n" + "=" * 60)
print("TASK 3 COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)