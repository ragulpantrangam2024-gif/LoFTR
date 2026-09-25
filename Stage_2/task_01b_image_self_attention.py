import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


# ============================================================
# Configuration
# ============================================================

torch.manual_seed(42)

IMAGE_PATH = (
    "../datasets/hpatches-sequences-release/"
    "v_woman/1.ppm"
)

RESULTS_DIR = "results/task_01b"

IMAGE_SIZE = 256
PATCH_SIZE = 16
EMBED_DIM = 64

# Patch whose attention distribution we will visualize
SELECTED_PATCH = 128


# ============================================================
# Create results directory
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Load image
# ============================================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


print("=" * 60)
print("IMAGE INFORMATION")
print("=" * 60)

print(f"Original image shape: {image.shape}")


# ============================================================
# Resize image
# ============================================================

image_resized = cv2.resize(
    image,
    (IMAGE_SIZE, IMAGE_SIZE)
)

print(
    f"Resized image shape: "
    f"{image_resized.shape}"
)


# ============================================================
# Save original/resized image
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(
    image_resized,
    cmap="gray"
)

plt.title("HPatches v_woman/1.ppm")
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
# Convert image to tensor
# ============================================================

image_tensor = torch.tensor(
    image_resized,
    dtype=torch.float32
)

# Normalize pixels to [0, 1]

image_tensor = image_tensor / 255.0


# ============================================================
# Extract image patches
# ============================================================

patches = image_tensor.unfold(
    0,
    PATCH_SIZE,
    PATCH_SIZE
).unfold(
    1,
    PATCH_SIZE,
    PATCH_SIZE
)

# Shape before flattening:
# [num_rows, num_cols, patch_height, patch_width]

num_rows = patches.shape[0]
num_cols = patches.shape[1]

num_patches = num_rows * num_cols

print("\n" + "=" * 60)
print("PATCH INFORMATION")
print("=" * 60)

print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
print(f"Number of patch rows: {num_rows}")
print(f"Number of patch columns: {num_cols}")
print(f"Total patches/tokens: {num_patches}")


# ============================================================
# Reshape patches
# ============================================================

patches = patches.contiguous().view(
    num_patches,
    PATCH_SIZE * PATCH_SIZE
)

print(
    f"Patch tensor shape: "
    f"{patches.shape}"
)


# ============================================================
# Visualize patch grid
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(
    image_resized,
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
    "16 × 16 Image Patch Grid"
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
# Patch embedding
# ============================================================

patch_embedding = nn.Linear(
    PATCH_SIZE * PATCH_SIZE,
    EMBED_DIM,
    bias=False
)

X = patch_embedding(patches)

print(
    f"Patch embedding shape: "
    f"{X.shape}"
)


# ============================================================
# Query, Key, Value projections
# ============================================================

W_Q = nn.Linear(
    EMBED_DIM,
    EMBED_DIM,
    bias=False
)

W_K = nn.Linear(
    EMBED_DIM,
    EMBED_DIM,
    bias=False
)

W_V = nn.Linear(
    EMBED_DIM,
    EMBED_DIM,
    bias=False
)


Q = W_Q(X)
K = W_K(X)
V = W_V(X)


print("\n" + "=" * 60)
print("Q, K, V")
print("=" * 60)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")


# ============================================================
# Scaled dot-product attention
# ============================================================

d_k = K.shape[-1]

scores = torch.matmul(
    Q,
    K.transpose(0, 1)
)

scaled_scores = (
    scores /
    torch.sqrt(
        torch.tensor(
            float(d_k)
        )
    )
)


# ============================================================
# Softmax
# ============================================================

attention_weights = F.softmax(
    scaled_scores,
    dim=-1
)


print("\n" + "=" * 60)
print("ATTENTION")
print("=" * 60)

print(
    f"Attention matrix shape: "
    f"{attention_weights.shape}"
)


# ============================================================
# Verify attention rows
# ============================================================

row_sums = attention_weights.sum(
    dim=-1
)

print(
    "\nMinimum attention row sum:",
    row_sums.min().item()
)

print(
    "Maximum attention row sum:",
    row_sums.max().item()
)


# ============================================================
# Self-attention output
# ============================================================

output = torch.matmul(
    attention_weights,
    V
)

print(
    f"\nSelf-attention output shape: "
    f"{output.shape}"
)


# ============================================================
# Attention matrix visualization
# ============================================================

attention_numpy = (
    attention_weights
    .detach()
    .numpy()
)

plt.figure(figsize=(8, 7))

plt.imshow(
    attention_numpy,
    aspect="auto"
)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel("Key Patch")
plt.ylabel("Query Patch")

plt.title(
    "Self-Attention Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "attention_matrix.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# Selected patch attention map
# ============================================================

selected_attention = attention_numpy[
    SELECTED_PATCH
]

attention_map = selected_attention.reshape(
    num_rows,
    num_cols
)


plt.figure(figsize=(7, 6))

plt.imshow(
    attention_map,
    interpolation="nearest"
)

plt.colorbar(
    label="Attention Weight"
)

plt.title(
    f"Attention from Patch {SELECTED_PATCH}"
)

plt.xlabel("Patch Column")
plt.ylabel("Patch Row")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        f"attention_map_patch_{SELECTED_PATCH}.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# Print strongest attended patches
# ============================================================

top_k = 10

top_indices = np.argsort(
    selected_attention
)[-top_k:][::-1]

print("\n" + "=" * 60)
print(
    f"TOP {top_k} ATTENDED PATCHES "
    f"FOR PATCH {SELECTED_PATCH}"
)
print("=" * 60)

for rank, patch_index in enumerate(
    top_indices,
    start=1
):

    row = patch_index // num_cols
    col = patch_index % num_cols

    weight = selected_attention[
        patch_index
    ]

    print(
        f"{rank:2d}. "
        f"Patch {patch_index:3d} "
        f"(row={row:2d}, col={col:2d}) "
        f"weight={weight:.6f}"
    )


# ============================================================
# Save numerical summary
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "task_01b_results.txt"
)

with open(
    results_path,
    "w"
) as file:

    file.write(
        "TASK 1B - IMAGE SELF-ATTENTION\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Original image shape: "
        f"{image.shape}\n"
    )

    file.write(
        f"Resized image shape: "
        f"{image_resized.shape}\n"
    )

    file.write(
        f"Patch size: "
        f"{PATCH_SIZE} x {PATCH_SIZE}\n"
    )

    file.write(
        f"Number of patches: "
        f"{num_patches}\n"
    )

    file.write(
        f"Embedding dimension: "
        f"{EMBED_DIM}\n\n"
    )

    file.write(
        f"Q shape: {Q.shape}\n"
    )

    file.write(
        f"K shape: {K.shape}\n"
    )

    file.write(
        f"V shape: {V.shape}\n"
    )

    file.write(
        f"Attention matrix shape: "
        f"{attention_weights.shape}\n"
    )

    file.write(
        f"Output shape: "
        f"{output.shape}\n\n"
    )

    file.write(
        f"Selected patch: "
        f"{SELECTED_PATCH}\n\n"
    )

    file.write(
        "Top attended patches:\n"
    )

    for rank, patch_index in enumerate(
        top_indices,
        start=1
    ):

        row = patch_index // num_cols
        col = patch_index % num_cols

        weight = selected_attention[
            patch_index
        ]

        file.write(
            f"{rank:2d}. "
            f"Patch {patch_index:3d} "
            f"(row={row:2d}, col={col:2d}) "
            f"weight={weight:.6f}\n"
        )


print("\n" + "=" * 60)
print("TASK 1B COMPLETED")
print("=" * 60)

print("\nResults saved to:")
print(RESULTS_DIR)