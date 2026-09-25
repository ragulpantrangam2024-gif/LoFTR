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

IMAGE_1_PATH = (
    "../datasets/hpatches-sequences-release/v_woman/1.ppm"
)

IMAGE_2_PATH = (
    "../datasets/hpatches-sequences-release/v_woman/2.ppm"
)

RESULTS_DIR = "results/task_02"

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

image1 = cv2.imread(IMAGE_1_PATH, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(IMAGE_2_PATH, cv2.IMREAD_GRAYSCALE)

if image1 is None:
    raise FileNotFoundError(
        f"Could not load Image 1: {IMAGE_1_PATH}"
    )

if image2 is None:
    raise FileNotFoundError(
        f"Could not load Image 2: {IMAGE_2_PATH}"
    )


original_shape_1 = image1.shape
original_shape_2 = image2.shape

image1 = cv2.resize(image1, (IMAGE_SIZE, IMAGE_SIZE))
image2 = cv2.resize(image2, (IMAGE_SIZE, IMAGE_SIZE))


# ============================================================
# IMAGE TO PATCH TOKENS
# ============================================================

def image_to_patches(image, patch_size):
    """
    Convert a grayscale image into flattened patches.
    """

    tensor = torch.from_numpy(image).float()

    patches = (
        tensor
        .unfold(0, patch_size, patch_size)
        .unfold(1, patch_size, patch_size)
    )

    num_rows = patches.shape[0]
    num_cols = patches.shape[1]

    patches = patches.contiguous().view(
        num_rows * num_cols,
        patch_size * patch_size
    )

    return patches


patches1 = image_to_patches(image1, PATCH_SIZE)
patches2 = image_to_patches(image2, PATCH_SIZE)


# Normalize pixel values

patches1 = patches1 / 255.0
patches2 = patches2 / 255.0


# ============================================================
# SHARED PATCH EMBEDDING
# ============================================================

input_dim = PATCH_SIZE * PATCH_SIZE

patch_embedding = nn.Linear(
    input_dim,
    EMBED_DIM
)


tokens1 = patch_embedding(patches1)
tokens2 = patch_embedding(patches2)


# ============================================================
# QUERY / KEY / VALUE PROJECTIONS
# ============================================================

W_Q = nn.Linear(EMBED_DIM, EMBED_DIM)
W_K = nn.Linear(EMBED_DIM, EMBED_DIM)
W_V = nn.Linear(EMBED_DIM, EMBED_DIM)


# ============================================================
# CROSS-ATTENTION FUNCTION
# ============================================================

def cross_attention(query_tokens, key_value_tokens):
    """
    Cross-attention:

        Q comes from one image.
        K and V come from another image.
    """

    Q = W_Q(query_tokens)
    K = W_K(key_value_tokens)
    V = W_V(key_value_tokens)

    d_k = Q.shape[-1]

    scores = torch.matmul(Q, K.T) / np.sqrt(d_k)

    attention_weights = F.softmax(
        scores,
        dim=-1
    )

    output = torch.matmul(
        attention_weights,
        V
    )

    return Q, K, V, scores, attention_weights, output


# ============================================================
# IMAGE 1 -> IMAGE 2
# ============================================================

(
    Q_12,
    K_12,
    V_12,
    scores_12,
    attention_12,
    output_12
) = cross_attention(
    tokens1,
    tokens2
)


# ============================================================
# IMAGE 2 -> IMAGE 1
# ============================================================

(
    Q_21,
    K_21,
    V_21,
    scores_21,
    attention_21,
    output_21
) = cross_attention(
    tokens2,
    tokens1
)


# ============================================================
# SAVE IMAGE PAIR
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image1, cmap="gray")
plt.title("Image 1 — v_woman/1.ppm")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image2, cmap="gray")
plt.title("Image 2 — v_woman/2.ppm")
plt.axis("off")

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
# CROSS-ATTENTION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    attention_12.detach().numpy(),
    aspect="auto"
)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel("Key Patch — Image 2")
plt.ylabel("Query Patch — Image 1")

plt.title(
    "Cross-Attention: Image 1 → Image 2"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "cross_attention_1_to_2.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# REVERSE CROSS-ATTENTION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    attention_21.detach().numpy(),
    aspect="auto"
)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel("Key Patch — Image 1")
plt.ylabel("Query Patch — Image 2")

plt.title(
    "Cross-Attention: Image 2 → Image 1"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "cross_attention_2_to_1.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# VISUALIZE ATTENTION FROM ONE PATCH
# ============================================================

selected_patch = 128

attention_map = (
    attention_12[selected_patch]
    .detach()
    .numpy()
)

attention_map = attention_map.reshape(
    IMAGE_SIZE // PATCH_SIZE,
    IMAGE_SIZE // PATCH_SIZE
)


plt.figure(figsize=(7, 6))

plt.imshow(
    attention_map,
    interpolation="nearest"
)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel("Patch Column — Image 2")
plt.ylabel("Patch Row — Image 2")

plt.title(
    f"Image 1 Patch {selected_patch} "
    "Attention on Image 2"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "patch_128_cross_attention.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# TOP ATTENDED PATCHES
# ============================================================

top_values, top_indices = torch.topk(
    attention_12[selected_patch],
    k=10
)


# ============================================================
# NUMERICAL VERIFICATION
# ============================================================

row_sums_12 = attention_12.sum(dim=1)

min_row_sum = row_sums_12.min().item()
max_row_sum = row_sums_12.max().item()


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("IMAGE INFORMATION")
print("=" * 60)

print(
    f"Original Image 1 shape: {original_shape_1}"
)

print(
    f"Original Image 2 shape: {original_shape_2}"
)

print(
    f"Resized Image 1 shape: {image1.shape}"
)

print(
    f"Resized Image 2 shape: {image2.shape}"
)


print("\n" + "=" * 60)
print("PATCH INFORMATION")
print("=" * 60)

print(
    f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}"
)

print(
    f"Total tokens Image 1: {patches1.shape[0]}"
)

print(
    f"Total tokens Image 2: {patches2.shape[0]}"
)

print(
    f"Patch dimension: {patches1.shape[1]}"
)


print("\n" + "=" * 60)
print("TOKEN EMBEDDINGS")
print("=" * 60)

print(
    f"Image 1 tokens: {tokens1.shape}"
)

print(
    f"Image 2 tokens: {tokens2.shape}"
)


print("\n" + "=" * 60)
print("CROSS-ATTENTION: IMAGE 1 -> IMAGE 2")
print("=" * 60)

print(
    f"Q shape: {Q_12.shape}"
)

print(
    f"K shape: {K_12.shape}"
)

print(
    f"V shape: {V_12.shape}"
)

print(
    f"Attention matrix shape: {attention_12.shape}"
)

print(
    f"Output shape: {output_12.shape}"
)


print("\n" + "=" * 60)
print("ATTENTION ROW VERIFICATION")
print("=" * 60)

print(
    f"Minimum row sum: {min_row_sum}"
)

print(
    f"Maximum row sum: {max_row_sum}"
)


print("\n" + "=" * 60)
print(
    f"TOP 10 ATTENDED PATCHES FOR IMAGE 1 PATCH {selected_patch}"
)
print("=" * 60)

for rank, (idx, weight) in enumerate(
    zip(
        top_indices.tolist(),
        top_values.tolist()
    ),
    start=1
):

    row = idx // (IMAGE_SIZE // PATCH_SIZE)
    col = idx % (IMAGE_SIZE // PATCH_SIZE)

    print(
        f"{rank:2d}. Patch {idx:3d} "
        f"(row={row:2d}, col={col:2d}) "
        f"weight={weight:.6f}"
    )


print("\n" + "=" * 60)
print("CROSS-ATTENTION: IMAGE 2 -> IMAGE 1")
print("=" * 60)

print(
    f"Attention matrix shape: {attention_21.shape}"
)

print(
    f"Output shape: {output_21.shape}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "task_02_results.txt"
)

with open(results_file, "w") as f:

    f.write("=" * 60 + "\n")
    f.write("TASK 2 — CROSS-ATTENTION RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write("IMAGE INFORMATION\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Original Image 1 shape: {original_shape_1}\n"
    )

    f.write(
        f"Original Image 2 shape: {original_shape_2}\n"
    )

    f.write(
        f"Resized Image 1 shape: {image1.shape}\n"
    )

    f.write(
        f"Resized Image 2 shape: {image2.shape}\n\n"
    )

    f.write("PATCH INFORMATION\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}\n"
    )

    f.write(
        f"Image 1 tokens: {patches1.shape[0]}\n"
    )

    f.write(
        f"Image 2 tokens: {patches2.shape[0]}\n"
    )

    f.write(
        f"Patch dimension: {patches1.shape[1]}\n\n"
    )

    f.write("TOKEN EMBEDDINGS\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Image 1 tokens: {tokens1.shape}\n"
    )

    f.write(
        f"Image 2 tokens: {tokens2.shape}\n\n"
    )

    f.write("CROSS-ATTENTION IMAGE 1 -> IMAGE 2\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Q shape: {Q_12.shape}\n"
    )

    f.write(
        f"K shape: {K_12.shape}\n"
    )

    f.write(
        f"V shape: {V_12.shape}\n"
    )

    f.write(
        f"Attention shape: {attention_12.shape}\n"
    )

    f.write(
        f"Output shape: {output_12.shape}\n\n"
    )

    f.write("ATTENTION ROW VERIFICATION\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Minimum row sum: {min_row_sum}\n"
    )

    f.write(
        f"Maximum row sum: {max_row_sum}\n\n"
    )

    f.write(
        f"TOP 10 ATTENDED PATCHES FOR "
        f"IMAGE 1 PATCH {selected_patch}\n"
    )

    f.write("-" * 60 + "\n")

    for rank, (idx, weight) in enumerate(
        zip(
            top_indices.tolist(),
            top_values.tolist()
        ),
        start=1
    ):

        row = idx // (IMAGE_SIZE // PATCH_SIZE)
        col = idx % (IMAGE_SIZE // PATCH_SIZE)

        f.write(
            f"{rank:2d}. Patch {idx:3d} "
            f"(row={row:2d}, col={col:2d}) "
            f"weight={weight:.6f}\n"
        )

    f.write("\n")

    f.write("CROSS-ATTENTION IMAGE 2 -> IMAGE 1\n")
    f.write("-" * 60 + "\n")

    f.write(
        f"Attention shape: {attention_21.shape}\n"
    )

    f.write(
        f"Output shape: {output_21.shape}\n"
    )


print("\n" + "=" * 60)
print("TASK 2 COMPLETED")
print("=" * 60)

print(
    f"\nResults saved to:\n{RESULTS_DIR}"
)