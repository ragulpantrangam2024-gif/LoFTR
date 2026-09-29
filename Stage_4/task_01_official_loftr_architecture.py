"""
Stage 4 - Task 1
Official LoFTR Architecture Study

Purpose:
    Study and demonstrate the major architectural stages
    of the official LoFTR pipeline.

This task does NOT implement the official LoFTR model.

It demonstrates the architecture conceptually using:
    1. Dense CNN feature extraction
    2. Coarse feature representation
    3. Positional encoding
    4. Self-attention
    5. Cross-attention
    6. Coarse matching concept
    7. Fine-level refinement concept

The official pretrained LoFTR model will be used in Stage 4 Task 2.
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

IMAGE1_PATH = os.path.join(
    DATASET_DIR,
    "1.ppm"
)

IMAGE2_PATH = os.path.join(
    DATASET_DIR,
    "2.ppm"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "results",
    "task_01"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

IMAGE_SIZE = 256

EMBED_DIM = 64

NUM_HEADS = 8

COARSE_GRID = 16


# ============================================================
# Image Loading
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


# ============================================================
# Simplified Dense CNN
# ============================================================

class DenseCNN(nn.Module):

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
# 2D Positional Encoding
# ============================================================

def create_2d_positional_encoding(
    height,
    width,
    embed_dim
):

    if embed_dim % 4 != 0:

        raise ValueError(
            "Embedding dimension must be divisible by 4."
        )

    encoding = torch.zeros(
        height,
        width,
        embed_dim
    )

    quarter_dim = embed_dim // 4

    y = torch.arange(
        height,
        dtype=torch.float32
    )

    x = torch.arange(
        width,
        dtype=torch.float32
    )

    div_term = torch.exp(
        torch.arange(
            0,
            quarter_dim,
            dtype=torch.float32
        )
        *
        (
            -np.log(10000.0)
            /
            quarter_dim
        )
    )

    for i in range(quarter_dim):

        encoding[:, :, 2 * i] = torch.sin(
            y[:, None] * div_term[i]
        )

        encoding[:, :, 2 * i + 1] = torch.cos(
            y[:, None] * div_term[i]
        )

        offset = embed_dim // 2

        encoding[
            :,
            :,
            offset + 2 * i
        ] = torch.sin(
            x[None, :] * div_term[i]
        )

        encoding[
            :,
            :,
            offset + 2 * i + 1
        ] = torch.cos(
            x[None, :] * div_term[i]
        )

    return encoding


# ============================================================
# Transformer Attention Module
# ============================================================

class AttentionModule(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads
    ):

        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm_self = nn.LayerNorm(
            embed_dim
        )

        self.norm_cross = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        tokens1,
        tokens2
    ):

        # ----------------------------------------------------
        # Self-attention
        # ----------------------------------------------------

        self1, self_weights1 = (
            self.self_attention(
                tokens1,
                tokens1,
                tokens1,
                need_weights=True,
                average_attn_weights=False
            )
        )

        self2, self_weights2 = (
            self.self_attention(
                tokens2,
                tokens2,
                tokens2,
                need_weights=True,
                average_attn_weights=False
            )
        )

        tokens1 = self.norm_self(
            tokens1 + self1
        )

        tokens2 = self.norm_self(
            tokens2 + self2
        )

        # ----------------------------------------------------
        # Cross-attention
        # ----------------------------------------------------

        cross1, cross_weights1 = (
            self.cross_attention(
                tokens1,
                tokens2,
                tokens2,
                need_weights=True,
                average_attn_weights=False
            )
        )

        cross2, cross_weights2 = (
            self.cross_attention(
                tokens2,
                tokens1,
                tokens1,
                need_weights=True,
                average_attn_weights=False
            )
        )

        tokens1 = self.norm_cross(
            tokens1 + cross1
        )

        tokens2 = self.norm_cross(
            tokens2 + cross2
        )

        return (
            tokens1,
            tokens2,
            self_weights1,
            self_weights2,
            cross_weights1,
            cross_weights2
        )


# ============================================================
# Coarse Matching
# ============================================================

def calculate_similarity(
    features1,
    features2
):

    features1 = F.normalize(
        features1,
        p=2,
        dim=-1
    )

    features2 = F.normalize(
        features2,
        p=2,
        dim=-1
    )

    similarity = torch.matmul(
        features1,
        features2.transpose(1, 2)
    )

    return similarity


def mutual_nearest_matches(
    similarity
):

    best12 = torch.argmax(
        similarity,
        dim=2
    )

    best21 = torch.argmax(
        similarity,
        dim=1
    )

    matches = []

    for i in range(
        similarity.shape[1]
    ):

        j = best12[0, i].item()

        if best21[0, j].item() == i:

            score = similarity[
                0,
                i,
                j
            ].item()

            matches.append(
                (
                    i,
                    j,
                    score
                )
            )

    return matches


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Stage 4 - Task 1")
    print("Official LoFTR Architecture Study")
    print("=" * 60)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This task demonstrates the major LoFTR "
        "architectural components."
    )

    print(
        "It is NOT the official pretrained LoFTR model."
    )

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

    print(
        "Resized image shape:",
        resized1.shape
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn = DenseCNN()

    cnn.eval()

    with torch.no_grad():

        feature_map1 = cnn(
            tensor1
        )

        feature_map2 = cnn(
            tensor2
        )

    print()
    print("=== Dense Feature Extraction ===")

    print(
        "Feature map image 1:",
        tuple(feature_map1.shape)
    )

    print(
        "Feature map image 2:",
        tuple(feature_map2.shape)
    )

    # --------------------------------------------------------
    # Coarse feature representation
    # --------------------------------------------------------

    coarse1 = F.adaptive_avg_pool2d(
        feature_map1,
        (
            COARSE_GRID,
            COARSE_GRID
        )
    )

    coarse2 = F.adaptive_avg_pool2d(
        feature_map2,
        (
            COARSE_GRID,
            COARSE_GRID
        )
    )

    tokens1 = coarse1.permute(
        0,
        2,
        3,
        1
    )

    tokens2 = coarse2.permute(
        0,
        2,
        3,
        1
    )

    tokens1 = tokens1.reshape(
        1,
        COARSE_GRID * COARSE_GRID,
        EMBED_DIM
    )

    tokens2 = tokens2.reshape(
        1,
        COARSE_GRID * COARSE_GRID,
        EMBED_DIM
    )

    print()
    print("=== Coarse Representation ===")

    print(
        "Coarse feature map image 1:",
        tuple(coarse1.shape)
    )

    print(
        "Coarse feature map image 2:",
        tuple(coarse2.shape)
    )

    print(
        "Tokens image 1:",
        tuple(tokens1.shape)
    )

    print(
        "Tokens image 2:",
        tuple(tokens2.shape)
    )

    # --------------------------------------------------------
    # Positional encoding
    # --------------------------------------------------------

    positional_encoding = (
        create_2d_positional_encoding(
            COARSE_GRID,
            COARSE_GRID,
            EMBED_DIM
        )
    )

    positional_encoding = (
        positional_encoding
        .reshape(
            1,
            COARSE_GRID * COARSE_GRID,
            EMBED_DIM
        )
    )

    tokens1 = (
        tokens1
        +
        positional_encoding
    )

    tokens2 = (
        tokens2
        +
        positional_encoding
    )

    print()
    print("=== Positional Encoding ===")

    print(
        "Positional encoding:",
        tuple(positional_encoding.shape)
    )

    print(
        "Position-aware tokens image 1:",
        tuple(tokens1.shape)
    )

    print(
        "Position-aware tokens image 2:",
        tuple(tokens2.shape)
    )

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    transformer = AttentionModule(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS
    )

    transformer.eval()

    with torch.no_grad():

        (
            contextual1,
            contextual2,
            self_weights1,
            self_weights2,
            cross_weights1,
            cross_weights2
        ) = transformer(
            tokens1,
            tokens2
        )

    print()
    print("=== Local Feature Transformer ===")

    print(
        "Number of attention heads:",
        NUM_HEADS
    )

    print(
        "Contextualized features image 1:",
        tuple(contextual1.shape)
    )

    print(
        "Contextualized features image 2:",
        tuple(contextual2.shape)
    )

    print(
        "Self-attention image 1:",
        tuple(self_weights1.shape)
    )

    print(
        "Self-attention image 2:",
        tuple(self_weights2.shape)
    )

    print(
        "Cross-attention image 1 -> image 2:",
        tuple(cross_weights1.shape)
    )

    print(
        "Cross-attention image 2 -> image 1:",
        tuple(cross_weights2.shape)
    )

    # --------------------------------------------------------
    # Attention normalization
    # --------------------------------------------------------

    self_row_sum = (
        self_weights1.sum(dim=-1)
    )

    cross_row_sum = (
        cross_weights1.sum(dim=-1)
    )

    print()
    print("=== Attention Validation ===")

    print(
        "Self-attention row sum min:",
        self_row_sum.min().item()
    )

    print(
        "Self-attention row sum max:",
        self_row_sum.max().item()
    )

    print(
        "Cross-attention row sum min:",
        cross_row_sum.min().item()
    )

    print(
        "Cross-attention row sum max:",
        cross_row_sum.max().item()
    )

    # --------------------------------------------------------
    # Coarse matching
    # --------------------------------------------------------

    similarity = calculate_similarity(
        contextual1,
        contextual2
    )

    print()
    print("=== Coarse Matching ===")

    print(
        "Similarity matrix:",
        tuple(similarity.shape)
    )

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

    matches = mutual_nearest_matches(
        similarity
    )

    print(
        "Mutual nearest-neighbor matches:",
        len(matches)
    )

    # --------------------------------------------------------
    # Fine-level concept
    # --------------------------------------------------------

    print()
    print("=== Fine-Level Stage ===")

    print(
        "Fine-level refinement is represented "
        "conceptually in Task 1."
    )

    print(
        "The official pretrained fine-level "
        "implementation will be executed in Task 2+."
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        RESULTS_DIR,
        "task_01_results.txt"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        file.write(
            "Stage 4 - Task 1\n"
        )

        file.write(
            "Official LoFTR Architecture Study\n"
        )

        file.write(
            "========================================\n\n"
        )

        file.write(
            "IMPORTANT:\n"
        )

        file.write(
            "This is an educational architecture "
            "demonstration.\n"
        )

        file.write(
            "It is NOT the official pretrained LoFTR model.\n\n"
        )

        file.write(
            f"Image 1 original shape: {shape1}\n"
        )

        file.write(
            f"Image 2 original shape: {shape2}\n"
        )

        file.write(
            f"Resized image shape: {resized1.shape}\n\n"
        )

        file.write(
            f"Feature map image 1: "
            f"{tuple(feature_map1.shape)}\n"
        )

        file.write(
            f"Feature map image 2: "
            f"{tuple(feature_map2.shape)}\n\n"
        )

        file.write(
            f"Coarse grid: "
            f"{COARSE_GRID} x {COARSE_GRID}\n"
        )

        file.write(
            f"Coarse tokens: "
            f"{COARSE_GRID * COARSE_GRID}\n"
        )

        file.write(
            f"Embedding dimension: "
            f"{EMBED_DIM}\n"
        )

        file.write(
            f"Attention heads: "
            f"{NUM_HEADS}\n\n"
        )

        file.write(
            f"Self-attention image 1: "
            f"{tuple(self_weights1.shape)}\n"
        )

        file.write(
            f"Self-attention image 2: "
            f"{tuple(self_weights2.shape)}\n"
        )

        file.write(
            f"Cross-attention image 1 -> image 2: "
            f"{tuple(cross_weights1.shape)}\n"
        )

        file.write(
            f"Cross-attention image 2 -> image 1: "
            f"{tuple(cross_weights2.shape)}\n\n"
        )

        file.write(
            f"Similarity matrix: "
            f"{tuple(similarity.shape)}\n"
        )

        file.write(
            f"Similarity minimum: "
            f"{similarity.min().item()}\n"
        )

        file.write(
            f"Similarity maximum: "
            f"{similarity.max().item()}\n"
        )

        file.write(
            f"Similarity mean: "
            f"{similarity.mean().item()}\n"
        )

        file.write(
            f"Mutual nearest-neighbor matches: "
            f"{len(matches)}\n"
        )

        file.write(
            "\nArchitecture stages demonstrated:\n"
        )

        file.write(
            "1. Dense CNN feature extraction\n"
        )

        file.write(
            "2. Coarse feature representation\n"
        )

        file.write(
            "3. 2D positional encoding\n"
        )

        file.write(
            "4. Self-attention\n"
        )

        file.write(
            "5. Cross-attention\n"
        )

        file.write(
            "6. Coarse similarity matching\n"
        )

        file.write(
            "7. Fine-level refinement concept\n"
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Task 1 completed.")
    print("=" * 60)

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()