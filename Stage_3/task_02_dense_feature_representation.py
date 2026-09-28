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

IMAGE1_PATH = os.path.join(DATASET_DIR, "1.ppm")
IMAGE2_PATH = os.path.join(DATASET_DIR, "2.ppm")

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_02"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 256
EMBED_DIM = 64

torch.manual_seed(42)


# ============================================================
# CNN Feature Extractor
# Same architecture as Task 1
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Load image
# ============================================================

def load_image(path):

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image:\n{path}"
        )

    original_shape = image.shape

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return image, tensor, original_shape


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
        * -(np.log(10000.0) / quarter_dim)
    )

    # Y direction
    pe[:, :, 0:quarter_dim] = torch.sin(
        position_y * div_term
    ).unsqueeze(1)

    pe[:, :, quarter_dim:2 * quarter_dim] = torch.cos(
        position_y * div_term
    ).unsqueeze(1)

    # X direction
    pe[:, :, 2 * quarter_dim:3 * quarter_dim] = torch.sin(
        position_x * div_term
    ).unsqueeze(0)

    pe[:, :, 3 * quarter_dim:4 * quarter_dim] = torch.cos(
        position_x * div_term
    ).unsqueeze(0)

    return pe


# ============================================================
# Convert CNN feature map into dense tokens
# ============================================================

def feature_map_to_tokens(feature_map):

    # Input:
    # [1, 64, 128, 128]

    batch, channels, height, width = feature_map.shape

    # Move channels to the last dimension
    # [1, 128, 128, 64]

    features = feature_map.permute(
        0, 2, 3, 1
    )

    # Flatten spatial dimensions
    # [1, 16384, 64]

    tokens = features.reshape(
        batch,
        height * width,
        channels
    )

    return tokens


# ============================================================
# Process one image
# ============================================================

def process_image(
    image,
    image_tensor,
    model,
    image_name
):

    with torch.no_grad():

        feature_map = model(image_tensor)

    print(
        f"\n{image_name}"
    )

    print(
        "CNN feature map:",
        tuple(feature_map.shape)
    )

    tokens = feature_map_to_tokens(
        feature_map
    )

    print(
        "Dense tokens:",
        tuple(tokens.shape)
    )

    batch, num_tokens, channels = tokens.shape

    height = feature_map.shape[2]
    width = feature_map.shape[3]

    positional_encoding = create_2d_positional_encoding(
        height,
        width,
        channels
    )

    print(
        "Positional encoding:",
        tuple(positional_encoding.shape)
    )

    positional_encoding_flat = positional_encoding.reshape(
        height * width,
        channels
    )

    positional_encoding_flat = positional_encoding_flat.unsqueeze(0)

    print(
        "Flattened positional encoding:",
        tuple(positional_encoding_flat.shape)
    )

    tokens_with_position = (
        tokens + positional_encoding_flat
    )

    print(
        "Position-aware tokens:",
        tuple(tokens_with_position.shape)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    feature_norms = torch.norm(
        tokens,
        dim=2
    )

    position_norms = torch.norm(
        positional_encoding_flat,
        dim=2
    )

    combined_norms = torch.norm(
        tokens_with_position,
        dim=2
    )

    print(
        "\nFeature token norm:"
    )

    print(
        "  Mean:",
        feature_norms.mean().item()
    )

    print(
        "  Min:",
        feature_norms.min().item()
    )

    print(
        "  Max:",
        feature_norms.max().item()
    )

    print(
        "\nPosition encoding norm:"
    )

    print(
        "  Mean:",
        position_norms.mean().item()
    )

    print(
        "  Min:",
        position_norms.min().item()
    )

    print(
        "  Max:",
        position_norms.max().item()
    )

    print(
        "\nPosition-aware token norm:"
    )

    print(
        "  Mean:",
        combined_norms.mean().item()
    )

    print(
        "  Min:",
        combined_norms.min().item()
    )

    print(
        "  Max:",
        combined_norms.max().item()
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    # Feature magnitude
    feature_magnitude = torch.norm(
        feature_map[0],
        dim=0
    ).numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(feature_magnitude)
    plt.title(
        f"{image_name} - CNN Feature Magnitude"
    )
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{image_name}_feature_magnitude.png"
        ),
        dpi=150
    )

    plt.close()

    # Positional encoding channel
    position_channel = positional_encoding[
        :, :, 0
    ].numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(position_channel)
    plt.title(
        f"{image_name} - Positional Encoding Channel 0"
    )
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{image_name}_position_channel_0.png"
        ),
        dpi=150
    )

    plt.close()

    # Token norm map
    token_norm_map = feature_norms[
        0
    ].reshape(
        height,
        width
    ).numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(token_norm_map)
    plt.title(
        f"{image_name} - Dense Token Norm"
    )
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{image_name}_token_norm.png"
        ),
        dpi=150
    )

    plt.close()

    return (
        feature_map,
        tokens,
        positional_encoding,
        tokens_with_position
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Stage 3 - Task 2")
    print("Dense Feature Representation")
    print("=" * 60)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    image1, tensor1, original_shape1 = load_image(
        IMAGE1_PATH
    )

    image2, tensor2, original_shape2 = load_image(
        IMAGE2_PATH
    )

    print(
        "\nImage 1 original shape:",
        original_shape1
    )

    print(
        "Image 2 original shape:",
        original_shape2
    )

    print(
        "Resized image shape:",
        image1.shape
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    model = SimpleCNN()
    model.eval()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    result1 = process_image(
        image1,
        tensor1,
        model,
        "image1"
    )

    result2 = process_image(
        image2,
        tensor2,
        model,
        "image2"
    )

    # --------------------------------------------------------
    # Save numerical summary
    # --------------------------------------------------------

    feature_map1, tokens1, pos1, combined1 = result1
    feature_map2, tokens2, pos2, combined2 = result2

    summary_path = os.path.join(
        RESULTS_DIR,
        "task_02_results.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 3 - Task 2: Dense Feature Representation\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Dataset: HPatches v_soldiers\n"
        )

        f.write(
            "Images: 1.ppm and 2.ppm\n\n"
        )

        f.write(
            f"Image 1 original shape: {original_shape1}\n"
        )

        f.write(
            f"Image 2 original shape: {original_shape2}\n"
        )

        f.write(
            f"Resized shape: {image1.shape}\n\n"
        )

        f.write(
            f"Image 1 CNN feature map: "
            f"{tuple(feature_map1.shape)}\n"
        )

        f.write(
            f"Image 2 CNN feature map: "
            f"{tuple(feature_map2.shape)}\n\n"
        )

        f.write(
            f"Image 1 dense tokens: "
            f"{tuple(tokens1.shape)}\n"
        )

        f.write(
            f"Image 2 dense tokens: "
            f"{tuple(tokens2.shape)}\n\n"
        )

        f.write(
            f"Positional encoding: "
            f"{tuple(pos1.shape)}\n"
        )

        f.write(
            f"Position-aware tokens image 1: "
            f"{tuple(combined1.shape)}\n"
        )

        f.write(
            f"Position-aware tokens image 2: "
            f"{tuple(combined2.shape)}\n\n"
        )

        f.write(
            f"Image 1 mean feature-token norm: "
            f"{torch.norm(tokens1, dim=2).mean().item():.6f}\n"
        )

        f.write(
            f"Image 2 mean feature-token norm: "
            f"{torch.norm(tokens2, dim=2).mean().item():.6f}\n"
        )

        f.write(
            f"Mean positional encoding norm: "
            f"{torch.norm(pos1, dim=2).mean().item():.6f}\n"
        )

        f.write(
            f"Image 1 mean position-aware token norm: "
            f"{torch.norm(combined1, dim=2).mean().item():.6f}\n"
        )

        f.write(
            f"Image 2 mean position-aware token norm: "
            f"{torch.norm(combined2, dim=2).mean().item():.6f}\n"
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nTask 2 completed."
    )


if __name__ == "__main__":
    main()