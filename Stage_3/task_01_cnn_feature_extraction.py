import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE1_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_soldiers",
    "1.ppm"
)

IMAGE2_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_soldiers",
    "2.ppm"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_01"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 256

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD IMAGES
# ============================================================

image1 = cv2.imread(
    IMAGE1_PATH,
    cv2.IMREAD_GRAYSCALE
)

image2 = cv2.imread(
    IMAGE2_PATH,
    cv2.IMREAD_GRAYSCALE
)


if image1 is None:
    raise FileNotFoundError(
        f"Could not load Image 1:\n{IMAGE1_PATH}"
    )

if image2 is None:
    raise FileNotFoundError(
        f"Could not load Image 2:\n{IMAGE2_PATH}"
    )


original_shape1 = image1.shape
original_shape2 = image2.shape


# ============================================================
# RESIZE
# ============================================================

image1 = cv2.resize(
    image1,
    (IMAGE_SIZE, IMAGE_SIZE)
)

image2 = cv2.resize(
    image2,
    (IMAGE_SIZE, IMAGE_SIZE)
)


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

tensor1 = torch.tensor(
    image1,
    dtype=torch.float32
) / 255.0

tensor2 = torch.tensor(
    image2,
    dtype=torch.float32
) / 255.0


# Add batch and channel dimensions
#
# [H, W]
#   ↓
# [1, 1, H, W]

tensor1 = tensor1.unsqueeze(
    0
).unsqueeze(
    0
)

tensor2 = tensor2.unsqueeze(
    0
).unsqueeze(
    0
)


# ============================================================
# CNN FEATURE EXTRACTOR
# ============================================================

class SimpleCNNFeatureExtractor(
    nn.Module
):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )


        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )


        # ----------------------------------------------------
        # Spatial downsampling
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ----------------------------------------------------
        # Third convolution
        # ----------------------------------------------------

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )


        self.relu = nn.ReLU()


    def forward(
        self,
        x
    ):

        feature_shapes = {}


        # ----------------------------------------------------
        # Conv 1
        # ----------------------------------------------------

        x = self.conv1(x)

        feature_shapes[
            "conv1"
        ] = tuple(x.shape)


        x = self.relu(x)


        # ----------------------------------------------------
        # Conv 2
        # ----------------------------------------------------

        x = self.conv2(x)

        feature_shapes[
            "conv2"
        ] = tuple(x.shape)


        x = self.relu(x)


        # ----------------------------------------------------
        # Max Pool
        # ----------------------------------------------------

        x = self.pool(x)

        feature_shapes[
            "pool"
        ] = tuple(x.shape)


        # ----------------------------------------------------
        # Conv 3
        # ----------------------------------------------------

        x = self.conv3(x)

        feature_shapes[
            "conv3"
        ] = tuple(x.shape)


        x = self.relu(x)


        feature_shapes[
            "output"
        ] = tuple(x.shape)


        return x, feature_shapes


# ============================================================
# CREATE MODEL
# ============================================================

model = SimpleCNNFeatureExtractor()

model.eval()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

with torch.no_grad():

    features1, shapes1 = model(
        tensor1
    )

    features2, shapes2 = model(
        tensor2
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

def feature_statistics(
    features
):

    return {

        "mean": float(
            features.mean().item()
        ),

        "std": float(
            features.std().item()
        ),

        "min": float(
            features.min().item()
        ),

        "max": float(
            features.max().item()
        ),

        "mean_abs": float(
            features.abs().mean().item()
        )
    }


stats1 = feature_statistics(
    features1
)

stats2 = feature_statistics(
    features2
)


# ============================================================
# FEATURE MAP MAGNITUDE
# ============================================================

def feature_magnitude(
    features
):

    # features:
    # [B, C, H, W]

    magnitude = torch.norm(
        features,
        p=2,
        dim=1
    )

    return magnitude


magnitude1 = feature_magnitude(
    features1
)

magnitude2 = feature_magnitude(
    features2
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

features1_np = (
    features1[0]
    .cpu()
    .numpy()
)

features2_np = (
    features2[0]
    .cpu()
    .numpy()
)

magnitude1_np = (
    magnitude1[0]
    .cpu()
    .numpy()
)

magnitude2_np = (
    magnitude2[0]
    .cpu()
    .numpy()
)


# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def save_feature_channel(
    feature_map,
    channel,
    filename,
    title
):

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        feature_map[channel],
        cmap="viridis"
    )

    plt.colorbar()

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# SAVE INPUT IMAGE 1
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    image1,
    cmap="gray"
)

plt.title(
    "v_soldiers - Image 1"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "input_image_1.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# SAVE INPUT IMAGE 2
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    image2,
    cmap="gray"
)

plt.title(
    "v_soldiers - Image 2"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "input_image_2.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# FEATURE CHANNELS — IMAGE 1
# ============================================================

channels_to_visualize = [
    0,
    7,
    15,
    31
]


for channel in channels_to_visualize:

    save_feature_channel(
        features1_np,
        channel,
        f"feature_channel_{channel + 1}.png",
        f"Image 1 - Feature Channel {channel + 1}"
    )


# ============================================================
# FEATURE CHANNELS — IMAGE 2
# ============================================================

for channel in channels_to_visualize:

    save_feature_channel(
        features2_np,
        channel,
        f"image2_feature_channel_{channel + 1}.png",
        f"Image 2 - Feature Channel {channel + 1}"
    )


# ============================================================
# FEATURE MAGNITUDE — IMAGE 1
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    magnitude1_np,
    cmap="viridis"
)

plt.colorbar(
    label="Feature Magnitude"
)

plt.title(
    "CNN Feature Magnitude - Image 1"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "feature_magnitude_image1.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# FEATURE MAGNITUDE — IMAGE 2
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    magnitude2_np,
    cmap="viridis"
)

plt.colorbar(
    label="Feature Magnitude"
)

plt.title(
    "CNN Feature Magnitude - Image 2"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "feature_magnitude_image2.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    RESULTS_DIR,
    "task_01_results.txt"
)


with open(
    results_path,
    "w"
) as f:

    f.write(
        "Stage 3 Task 1 — CNN Feature Extraction\n"
    )

    f.write(
        "=========================================\n\n"
    )


    # --------------------------------------------------------
    # Input information
    # --------------------------------------------------------

    f.write(
        "INPUT IMAGES\n"
    )

    f.write(
        "------------\n"
    )

    f.write(
        f"Dataset sequence: v_soldiers\n"
    )

    f.write(
        f"Image 1 original shape: "
        f"{original_shape1}\n"
    )

    f.write(
        f"Image 2 original shape: "
        f"{original_shape2}\n"
    )

    f.write(
        f"Resized shape: "
        f"({IMAGE_SIZE}, {IMAGE_SIZE})\n"
    )


    # --------------------------------------------------------
    # Network architecture
    # --------------------------------------------------------

    f.write(
        "\nCNN ARCHITECTURE\n"
    )

    f.write(
        "----------------\n"
    )

    f.write(
        "Conv2D: 1 -> 32, kernel=3, padding=1\n"
    )

    f.write(
        "ReLU\n"
    )

    f.write(
        "Conv2D: 32 -> 64, kernel=3, padding=1\n"
    )

    f.write(
        "ReLU\n"
    )

    f.write(
        "MaxPool2D: kernel=2, stride=2\n"
    )

    f.write(
        "Conv2D: 64 -> 64, kernel=3, padding=1\n"
    )

    f.write(
        "ReLU\n"
    )


    # --------------------------------------------------------
    # Feature shapes
    # --------------------------------------------------------

    f.write(
        "\nFEATURE SHAPES — IMAGE 1\n"
    )

    f.write(
        "-------------------------\n"
    )

    for name, shape in shapes1.items():

        f.write(
            f"{name}: {shape}\n"
        )


    f.write(
        "\nFEATURE SHAPES — IMAGE 2\n"
    )

    f.write(
        "-------------------------\n"
    )

    for name, shape in shapes2.items():

        f.write(
            f"{name}: {shape}\n"
        )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    f.write(
        "\nFEATURE STATISTICS — IMAGE 1\n"
    )

    f.write(
        "-----------------------------\n"
    )

    for key, value in stats1.items():

        f.write(
            f"{key}: {value:.6f}\n"
        )


    f.write(
        "\nFEATURE STATISTICS — IMAGE 2\n"
    )

    f.write(
        "-----------------------------\n"
    )

    for key, value in stats2.items():

        f.write(
            f"{key}: {value:.6f}\n"
        )


    # --------------------------------------------------------
    # Important note
    # --------------------------------------------------------

    f.write(
        "\nNOTE\n"
    )

    f.write(
        "----\n"
    )

    f.write(
        "The CNN is randomly initialized and not trained.\n"
    )

    f.write(
        "Therefore, the feature maps demonstrate the "
        "architecture and dense spatial representation, "
        "not learned semantic or correspondence features.\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()
print(
    "=============================================="
)

print(
    "Stage 3 Task 1 completed"
)

print(
    "=============================================="
)


print(
    f"Dataset sequence: v_soldiers"
)

print(
    f"Original Image 1: "
    f"{original_shape1}"
)

print(
    f"Original Image 2: "
    f"{original_shape2}"
)

print(
    f"Resized images: "
    f"{IMAGE_SIZE} x {IMAGE_SIZE}"
)


print()
print(
    "IMAGE 1 FEATURE SHAPES"
)

print(
    "----------------------------------------------"
)

for name, shape in shapes1.items():

    print(
        f"{name}: {shape}"
    )


print()
print(
    "IMAGE 2 FEATURE SHAPES"
)

print(
    "----------------------------------------------"
)

for name, shape in shapes2.items():

    print(
        f"{name}: {shape}"
    )


print()
print(
    "FINAL FEATURE MAP"
)

print(
    "----------------------------------------------"
)

print(
    f"Image 1: "
    f"{features1.shape}"
)

print(
    f"Image 2: "
    f"{features2.shape}"
)


print()
print(
    "IMAGE 1 FEATURE STATISTICS"
)

print(
    "----------------------------------------------"
)

for key, value in stats1.items():

    print(
        f"{key}: {value:.6f}"
    )


print()
print(
    "IMAGE 2 FEATURE STATISTICS"
)

print(
    "----------------------------------------------"
)

for key, value in stats2.items():

    print(
        f"{key}: {value:.6f}"
    )


print()
print(
    "Results saved to:"
)

print(
    RESULTS_DIR
)

print(
    "=============================================="
)