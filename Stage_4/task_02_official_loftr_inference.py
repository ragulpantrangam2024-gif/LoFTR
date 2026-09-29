"""
Stage 4 - Task 2
Official Pretrained LoFTR Inference

Purpose:
    Run the official ZJU3DV LoFTR implementation using a pretrained
    indoor checkpoint.

Dataset:
    HPatches - v_soldiers

This task uses the official pretrained LoFTR model.
It is NOT a simplified implementation.

For the first CPU test, images are resized so that the maximum
dimension is limited to MAX_IMAGE_DIM.

Outputs:
    - matches.npz
    - matches_visualization.png
    - task_02_results.txt
"""

import os
import sys
import time

import cv2
import numpy as np
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OFFICIAL_LOFTR_DIR = os.path.join(
    PROJECT_ROOT,
    "official_loftr"
)

OFFICIAL_SRC_DIR = os.path.join(
    OFFICIAL_LOFTR_DIR,
    "src"
)

WEIGHTS_PATH = os.path.join(
    OFFICIAL_LOFTR_DIR,
    "weights",
    "indoor_ds.ckpt"
)

IMAGE_0_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_soldiers",
    "1.ppm"
)

IMAGE_1_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_soldiers",
    "2.ppm"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_02"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CPU TEST SETTINGS
# ============================================================

# Maximum image dimension for the first CPU test.
#
# Original v_soldiers images:
#     1290 x 968
#
# With MAX_IMAGE_DIM = 640:
#     approximately 640 x 480
#
# Both dimensions are adjusted to multiples of 8.
MAX_IMAGE_DIM = 640

# Number of matches shown in visualization.
# All matches are still saved in matches.npz.
MAX_VISUALIZATION_MATCHES = 300

# Confidence threshold used ONLY for reporting.
# It does not change the raw LoFTR output.
CONFIDENCE_REPORT_THRESHOLD = 0.0


# ============================================================
# IMPORT OFFICIAL LOFTR
# ============================================================

if OFFICIAL_SRC_DIR not in sys.path:
    sys.path.insert(0, OFFICIAL_SRC_DIR)

from loftr import LoFTR, default_cfg


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resize_for_loftr(image):
    """
    Resize image while preserving aspect ratio.

    The resulting dimensions are multiples of 8, which is
    required by the LoFTR architecture.
    """

    height, width = image.shape[:2]

    scale = min(
        1.0,
        MAX_IMAGE_DIM / max(height, width)
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    # Make dimensions divisible by 8.
    new_width = max(8, (new_width // 8) * 8)
    new_height = max(8, (new_height // 8) * 8)

    if new_width != width or new_height != height:
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        resized = image.copy()

    return resized


def image_to_tensor(image, device):
    """
    Convert grayscale uint8 image to LoFTR tensor.

    Output:
        [1, 1, H, W]
    """

    tensor = torch.from_numpy(
        image.astype(np.float32) / 255.0
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(device)


def save_match_visualization(
    image0,
    image1,
    mkpts0,
    mkpts1,
    confidence,
    output_path
):
    """
    Create a side-by-side visualization of LoFTR matches.

    Only the highest-confidence matches are displayed to keep
    the visualization readable. All matches remain available
    in matches.npz.
    """

    if len(mkpts0) == 0:
        print("No matches available for visualization.")
        return

    # Sort by confidence.
    order = np.argsort(-confidence)

    order = order[:MAX_VISUALIZATION_MATCHES]

    points0 = mkpts0[order]
    points1 = mkpts1[order]
    conf = confidence[order]

    # Convert grayscale to BGR.
    if len(image0.shape) == 2:
        image0_color = cv2.cvtColor(
            image0,
            cv2.COLOR_GRAY2BGR
        )
    else:
        image0_color = image0.copy()

    if len(image1.shape) == 2:
        image1_color = cv2.cvtColor(
            image1,
            cv2.COLOR_GRAY2BGR
        )
    else:
        image1_color = image1.copy()

    # Put images side by side.
    h0, w0 = image0_color.shape[:2]
    h1, w1 = image1_color.shape[:2]

    canvas_height = max(h0, h1)
    canvas_width = w0 + w1

    canvas = np.zeros(
        (canvas_height, canvas_width, 3),
        dtype=np.uint8
    )

    canvas[:h0, :w0] = image0_color
    canvas[:h1, w0:w0 + w1] = image1_color

    # Draw matches.
    for p0, p1, c in zip(points0, points1, conf):

        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])) + w0, int(round(p1[1]))

        cv2.circle(
            canvas,
            (x0, y0),
            3,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            canvas,
            (x1, y1),
            3,
            (0, 255, 0),
            -1
        )

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            1
        )

    cv2.imwrite(output_path, canvas)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Stage 4 - Task 2")
    print("Official Pretrained LoFTR Inference")
    print("=" * 60)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # PATH CHECKS
    # --------------------------------------------------------

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            "\nPretrained LoFTR checkpoint not found.\n\n"
            f"Expected location:\n{WEIGHTS_PATH}\n"
        )

    if not os.path.exists(IMAGE_0_PATH):
        raise FileNotFoundError(
            f"Image 0 not found:\n{IMAGE_0_PATH}"
        )

    if not os.path.exists(IMAGE_1_PATH):
        raise FileNotFoundError(
            f"Image 1 not found:\n{IMAGE_1_PATH}"
        )

    print("Checkpoint:", WEIGHTS_PATH)

    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    image0_original = cv2.imread(
        IMAGE_0_PATH,
        cv2.IMREAD_GRAYSCALE
    )

    image1_original = cv2.imread(
        IMAGE_1_PATH,
        cv2.IMREAD_GRAYSCALE
    )

    if image0_original is None:
        raise RuntimeError(
            f"Could not read:\n{IMAGE_0_PATH}"
        )

    if image1_original is None:
        raise RuntimeError(
            f"Could not read:\n{IMAGE_1_PATH}"
        )

    print()
    print(
        "Original image 0 shape:",
        image0_original.shape
    )

    print(
        "Original image 1 shape:",
        image1_original.shape
    )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image0 = resize_for_loftr(image0_original)
    image1 = resize_for_loftr(image1_original)

    print()
    print(
        "Processed image 0:",
        image0.shape
    )

    print(
        "Processed image 1:",
        image1.shape
    )

    # --------------------------------------------------------
    # CONVERT TO TENSOR
    # --------------------------------------------------------

    tensor0 = image_to_tensor(
        image0,
        device
    )

    tensor1 = image_to_tensor(
        image1,
        device
    )

    print()
    print("Tensor 0:", tuple(tensor0.shape))
    print("Tensor 1:", tuple(tensor1.shape))

    # --------------------------------------------------------
    # LOAD OFFICIAL LOFTR
    # --------------------------------------------------------

    print()
    print("Loading official LoFTR model...")

    model_load_start = time.perf_counter()

    matcher = LoFTR(
        config=default_cfg
    )

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=device
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    matcher.load_state_dict(
        state_dict,
        strict=True
    )

    matcher = matcher.to(device)
    matcher.eval()

    model_load_time = (
        time.perf_counter() - model_load_start
    )

    print(
        f"Model loaded in {model_load_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # PREPARE BATCH
    # --------------------------------------------------------

    batch = {
        "image0": tensor0,
        "image1": tensor1
    }

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("Starting LoFTR inference...")
    print("Please wait...")
    print("-" * 60)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    with torch.no_grad():
        matcher(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter() - inference_start
    )

    print()
    print(
        f"Inference completed in "
        f"{inference_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # GET MATCHES
    # --------------------------------------------------------

    mkpts0 = batch["mkpts0_f"]
    mkpts1 = batch["mkpts1_f"]
    mconf = batch["mconf"]

    if torch.is_tensor(mkpts0):
        mkpts0 = mkpts0.detach().cpu().numpy()

    if torch.is_tensor(mkpts1):
        mkpts1 = mkpts1.detach().cpu().numpy()

    if torch.is_tensor(mconf):
        mconf = mconf.detach().cpu().numpy()

    mkpts0 = np.asarray(mkpts0)
    mkpts1 = np.asarray(mkpts1)
    mconf = np.asarray(mconf)

    # --------------------------------------------------------
    # MATCH STATISTICS
    # --------------------------------------------------------

    num_matches = len(mkpts0)

    print()
    print("=" * 60)
    print("LoFTR MATCHING RESULTS")
    print("=" * 60)

    print(
        "Number of matches:",
        num_matches
    )

    if num_matches > 0:

        print(
            "Confidence minimum:",
            f"{mconf.min():.6f}"
        )

        print(
            "Confidence maximum:",
            f"{mconf.max():.6f}"
        )

        print(
            "Confidence mean:",
            f"{mconf.mean():.6f}"
        )

        print(
            "Confidence median:",
            f"{np.median(mconf):.6f}"
        )

        high_conf = (
            mconf >= CONFIDENCE_REPORT_THRESHOLD
        )

        print(
            f"Matches with confidence >= "
            f"{CONFIDENCE_REPORT_THRESHOLD:.2f}:",
            int(np.sum(high_conf))
        )

    # --------------------------------------------------------
    # SAVE RAW MATCHES
    # --------------------------------------------------------

    npz_path = os.path.join(
        RESULTS_DIR,
        "matches.npz"
    )

    np.savez(
        npz_path,
        mkpts0=mkpts0,
        mkpts1=mkpts1,
        mconf=mconf,
        image0_shape=np.array(image0.shape),
        image1_shape=np.array(image1.shape),
        original_image0_shape=np.array(
            image0_original.shape
        ),
        original_image1_shape=np.array(
            image1_original.shape
        )
    )

    print()
    print("Matches saved to:")
    print(npz_path)

    # --------------------------------------------------------
    # SAVE VISUALIZATION
    # --------------------------------------------------------

    visualization_path = os.path.join(
        RESULTS_DIR,
        "matches_visualization.png"
    )

    save_match_visualization(
        image0,
        image1,
        mkpts0,
        mkpts1,
        mconf,
        visualization_path
    )

    print()
    print("Visualization saved to:")
    print(visualization_path)

    # --------------------------------------------------------
    # SAVE TEXT REPORT
    # --------------------------------------------------------

    report_path = os.path.join(
        RESULTS_DIR,
        "task_02_results.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 4 - Task 2\n"
        )

        f.write(
            "Official Pretrained LoFTR Inference\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Device: {device}\n"
        )

        f.write(
            f"Checkpoint: {WEIGHTS_PATH}\n\n"
        )

        f.write(
            f"Original image 0 shape: "
            f"{image0_original.shape}\n"
        )

        f.write(
            f"Original image 1 shape: "
            f"{image1_original.shape}\n\n"
        )

        f.write(
            f"Processed image 0 shape: "
            f"{image0.shape}\n"
        )

        f.write(
            f"Processed image 1 shape: "
            f"{image1.shape}\n\n"
        )

        f.write(
            f"Maximum image dimension: "
            f"{MAX_IMAGE_DIM}\n\n"
        )

        f.write(
            f"Model loading time: "
            f"{model_load_time:.4f} seconds\n"
        )

        f.write(
            f"Inference time: "
            f"{inference_time:.4f} seconds\n\n"
        )

        f.write(
            f"Number of matches: "
            f"{num_matches}\n"
        )

        if num_matches > 0:

            f.write(
                f"Confidence minimum: "
                f"{mconf.min():.6f}\n"
            )

            f.write(
                f"Confidence maximum: "
                f"{mconf.max():.6f}\n"
            )

            f.write(
                f"Confidence mean: "
                f"{mconf.mean():.6f}\n"
            )

            f.write(
                f"Confidence median: "
                f"{np.median(mconf):.6f}\n"
            )

        f.write("\n")
        f.write(
            "Important:\n"
        )

        f.write(
            "This task uses the official pretrained "
            "ZJU3DV LoFTR implementation.\n"
        )

        f.write(
            "The images were resized for CPU inference.\n"
        )

        f.write(
            "The reported matches are raw pretrained "
            "LoFTR correspondences.\n"
        )

    print()
    print("Results report saved to:")
    print(report_path)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Task 2 completed.")
    print("=" * 60)

    print()
    print("Results directory:")
    print(RESULTS_DIR)

    print()
    print("Generated files:")

    print("  - matches.npz")
    print("  - matches_visualization.png")
    print("  - task_02_results.txt")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()