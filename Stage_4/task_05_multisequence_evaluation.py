import os
import sys
import time
import csv
import cv2
import torch
import numpy as np

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

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release"
)

CHECKPOINT_PATH = os.path.join(
    OFFICIAL_LOFTR_DIR,
    "weights",
    "indoor_ds.ckpt"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_05"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Add official LoFTR source directory
sys.path.insert(
    0,
    os.path.join(OFFICIAL_LOFTR_DIR, "src")
)

# ============================================================
# IMPORT OFFICIAL LOFTR
# ============================================================

from loftr import LoFTR, default_cfg


# ============================================================
# SEQUENCES
# ============================================================

VIEWPOINT_SEQUENCES = [
    "v_woman",
    "v_graffiti",
    "v_wall",
    "v_yard",
    "v_london",
    "v_bark",
]

ILLUMINATION_SEQUENCES = [
    "i_ajuntament",
    "i_bologna",
    "i_londonbridge",
    "i_santuario",
    "i_school",
    "i_zion",
]

ALL_SEQUENCES = (
    [("viewpoint", seq) for seq in VIEWPOINT_SEQUENCES]
    +
    [("illumination", seq) for seq in ILLUMINATION_SEQUENCES]
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_IMAGE_DIM = 640

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("Stage 4 - Task 5: Multi-Sequence Evaluation")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_DIR}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Number of sequences: {len(ALL_SEQUENCES)}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resize_image_keep_aspect(image, max_dim=640):
    """
    Resize image while preserving aspect ratio.

    Dimensions are adjusted to multiples of 8 because
    LoFTR's CNN backbone requires compatible spatial dimensions.
    """

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_dim / max(h, w)
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Ensure dimensions are multiples of 8
    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# HOMOGRAPHY TRANSFORMATION
# ============================================================

def transform_homography(
    H_original,
    original_shape1,
    resized_shape1,
    original_shape2,
    resized_shape2
):
    """
    Transform the HPatches homography from original image
    coordinates into resized image coordinates.

    H_resized = S2 @ H_original @ inv(S1)
    """

    h1, w1 = original_shape1[:2]
    h2, w2 = original_shape2[:2]

    rh1, rw1 = resized_shape1[:2]
    rh2, rw2 = resized_shape2[:2]

    sx1 = rw1 / w1
    sy1 = rh1 / h1

    sx2 = rw2 / w2
    sy2 = rh2 / h2

    S1 = np.array(
        [
            [sx1, 0, 0],
            [0, sy1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64
    )

    S2 = np.array(
        [
            [sx2, 0, 0],
            [0, sy2, 0],
            [0, 0, 1],
        ],
        dtype=np.float64
    )

    H_resized = (
        S2
        @ H_original
        @ np.linalg.inv(S1)
    )

    H_resized /= H_resized[2, 2]

    return H_resized


# ============================================================
# GEOMETRIC ERROR
# ============================================================

def calculate_errors(mkpts0, mkpts1, H):
    """
    Calculate Euclidean geometric error.

    Ground-truth points from image 0 are projected into
    image 1 using the HPatches homography.
    """

    if len(mkpts0) == 0:
        return np.array([])

    points = np.asarray(
        mkpts0,
        dtype=np.float64
    )

    ones = np.ones(
        (len(points), 1),
        dtype=np.float64
    )

    points_h = np.hstack(
        [points, ones]
    )

    projected = (
        H @ points_h.T
    ).T

    projected_xy = (
        projected[:, :2]
        /
        projected[:, 2:3]
    )

    errors = np.linalg.norm(
        projected_xy
        -
        np.asarray(mkpts1),
        axis=1
    )

    return errors


# ============================================================
# LOAD LOFTR
# ============================================================

def load_model():

    print("\nLoading official pretrained LoFTR...")

    model = LoFTR(
        config=default_cfg
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model = model.to(DEVICE)
    model.eval()

    print("LoFTR model loaded successfully.")

    return model


# ============================================================
# RUN LOFTR
# ============================================================

def run_loftr(
    model,
    image0,
    image1
):

    gray0 = cv2.cvtColor(
        image0,
        cv2.COLOR_BGR2GRAY
    )

    gray1 = cv2.cvtColor(
        image1,
        cv2.COLOR_BGR2GRAY
    )

    tensor0 = (
        torch.from_numpy(gray0)
        .float()
        / 255.0
    )

    tensor1 = (
        torch.from_numpy(gray1)
        .float()
        / 255.0
    )

    tensor0 = tensor0.unsqueeze(0).unsqueeze(0)
    tensor1 = tensor1.unsqueeze(0).unsqueeze(0)

    tensor0 = tensor0.to(DEVICE)
    tensor1 = tensor1.to(DEVICE)

    batch = {
        "image0": tensor0,
        "image1": tensor1,
    }

    with torch.no_grad():

        start = time.time()

        model(batch)

        inference_time = time.time() - start

    mkpts0 = (
        batch["mkpts0_f"]
        .detach()
        .cpu()
        .numpy()
    )

    mkpts1 = (
        batch["mkpts1_f"]
        .detach()
        .cpu()
        .numpy()
    )

    mconf = (
        batch["mconf"]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        mkpts0,
        mkpts1,
        mconf,
        inference_time
    )


# ============================================================
# EVALUATE ONE SEQUENCE
# ============================================================

def evaluate_sequence(
    model,
    category,
    sequence_name,
    sequence_index
):

    print("\n" + "=" * 70)
    print(
        f"[{sequence_index}/{len(ALL_SEQUENCES)}] "
        f"Evaluating {sequence_name}"
    )
    print("=" * 70)

    sequence_start = time.time()

    sequence_dir = os.path.join(
        DATASET_DIR,
        sequence_name
    )

    image1_path = os.path.join(
        sequence_dir,
        "1.ppm"
    )

    image2_path = os.path.join(
        sequence_dir,
        "2.ppm"
    )

    H_path = os.path.join(
        sequence_dir,
        "H_1_2"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.isdir(sequence_dir):

        print(
            f"Status: missing_sequence"
        )

        return {
            "category": category,
            "sequence": sequence_name,
            "status": "missing_sequence",
        }

    if not all(
        os.path.exists(path)
        for path in [
            image1_path,
            image2_path,
            H_path
        ]
    ):

        print(
            "Status: missing_files"
        )

        return {
            "category": category,
            "sequence": sequence_name,
            "status": "missing_files",
        }

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    image1_original = cv2.imread(
        image1_path,
        cv2.IMREAD_COLOR
    )

    image2_original = cv2.imread(
        image2_path,
        cv2.IMREAD_COLOR
    )

    if image1_original is None or image2_original is None:

        print(
            "Status: image_load_failed"
        )

        return {
            "category": category,
            "sequence": sequence_name,
            "status": "image_load_failed",
        }

    original_shape1 = image1_original.shape
    original_shape2 = image2_original.shape

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image1, _ = resize_image_keep_aspect(
        image1_original,
        MAX_IMAGE_DIM
    )

    image2, _ = resize_image_keep_aspect(
        image2_original,
        MAX_IMAGE_DIM
    )

    # --------------------------------------------------------
    # Load homography
    # --------------------------------------------------------

    H_original = np.loadtxt(
        H_path
    )

    H_resized = transform_homography(
        H_original,
        original_shape1,
        image1.shape,
        original_shape2,
        image2.shape
    )

    # --------------------------------------------------------
    # LoFTR inference
    # --------------------------------------------------------

    try:

        (
            mkpts0,
            mkpts1,
            mconf,
            inference_time
        ) = run_loftr(
            model,
            image1,
            image2
        )

    except Exception as e:

        print(
            f"Status: inference_failed"
        )

        print(
            f"Error: {e}"
        )

        return {
            "category": category,
            "sequence": sequence_name,
            "status": "inference_failed",
            "error_message": str(e),
        }

    num_matches = len(mkpts0)

    # --------------------------------------------------------
    # No matches
    # --------------------------------------------------------

    if num_matches == 0:

        print(
            "Status: no_matches"
        )

        sequence_runtime = (
            time.time()
            -
            sequence_start
        )

        print(
            f"Sequence runtime: "
            f"{sequence_runtime:.2f} s"
        )

        return {
            "category": category,
            "sequence": sequence_name,
            "status": "no_matches",
            "num_matches": 0,
            "inference_time": inference_time,
            "sequence_runtime": sequence_runtime,
        }

    # --------------------------------------------------------
    # Geometric errors
    # --------------------------------------------------------

    errors = calculate_errors(
        mkpts0,
        mkpts1,
        H_resized
    )

    # Make sure confidence and points remain aligned
    valid_count = min(
        len(errors),
        len(mconf)
    )

    errors = errors[:valid_count]
    mconf_valid = mconf[:valid_count]

    if len(errors) == 0:

        print(
            "Status: no_valid_geometric_evaluations"
        )

        return {
            "category": category,
            "sequence": sequence_name,
            "status": "no_valid_geometric_evaluations",
            "num_matches": num_matches,
            "inference_time": inference_time,
        }

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_error = float(
        np.mean(errors)
    )

    median_error = float(
        np.median(errors)
    )

    min_error = float(
        np.min(errors)
    )

    max_error = float(
        np.max(errors)
    )

    within_1 = int(
        np.sum(errors <= 1.0)
    )

    within_3 = int(
        np.sum(errors <= 3.0)
    )

    within_5 = int(
        np.sum(errors <= 5.0)
    )

    within_10 = int(
        np.sum(errors <= 10.0)
    )

    mean_confidence = float(
        np.mean(mconf_valid)
    )

    median_confidence = float(
        np.median(mconf_valid)
    )

    min_confidence = float(
        np.min(mconf_valid)
    )

    max_confidence = float(
        np.max(mconf_valid)
    )

    sequence_runtime = (
        time.time()
        -
        sequence_start
    )

    # --------------------------------------------------------
    # Save per-sequence data
    # --------------------------------------------------------

    sequence_results_dir = os.path.join(
        RESULTS_DIR,
        sequence_name
    )

    os.makedirs(
        sequence_results_dir,
        exist_ok=True
    )

    np.savez(
        os.path.join(
            sequence_results_dir,
            "matches_and_errors.npz"
        ),
        mkpts0=mkpts0[:valid_count],
        mkpts1=mkpts1[:valid_count],
        confidence=mconf_valid,
        errors=errors,
        H_original=H_original,
        H_resized=H_resized,
    )

    # --------------------------------------------------------
    # Save error CSV
    # --------------------------------------------------------

    error_csv_path = os.path.join(
        sequence_results_dir,
        "errors.csv"
    )

    with open(
        error_csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "match_index",
                "x0",
                "y0",
                "x1",
                "y1",
                "confidence",
                "geometric_error_px",
            ]
        )

        for i in range(valid_count):

            writer.writerow(
                [
                    i,
                    mkpts0[i, 0],
                    mkpts0[i, 1],
                    mkpts1[i, 0],
                    mkpts1[i, 1],
                    mconf_valid[i],
                    errors[i],
                ]
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "Status: success"
    )

    print(
        f"Matches: {num_matches}"
    )

    print(
        f"Inference: "
        f"{inference_time:.2f} s"
    )

    print(
        f"Mean error: "
        f"{mean_error:.4f} px"
    )

    print(
        f"Median error: "
        f"{median_error:.4f} px"
    )

    print(
        f"≤ 1 px: "
        f"{within_1}/{valid_count} "
        f"({within_1 / valid_count * 100:.2f}%)"
    )

    print(
        f"≤ 3 px: "
        f"{within_3}/{valid_count} "
        f"({within_3 / valid_count * 100:.2f}%)"
    )

    print(
        f"≤ 5 px: "
        f"{within_5}/{valid_count} "
        f"({within_5 / valid_count * 100:.2f}%)"
    )

    print(
        f"≤ 10 px: "
        f"{within_10}/{valid_count} "
        f"({within_10 / valid_count * 100:.2f}%)"
    )

    print(
        f"Mean confidence: "
        f"{mean_confidence:.4f}"
    )

    print(
        f"Sequence runtime: "
        f"{sequence_runtime:.2f} s"
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "category": category,
        "sequence": sequence_name,
        "status": "success",
        "num_matches": num_matches,
        "valid_evaluations": valid_count,
        "inference_time": inference_time,
        "sequence_runtime": sequence_runtime,
        "mean_error": mean_error,
        "median_error": median_error,
        "min_error": min_error,
        "max_error": max_error,
        "within_1_px": within_1,
        "within_3_px": within_3,
        "within_5_px": within_5,
        "within_10_px": within_10,
        "within_1_rate": within_1 / valid_count,
        "within_3_rate": within_3 / valid_count,
        "within_5_rate": within_5 / valid_count,
        "within_10_rate": within_10 / valid_count,
        "mean_confidence": mean_confidence,
        "median_confidence": median_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
    }


# ============================================================
# SAVE AGGREGATED CSV
# ============================================================

def save_results_csv(results):

    csv_path = os.path.join(
        RESULTS_DIR,
        "multisequence_results.csv"
    )

    fieldnames = [
        "category",
        "sequence",
        "status",
        "num_matches",
        "valid_evaluations",
        "inference_time",
        "sequence_runtime",
        "mean_error",
        "median_error",
        "min_error",
        "max_error",
        "within_1_px",
        "within_3_px",
        "within_5_px",
        "within_10_px",
        "within_1_rate",
        "within_3_rate",
        "within_5_rate",
        "within_10_rate",
        "mean_confidence",
        "median_confidence",
        "min_confidence",
        "max_confidence",
    ]

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            row = {
                field: result.get(
                    field,
                    ""
                )
                for field in fieldnames
            }

            writer.writerow(row)

    return csv_path


# ============================================================
# SAVE SUMMARY REPORT
# ============================================================

def save_summary(results, total_runtime):

    summary_path = os.path.join(
        RESULTS_DIR,
        "multisequence_summary.txt"
    )

    successful = [
        r for r in results
        if r.get("status") == "success"
    ]

    viewpoint_results = [
        r for r in successful
        if r.get("category") == "viewpoint"
    ]

    illumination_results = [
        r for r in successful
        if r.get("category") == "illumination"
    ]

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 4 - Task 5\n"
        )

        f.write(
            "Official LoFTR Multi-Sequence Evaluation\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Device: {DEVICE}\n"
        )

        f.write(
            f"Maximum image dimension: "
            f"{MAX_IMAGE_DIM}px\n"
        )

        f.write(
            f"Total sequences requested: "
            f"{len(results)}\n"
        )

        f.write(
            f"Successful sequences: "
            f"{len(successful)}\n"
        )

        f.write(
            f"Total runtime: "
            f"{total_runtime:.2f} seconds\n\n"
        )

        # ----------------------------------------------------
        # Detailed results
        # ----------------------------------------------------

        f.write(
            "SEQUENCE RESULTS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for r in results:

            f.write(
                f"\n{r['sequence']} "
                f"({r['category']})\n"
            )

            f.write(
                f"Status: "
                f"{r['status']}\n"
            )

            if r["status"] == "success":

                f.write(
                    f"Matches: "
                    f"{r['num_matches']}\n"
                )

                f.write(
                    f"Mean error: "
                    f"{r['mean_error']:.4f} px\n"
                )

                f.write(
                    f"Median error: "
                    f"{r['median_error']:.4f} px\n"
                )

                f.write(
                    f"≤ 1 px: "
                    f"{r['within_1_px']}/"
                    f"{r['valid_evaluations']} "
                    f"("
                    f"{r['within_1_rate'] * 100:.2f}%"
                    f")\n"
                )

                f.write(
                    f"≤ 3 px: "
                    f"{r['within_3_px']}/"
                    f"{r['valid_evaluations']} "
                    f"("
                    f"{r['within_3_rate'] * 100:.2f}%"
                    f")\n"
                )

                f.write(
                    f"≤ 5 px: "
                    f"{r['within_5_px']}/"
                    f"{r['valid_evaluations']} "
                    f"("
                    f"{r['within_5_rate'] * 100:.2f}%"
                    f")\n"
                )

                f.write(
                    f"≤ 10 px: "
                    f"{r['within_10_px']}/"
                    f"{r['valid_evaluations']} "
                    f"("
                    f"{r['within_10_rate'] * 100:.2f}%"
                    f")\n"
                )

                f.write(
                    f"Mean confidence: "
                    f"{r['mean_confidence']:.4f}\n"
                )

                f.write(
                    f"Inference time: "
                    f"{r['inference_time']:.2f} s\n"
                )

        # ----------------------------------------------------
        # Category summary
        # ----------------------------------------------------

        f.write(
            "\n\nCATEGORY SUMMARY\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            f"Viewpoint successful sequences: "
            f"{len(viewpoint_results)}\n"
        )

        f.write(
            f"Illumination successful sequences: "
            f"{len(illumination_results)}\n"
        )

        # ----------------------------------------------------
        # Weighted aggregate statistics
        # ----------------------------------------------------

        for category, category_results in [
            ("Viewpoint", viewpoint_results),
            ("Illumination", illumination_results),
        ]:

            if not category_results:
                continue

            total_matches = sum(
                r["valid_evaluations"]
                for r in category_results
            )

            all_errors = []

            for r in category_results:

                sequence_dir = os.path.join(
                    RESULTS_DIR,
                    r["sequence"]
                )

                npz_path = os.path.join(
                    sequence_dir,
                    "matches_and_errors.npz"
                )

                if os.path.exists(npz_path):

                    data = np.load(
                        npz_path
                    )

                    all_errors.extend(
                        data["errors"].tolist()
                    )

            if all_errors:

                all_errors = np.asarray(
                    all_errors
                )

                f.write(
                    f"\n{category}\n"
                )

                f.write(
                    f"Total valid matches: "
                    f"{total_matches}\n"
                )

                f.write(
                    f"Aggregate mean error: "
                    f"{np.mean(all_errors):.4f} px\n"
                )

                f.write(
                    f"Aggregate median error: "
                    f"{np.median(all_errors):.4f} px\n"
                )

                f.write(
                    f"Aggregate ≤3 px: "
                    f"{np.mean(all_errors <= 3.0) * 100:.2f}%\n"
                )

                f.write(
                    f"Aggregate ≤5 px: "
                    f"{np.mean(all_errors <= 5.0) * 100:.2f}%\n"
                )

                f.write(
                    f"Aggregate ≤10 px: "
                    f"{np.mean(all_errors <= 10.0) * 100:.2f}%\n"
                )

    return summary_path


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    # --------------------------------------------------------
    # Load model once
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Evaluate all sequences
    # --------------------------------------------------------

    results = []

    for index, (category, sequence_name) in enumerate(
        ALL_SEQUENCES,
        start=1
    ):

        result = evaluate_sequence(
            model,
            category,
            sequence_name,
            index
        )

        results.append(result)

    # --------------------------------------------------------
    # Total runtime
    # --------------------------------------------------------

    total_runtime = (
        time.time()
        -
        total_start
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    csv_path = save_results_csv(
        results
    )

    summary_path = save_summary(
        results,
        total_runtime
    )

    successful = sum(
        r.get("status") == "success"
        for r in results
    )

    print("\n")
    print("=" * 70)
    print("Task 5 completed.")
    print("=" * 70)

    print(
        f"Successful sequences: "
        f"{successful} / {len(results)}"
    )

    print(
        f"Total runtime: "
        f"{total_runtime:.2f} seconds"
    )

    print("\nCSV results:")
    print(csv_path)

    print("\nSummary report:")
    print(summary_path)


if __name__ == "__main__":
    main()