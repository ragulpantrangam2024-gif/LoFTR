"""
Stage 4 - Task 3
Quantitative Evaluation of Official Pretrained LoFTR

Purpose:
    Evaluate the geometric accuracy of official pretrained LoFTR
    correspondences using the HPatches ground-truth homography.

Dataset:
    HPatches - v_soldiers

Input:
    Stage 4 Task 2 LoFTR matches
    v_soldiers/H_1_2

Evaluation:
    - Homography reprojection error
    - Mean error
    - Median error
    - Minimum / maximum error
    - Accuracy within 1 px
    - Accuracy within 3 px
    - Accuracy within 5 px
    - Accuracy within 10 px
    - Precision
    - Recall
    - Inlier ratio

Important:
    Task 2 used resized images (640 x 480), therefore the original
    HPatches homography is transformed into the resized coordinate
    system before evaluation.
"""

import os

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SEQUENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "hpatches-sequences-release",
    "v_soldiers"
)

MATCHES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_02",
    "matches.npz"
)

HOMOGRAPHY_PATH = os.path.join(
    SEQUENCE_DIR,
    "H_1_2"
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
# EVALUATION SETTINGS
# ============================================================

THRESHOLDS = [
    1.0,
    3.0,
    5.0,
    10.0
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_homography(path):
    """
    Load HPatches homography.

    HPatches stores the homography as text.
    """

    H = np.loadtxt(path)

    if H.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 homography, got {H.shape}"
        )

    return H.astype(np.float64)


def compute_resize_matrix(
    original_shape,
    resized_shape
):
    """
    Construct a coordinate scaling matrix.

    OpenCV image shape:
        (height, width)

    Coordinate convention:
        x = width
        y = height
    """

    original_height, original_width = original_shape
    resized_height, resized_width = resized_shape

    scale_x = resized_width / original_width
    scale_y = resized_height / original_height

    S = np.array(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )

    return S


def transform_homography(
    H_original,
    original_shape_0,
    original_shape_1,
    resized_shape_0,
    resized_shape_1
):
    """
    Transform the original homography into the resized
    image coordinate system.

    H_resized = S2 @ H_original @ inv(S1)
    """

    S1 = compute_resize_matrix(
        original_shape_0,
        resized_shape_0
    )

    S2 = compute_resize_matrix(
        original_shape_1,
        resized_shape_1
    )

    H_resized = (
        S2
        @ H_original
        @ np.linalg.inv(S1)
    )

    # Normalize.
    H_resized = (
        H_resized
        / H_resized[2, 2]
    )

    return H_resized


def project_points(
    points,
    H
):
    """
    Project 2D points using a homography.

    points:
        N x 2
    """

    if len(points) == 0:
        return np.empty(
            (0, 2),
            dtype=np.float64
        )

    points_h = np.concatenate(
        [
            points,
            np.ones(
                (len(points), 1),
                dtype=np.float64
            )
        ],
        axis=1
    )

    projected_h = (
        H
        @ points_h.T
    ).T

    denominator = projected_h[:, 2]

    valid = np.abs(denominator) > 1e-12

    projected = np.full(
        (len(points), 2),
        np.nan,
        dtype=np.float64
    )

    projected[valid] = (
        projected_h[valid, :2]
        /
        denominator[valid, None]
    )

    return projected


def calculate_errors(
    mkpts0,
    mkpts1,
    H
):
    """
    Calculate geometric reprojection error.

    For each point in image 0:

        predicted point in image 1
            =
        H @ point

    Error:

        ||predicted - LoFTR_prediction||
    """

    projected = project_points(
        mkpts0,
        H
    )

    errors = np.linalg.norm(
        projected - mkpts1,
        axis=1
    )

    valid = np.isfinite(errors)

    return (
        errors[valid],
        projected[valid]
    )


def calculate_metrics(
    errors,
    total_matches
):
    """
    Calculate accuracy metrics.
    """

    metrics = {}

    if len(errors) == 0:
        return metrics

    metrics["mean_error"] = float(
        np.mean(errors)
    )

    metrics["median_error"] = float(
        np.median(errors)
    )

    metrics["min_error"] = float(
        np.min(errors)
    )

    metrics["max_error"] = float(
        np.max(errors)
    )

    for threshold in THRESHOLDS:

        correct = int(
            np.sum(errors <= threshold)
        )

        accuracy = (
            correct / len(errors)
        ) * 100.0

        metrics[
            f"within_{threshold:g}_px"
        ] = correct

        metrics[
            f"accuracy_{threshold:g}_px"
        ] = accuracy

    # Use 3 px as the primary geometric correctness threshold.
    inliers = int(
        np.sum(errors <= 3.0)
    )

    precision = (
        inliers / total_matches
    ) * 100.0 if total_matches > 0 else 0.0

    metrics["inliers_3px"] = inliers
    metrics["precision_3px"] = precision

    # For this HPatches pair, all valid LoFTR matches are treated
    # as candidate correspondences. Recall is therefore reported
    # as the fraction of candidate matches that satisfy the
    # geometric correctness criterion.
    #
    # This is a candidate-level retention measure and should not
    # be confused with recall over all possible image correspondences.
    metrics["candidate_recall_3px"] = precision

    return metrics


def save_error_distribution(
    errors,
    output_path
):
    """
    Save all geometric errors to CSV.
    """

    indices = np.arange(
        len(errors)
    )

    data = np.column_stack(
        [
            indices,
            errors
        ]
    )

    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="match_index,error_px",
        comments=""
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Stage 4 - Task 3")
    print("Quantitative Evaluation of Official LoFTR")
    print("=" * 60)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(MATCHES_PATH):
        raise FileNotFoundError(
            f"LoFTR matches not found:\n{MATCHES_PATH}"
        )

    if not os.path.exists(HOMOGRAPHY_PATH):
        raise FileNotFoundError(
            f"HPatches homography not found:\n{HOMOGRAPHY_PATH}"
        )

    # --------------------------------------------------------
    # LOAD MATCHES
    # --------------------------------------------------------

    data = np.load(
        MATCHES_PATH
    )

    mkpts0 = data["mkpts0"].astype(
        np.float64
    )

    mkpts1 = data["mkpts1"].astype(
        np.float64
    )

    mconf = data["mconf"].astype(
        np.float64
    )

    original_shape_0 = tuple(
        data["original_image0_shape"].tolist()
    )

    original_shape_1 = tuple(
        data["original_image1_shape"].tolist()
    )

    resized_shape_0 = tuple(
        data["image0_shape"].tolist()
    )

    resized_shape_1 = tuple(
        data["image1_shape"].tolist()
    )

    total_matches = len(
        mkpts0
    )

    print()
    print("Matches loaded:", total_matches)

    print(
        "Original image 0:",
        original_shape_0
    )

    print(
        "Original image 1:",
        original_shape_1
    )

    print(
        "Resized image 0:",
        resized_shape_0
    )

    print(
        "Resized image 1:",
        resized_shape_1
    )

    # --------------------------------------------------------
    # LOAD HOMOGRAPHY
    # --------------------------------------------------------

    H_original = load_homography(
        HOMOGRAPHY_PATH
    )

    print()
    print("Original HPatches homography:")
    print(H_original)

    # --------------------------------------------------------
    # TRANSFORM HOMOGRAPHY
    # --------------------------------------------------------

    H_resized = transform_homography(
        H_original,
        original_shape_0,
        original_shape_1,
        resized_shape_0,
        resized_shape_1
    )

    print()
    print("Resized-coordinate homography:")
    print(H_resized)

    # --------------------------------------------------------
    # GEOMETRIC ERROR
    # --------------------------------------------------------

    errors, projected = calculate_errors(
        mkpts0,
        mkpts1,
        H_resized
    )

    if len(errors) == 0:
        raise RuntimeError(
            "No valid geometric errors could be calculated."
        )

    print()
    print("Valid geometric evaluations:", len(errors))

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        errors,
        total_matches
    )

    print()
    print("=" * 60)
    print("GEOMETRIC EVALUATION")
    print("=" * 60)

    print(
        f"Mean error:   "
        f"{metrics['mean_error']:.4f} px"
    )

    print(
        f"Median error: "
        f"{metrics['median_error']:.4f} px"
    )

    print(
        f"Minimum error: "
        f"{metrics['min_error']:.4f} px"
    )

    print(
        f"Maximum error: "
        f"{metrics['max_error']:.4f} px"
    )

    print()

    for threshold in THRESHOLDS:

        key_count = (
            f"within_{threshold:g}_px"
        )

        key_accuracy = (
            f"accuracy_{threshold:g}_px"
        )

        print(
            f"Within {threshold:g} px: "
            f"{metrics[key_count]}/"
            f"{total_matches} "
            f"({metrics[key_accuracy]:.2f}%)"
        )

    print()

    print(
        "3 px geometric inliers:",
        metrics["inliers_3px"]
    )

    print(
        "3 px precision:",
        f"{metrics['precision_3px']:.2f}%"
    )

    print(
        "Candidate recall at 3 px:",
        f"{metrics['candidate_recall_3px']:.2f}%"
    )

    # --------------------------------------------------------
    # CONFIDENCE VS ERROR
    # --------------------------------------------------------

    confidence_error = np.column_stack(
        [
            np.arange(total_matches),
            mconf,
            errors
        ]
    )

    confidence_error_path = os.path.join(
        RESULTS_DIR,
        "confidence_vs_error.csv"
    )

    np.savetxt(
        confidence_error_path,
        confidence_error,
        delimiter=",",
        header="match_index,confidence,error_px",
        comments=""
    )

    # --------------------------------------------------------
    # ERROR DISTRIBUTION
    # --------------------------------------------------------

    error_path = os.path.join(
        RESULTS_DIR,
        "geometric_errors.csv"
    )

    save_error_distribution(
        errors,
        error_path
    )

    # --------------------------------------------------------
    # SAVE PROJECTED POINTS
    # --------------------------------------------------------

    projected_path = os.path.join(
        RESULTS_DIR,
        "projected_points.npz"
    )

    np.savez(
        projected_path,
        mkpts0=mkpts0,
        mkpts1=mkpts1,
        projected=projected,
        errors=errors,
        confidence=mconf,
        H_original=H_original,
        H_resized=H_resized
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_path = os.path.join(
        RESULTS_DIR,
        "evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 4 - Task 3\n"
        )

        f.write(
            "Quantitative Evaluation of Official "
            "Pretrained LoFTR\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Dataset: HPatches v_soldiers\n"
        )

        f.write(
            f"Total LoFTR matches: "
            f"{total_matches}\n\n"
        )

        f.write(
            "Original image shapes:\n"
        )

        f.write(
            f"  Image 0: {original_shape_0}\n"
        )

        f.write(
            f"  Image 1: {original_shape_1}\n\n"
        )

        f.write(
            "Resized image shapes:\n"
        )

        f.write(
            f"  Image 0: {resized_shape_0}\n"
        )

        f.write(
            f"  Image 1: {resized_shape_1}\n\n"
        )

        f.write(
            "Evaluation homography:\n"
        )

        f.write(
            str(H_resized)
            + "\n\n"
        )

        f.write(
            f"Mean geometric error: "
            f"{metrics['mean_error']:.6f} px\n"
        )

        f.write(
            f"Median geometric error: "
            f"{metrics['median_error']:.6f} px\n"
        )

        f.write(
            f"Minimum error: "
            f"{metrics['min_error']:.6f} px\n"
        )

        f.write(
            f"Maximum error: "
            f"{metrics['max_error']:.6f} px\n\n"
        )

        for threshold in THRESHOLDS:

            f.write(
                f"Within {threshold:g} px: "
                f"{metrics[f'within_{threshold:g}_px']}/"
                f"{total_matches} "
                f"("
                f"{metrics[f'accuracy_{threshold:g}_px']:.2f}%"
                f")\n"
            )

        f.write("\n")

        f.write(
            f"3 px geometric inliers: "
            f"{metrics['inliers_3px']}\n"
        )

        f.write(
            f"3 px precision: "
            f"{metrics['precision_3px']:.4f}%\n"
        )

        f.write(
            f"Candidate recall at 3 px: "
            f"{metrics['candidate_recall_3px']:.4f}%\n\n"
        )

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "Geometric error measures the distance between "
            "the LoFTR predicted point in image 1 and the "
            "point obtained by projecting the image-0 point "
            "using the HPatches ground-truth homography.\n\n"
        )

        f.write(
            "The 3 px precision value represents the fraction "
            "of evaluated LoFTR correspondences that are "
            "consistent with the ground-truth homography "
            "within 3 pixels.\n\n"
        )

        f.write(
            "Candidate recall is reported here as a "
            "candidate-level geometric retention measure. "
            "It should not be interpreted as recall over all "
            "possible image correspondences.\n"
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Task 3 completed.")
    print("=" * 60)

    print()
    print("Results saved to:")
    print(RESULTS_DIR)

    print()
    print("Generated files:")
    print("  - evaluation_summary.txt")
    print("  - geometric_errors.csv")
    print("  - confidence_vs_error.csv")
    print("  - projected_points.npz")


if __name__ == "__main__":
    main()