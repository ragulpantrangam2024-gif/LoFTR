"""
Stage 4 - Task 4
LoFTR Confidence Threshold Analysis

Purpose:
    Investigate the relationship between LoFTR matching confidence
    and geometric correctness.

The task reuses the matches generated in Task 2 and the geometric
errors calculated in Task 3.

No LoFTR inference is performed in this task.

For several confidence thresholds, the following are calculated:

    - retained matches
    - retained percentage
    - mean confidence
    - mean geometric error
    - median geometric error
    - accuracy within 1 px
    - accuracy within 3 px
    - accuracy within 5 px
    - accuracy within 10 px

The primary correctness measure is the 3-pixel geometric
inlier rate.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

STAGE_4_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TASK_02_DIR = os.path.join(
    STAGE_4_DIR,
    "results",
    "task_02"
)

TASK_03_DIR = os.path.join(
    STAGE_4_DIR,
    "results",
    "task_03"
)

RESULTS_DIR = os.path.join(
    STAGE_4_DIR,
    "results",
    "task_04"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

MATCHES_PATH = os.path.join(
    TASK_02_DIR,
    "matches.npz"
)

ERRORS_PATH = os.path.join(
    TASK_03_DIR,
    "geometric_errors.csv"
)

CONFIDENCE_ERROR_PATH = os.path.join(
    TASK_03_DIR,
    "confidence_vs_error.csv"
)


# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

CONFIDENCE_THRESHOLDS = np.array(
    [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55
    ],
    dtype=np.float64
)


# ============================================================
# EVALUATION THRESHOLDS
# ============================================================

GEOMETRIC_THRESHOLDS = [
    1.0,
    3.0,
    5.0,
    10.0
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(
        MATCHES_PATH
    ):
        raise FileNotFoundError(
            f"Task 2 matches not found:\n"
            f"{MATCHES_PATH}"
        )

    if not os.path.exists(
        CONFIDENCE_ERROR_PATH
    ):
        raise FileNotFoundError(
            f"Task 3 confidence/error data not found:\n"
            f"{CONFIDENCE_ERROR_PATH}"
        )

    matches_data = np.load(
        MATCHES_PATH
    )

    mconf_from_matches = (
        matches_data["mconf"]
        .astype(np.float64)
    )

    confidence_error = np.loadtxt(
        CONFIDENCE_ERROR_PATH,
        delimiter=",",
        skiprows=1
    )

    confidence = confidence_error[:, 1]
    errors = confidence_error[:, 2]

    if len(confidence) != len(errors):
        raise RuntimeError(
            "Confidence and error arrays have different lengths."
        )

    if len(confidence) != len(mconf_from_matches):
        raise RuntimeError(
            "Task 2 and Task 3 data contain different "
            "numbers of matches."
        )

    return confidence, errors


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_threshold_metrics(
    confidence,
    errors,
    threshold
):

    total_matches = len(
        confidence
    )

    selected = (
        confidence >= threshold
    )

    selected_confidence = confidence[
        selected
    ]

    selected_errors = errors[
        selected
    ]

    retained_matches = len(
        selected_errors
    )

    if retained_matches == 0:

        return {
            "threshold": threshold,
            "retained_matches": 0,
            "retained_percentage": 0.0,
            "mean_confidence": np.nan,
            "mean_error": np.nan,
            "median_error": np.nan,
            "within_1px": 0,
            "accuracy_1px": np.nan,
            "within_3px": 0,
            "accuracy_3px": np.nan,
            "within_5px": 0,
            "accuracy_5px": np.nan,
            "within_10px": 0,
            "accuracy_10px": np.nan
        }

    result = {
        "threshold": threshold,
        "retained_matches": retained_matches,
        "retained_percentage":
            100.0 * retained_matches / total_matches,
        "mean_confidence":
            float(np.mean(selected_confidence)),
        "mean_error":
            float(np.mean(selected_errors)),
        "median_error":
            float(np.median(selected_errors))
    }

    for geometric_threshold in GEOMETRIC_THRESHOLDS:

        count = int(
            np.sum(
                selected_errors
                <= geometric_threshold
            )
        )

        accuracy = (
            100.0
            * count
            / retained_matches
        )

        result[
            f"within_{geometric_threshold:g}px"
        ] = count

        result[
            f"accuracy_{geometric_threshold:g}px"
        ] = accuracy

    return result


# ============================================================
# SAVE CSV
# ============================================================

def save_results(
    results,
    output_path
):

    header = [
        "confidence_threshold",
        "retained_matches",
        "retained_percentage",
        "mean_confidence",
        "mean_error_px",
        "median_error_px",
        "within_1px",
        "accuracy_1px",
        "within_3px",
        "accuracy_3px",
        "within_5px",
        "accuracy_5px",
        "within_10px",
        "accuracy_10px"
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            ",".join(header) + "\n"
        )

        for result in results:

            values = [
                result["threshold"],
                result["retained_matches"],
                result["retained_percentage"],
                result["mean_confidence"],
                result["mean_error"],
                result["median_error"],
                result["within_1px"],
                result["accuracy_1px"],
                result["within_3px"],
                result["accuracy_3px"],
                result["within_5px"],
                result["accuracy_5px"],
                result["within_10px"],
                result["accuracy_10px"]
            ]

            f.write(
                ",".join(
                    str(value)
                    for value in values
                )
                + "\n"
            )


# ============================================================
# PLOT 1
# CONFIDENCE VS GEOMETRIC ERROR
# ============================================================

def plot_confidence_vs_error(
    confidence,
    errors
):

    output_path = os.path.join(
        RESULTS_DIR,
        "confidence_vs_geometric_error.png"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        confidence,
        errors,
        s=25,
        alpha=0.7
    )

    plt.axhline(
        3.0,
        linestyle="--",
        label="3 px threshold"
    )

    plt.xlabel(
        "LoFTR Confidence"
    )

    plt.ylabel(
        "Geometric Error (pixels)"
    )

    plt.title(
        "LoFTR Confidence vs Geometric Error"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT 2
# RETAINED MATCHES VS CONFIDENCE
# ============================================================

def plot_retained_matches(
    results
):

    thresholds = np.array(
        [
            r["threshold"]
            for r in results
        ]
    )

    retained = np.array(
        [
            r["retained_matches"]
            for r in results
        ]
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "confidence_vs_match_count.png"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        thresholds,
        retained,
        marker="o"
    )

    plt.xlabel(
        "Confidence Threshold"
    )

    plt.ylabel(
        "Retained Matches"
    )

    plt.title(
        "Confidence Threshold vs Number of Matches"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT 3
# CONFIDENCE VS GEOMETRIC INLIER RATE
# ============================================================

def plot_inlier_rate(
    results
):

    thresholds = np.array(
        [
            r["threshold"]
            for r in results
        ]
    )

    accuracy_3px = np.array(
        [
            r["accuracy_3px"]
            for r in results
        ]
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "confidence_vs_3px_inlier_rate.png"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        thresholds,
        accuracy_3px,
        marker="o"
    )

    plt.xlabel(
        "Confidence Threshold"
    )

    plt.ylabel(
        "3-pixel Geometric Inlier Rate (%)"
    )

    plt.title(
        "Confidence Threshold vs 3-pixel Inlier Rate"
    )

    plt.ylim(
        0,
        105
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT 4
# CONFIDENCE VS MEAN ERROR
# ============================================================

def plot_mean_error(
    results
):

    thresholds = np.array(
        [
            r["threshold"]
            for r in results
        ]
    )

    mean_error = np.array(
        [
            r["mean_error"]
            for r in results
        ]
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "confidence_vs_mean_error.png"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        thresholds,
        mean_error,
        marker="o"
    )

    plt.xlabel(
        "Confidence Threshold"
    )

    plt.ylabel(
        "Mean Geometric Error (pixels)"
    )

    plt.title(
        "Confidence Threshold vs Mean Geometric Error"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# SUMMARY REPORT
# ============================================================

def save_summary(
    results,
    confidence,
    errors
):

    output_path = os.path.join(
        RESULTS_DIR,
        "confidence_analysis_summary.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 4 - Task 4\n"
        )

        f.write(
            "LoFTR Confidence Threshold Analysis\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Total LoFTR matches: "
            f"{len(confidence)}\n"
        )

        f.write(
            f"Minimum confidence: "
            f"{np.min(confidence):.6f}\n"
        )

        f.write(
            f"Maximum confidence: "
            f"{np.max(confidence):.6f}\n"
        )

        f.write(
            f"Mean confidence: "
            f"{np.mean(confidence):.6f}\n"
        )

        f.write(
            f"Median confidence: "
            f"{np.median(confidence):.6f}\n\n"
        )

        f.write(
            "Threshold analysis\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        for result in results:

            f.write(
                f"\nConfidence threshold: "
                f"{result['threshold']:.2f}\n"
            )

            f.write(
                f"Retained matches: "
                f"{result['retained_matches']}\n"
            )

            f.write(
                f"Retained percentage: "
                f"{result['retained_percentage']:.2f}%\n"
            )

            f.write(
                f"Mean confidence: "
                f"{result['mean_confidence']:.6f}\n"
            )

            f.write(
                f"Mean geometric error: "
                f"{result['mean_error']:.4f} px\n"
            )

            f.write(
                f"Median geometric error: "
                f"{result['median_error']:.4f} px\n"
            )

            f.write(
                f"Within 1 px: "
                f"{result['within_1px']}/"
                f"{result['retained_matches']} "
                f"({result['accuracy_1px']:.2f}%)\n"
            )

            f.write(
                f"Within 3 px: "
                f"{result['within_3px']}/"
                f"{result['retained_matches']} "
                f"({result['accuracy_3px']:.2f}%)\n"
            )

            f.write(
                f"Within 5 px: "
                f"{result['within_5px']}/"
                f"{result['retained_matches']} "
                f"({result['accuracy_5px']:.2f}%)\n"
            )

            f.write(
                f"Within 10 px: "
                f"{result['within_10px']}/"
                f"{result['retained_matches']} "
                f"({result['accuracy_10px']:.2f}%)\n"
            )

        f.write(
            "\n\nInterpretation\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            "Increasing the LoFTR confidence threshold removes "
            "lower-confidence correspondences. The resulting "
            "change in geometric error and geometric inlier rate "
            "shows how useful the model confidence is as a "
            "filter for correspondence quality.\n\n"
        )

        f.write(
            "The 3-pixel geometric inlier rate is used as the "
            "primary correctness measure. It represents the "
            "fraction of retained LoFTR matches that are "
            "consistent with the HPatches ground-truth "
            "homography within 3 pixels.\n\n"
        )

        f.write(
            "A higher confidence threshold may improve geometric "
            "precision while reducing the number of available "
            "correspondences. Therefore, confidence filtering "
            "should be interpreted as a quality-versus-quantity "
            "trade-off rather than as an unconditional improvement.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Stage 4 - Task 4")
    print("LoFTR Confidence Threshold Analysis")
    print("=" * 60)

    confidence, errors = load_data()

    print()
    print(
        "Total matches:",
        len(confidence)
    )

    print(
        f"Confidence range: "
        f"{np.min(confidence):.6f} - "
        f"{np.max(confidence):.6f}"
    )

    print()
    print("=" * 60)
    print("CONFIDENCE THRESHOLD RESULTS")
    print("=" * 60)

    results = []

    for threshold in CONFIDENCE_THRESHOLDS:

        result = calculate_threshold_metrics(
            confidence,
            errors,
            threshold
        )

        results.append(
            result
        )

        print()
        print(
            f"Threshold: {threshold:.2f}"
        )

        print(
            f"Retained matches: "
            f"{result['retained_matches']}"
        )

        print(
            f"Retained: "
            f"{result['retained_percentage']:.2f}%"
        )

        print(
            f"Mean error: "
            f"{result['mean_error']:.4f} px"
        )

        print(
            f"Median error: "
            f"{result['median_error']:.4f} px"
        )

        print(
            f"≤ 3 px: "
            f"{result['within_3px']}/"
            f"{result['retained_matches']} "
            f"({result['accuracy_3px']:.2f}%)"
        )

        print(
            f"≤ 5 px: "
            f"{result['within_5px']}/"
            f"{result['retained_matches']} "
            f"({result['accuracy_5px']:.2f}%)"
        )

    # --------------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        "confidence_threshold_results.csv"
    )

    save_results(
        results,
        csv_path
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    plot_confidence_vs_error(
        confidence,
        errors
    )

    plot_retained_matches(
        results
    )

    plot_inlier_rate(
        results
    )

    plot_mean_error(
        results
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    save_summary(
        results,
        confidence,
        errors
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Task 4 completed.")
    print("=" * 60)

    print()
    print("Results directory:")
    print(RESULTS_DIR)

    print()
    print("Generated files:")

    print(
        "  - confidence_threshold_results.csv"
    )

    print(
        "  - confidence_analysis_summary.txt"
    )

    print(
        "  - confidence_vs_geometric_error.png"
    )

    print(
        "  - confidence_vs_match_count.png"
    )

    print(
        "  - confidence_vs_3px_inlier_rate.png"
    )

    print(
        "  - confidence_vs_mean_error.png"
    )


if __name__ == "__main__":
    main()