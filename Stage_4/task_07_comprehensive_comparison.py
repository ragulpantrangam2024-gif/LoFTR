import os
import csv
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

STAGE_1_RESULTS = os.path.join(
    PROJECT_ROOT,
    "Stage_1",
    "results"
)

STAGE_3_RESULTS = os.path.join(
    PROJECT_ROOT,
    "Stage_3",
    "results"
)

STAGE_4_RESULTS = os.path.join(
    PROJECT_ROOT,
    "Stage_4",
    "results"
)

TASK_07_RESULTS = os.path.join(
    STAGE_4_RESULTS,
    "task_07"
)

os.makedirs(
    TASK_07_RESULTS,
    exist_ok=True
)


# ============================================================
# EXISTING RESULTS
# ============================================================

# Stage 1 - SIFT
SIFT_MEAN_ERROR = 17.6902
SIFT_MEDIAN_ERROR = 0.3596
SIFT_MATCHES = 513
SIFT_WITHIN_3 = 481
SIFT_WITHIN_3_RATE = 93.76


# Stage 1 - ORB
ORB_MEAN_ERROR = 22.5343
ORB_MEDIAN_ERROR = 0.9430
ORB_MATCHES = 437
ORB_WITHIN_3 = 393
ORB_WITHIN_3_RATE = 89.93


# Stage 3 - Simplified LoFTR
# Task 7 quantitative evaluation at threshold 0.80
SIMPLIFIED_LOFTR_MATCHES = 45
SIMPLIFIED_LOFTR_MEAN_ERROR = 24.0010
SIMPLIFIED_LOFTR_MEDIAN_ERROR = 17.2945
SIMPLIFIED_LOFTR_WITHIN_3 = None
SIMPLIFIED_LOFTR_WITHIN_3_RATE = None
SIMPLIFIED_LOFTR_WITHIN_5 = None
SIMPLIFIED_LOFTR_WITHIN_10 = 3
SIMPLIFIED_LOFTR_WITHIN_10_RATE = 6.67


# Stage 4 - Official LoFTR
# Task 3 single-pair evaluation
OFFICIAL_LOFTR_MATCHES = 206
OFFICIAL_LOFTR_MEAN_ERROR = 2.5279
OFFICIAL_LOFTR_MEDIAN_ERROR = 2.2551
OFFICIAL_LOFTR_WITHIN_3 = 140
OFFICIAL_LOFTR_WITHIN_3_RATE = 67.96
OFFICIAL_LOFTR_WITHIN_5 = 196
OFFICIAL_LOFTR_WITHIN_5_RATE = 95.15
OFFICIAL_LOFTR_WITHIN_10 = 205
OFFICIAL_LOFTR_WITHIN_10_RATE = 99.51


# ============================================================
# EXPERIMENT CONTEXT
# ============================================================

comparison_data = [
    {
        "method": "SIFT",
        "stage": "Stage 1",
        "sequence": "v_woman",
        "experiment": "Single-pair",
        "matches": SIFT_MATCHES,
        "mean_error": SIFT_MEAN_ERROR,
        "median_error": SIFT_MEDIAN_ERROR,
        "within_3": SIFT_WITHIN_3_RATE,
        "within_5": None,
        "within_10": None,
    },
    {
        "method": "ORB",
        "stage": "Stage 1",
        "sequence": "v_woman",
        "experiment": "Single-pair",
        "matches": ORB_MATCHES,
        "mean_error": ORB_MEAN_ERROR,
        "median_error": ORB_MEDIAN_ERROR,
        "within_3": ORB_WITHIN_3_RATE,
        "within_5": None,
        "within_10": None,
    },
    {
        "method": "Simplified LoFTR",
        "stage": "Stage 3",
        "sequence": "v_soldiers",
        "experiment": "Threshold 0.80",
        "matches": SIMPLIFIED_LOFTR_MATCHES,
        "mean_error": SIMPLIFIED_LOFTR_MEAN_ERROR,
        "median_error": SIMPLIFIED_LOFTR_MEDIAN_ERROR,
        "within_3": None,
        "within_5": None,
        "within_10": SIMPLIFIED_LOFTR_WITHIN_10_RATE,
    },
    {
        "method": "Official LoFTR",
        "stage": "Stage 4",
        "sequence": "v_soldiers",
        "experiment": "Single-pair",
        "matches": OFFICIAL_LOFTR_MATCHES,
        "mean_error": OFFICIAL_LOFTR_MEAN_ERROR,
        "median_error": OFFICIAL_LOFTR_MEDIAN_ERROR,
        "within_3": OFFICIAL_LOFTR_WITHIN_3_RATE,
        "within_5": OFFICIAL_LOFTR_WITHIN_5_RATE,
        "within_10": OFFICIAL_LOFTR_WITHIN_10_RATE,
    },
]


# ============================================================
# NOTE
# ============================================================

COMPARISON_NOTE = """
Important experimental limitation:

The four methods were not all evaluated on exactly the same
image pair and protocol.

SIFT and ORB were evaluated on v_woman during Stage 1.

The simplified LoFTR was evaluated on v_soldiers during Stage 3.

The official LoFTR single-pair evaluation was performed on
v_soldiers during Stage 4.

Therefore, the numerical values should not be interpreted
as a controlled benchmark ranking of all four methods.

The comparison is intended to summarize the behavior observed
throughout the project and identify methodological differences.
"""


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

def save_comparison_csv():

    output_path = os.path.join(
        TASK_07_RESULTS,
        "method_comparison.csv"
    )

    fieldnames = [
        "method",
        "stage",
        "sequence",
        "experiment",
        "matches",
        "mean_error",
        "median_error",
        "within_3_percent",
        "within_5_percent",
        "within_10_percent",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in comparison_data:

            writer.writerow(
                {
                    "method": row["method"],
                    "stage": row["stage"],
                    "sequence": row["sequence"],
                    "experiment": row["experiment"],
                    "matches": row["matches"],
                    "mean_error": row["mean_error"],
                    "median_error": row["median_error"],
                    "within_3_percent": row["within_3"],
                    "within_5_percent": row["within_5"],
                    "within_10_percent": row["within_10"],
                }
            )

    return output_path


# ============================================================
# MATCH COUNT PLOT
# ============================================================

def plot_match_counts():

    methods = [
        row["method"]
        for row in comparison_data
    ]

    values = [
        row["matches"]
        for row in comparison_data
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        methods,
        values
    )

    plt.ylabel(
        "Number of matches"
    )

    plt.xlabel(
        "Method"
    )

    plt.title(
        "Observed Match Count by Method"
    )

    plt.xticks(
        rotation=20
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        TASK_07_RESULTS,
        "match_count_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# MEAN ERROR PLOT
# ============================================================

def plot_mean_error():

    methods = [
        row["method"]
        for row in comparison_data
    ]

    values = [
        row["mean_error"]
        for row in comparison_data
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        methods,
        values
    )

    plt.ylabel(
        "Mean geometric error (pixels)"
    )

    plt.xlabel(
        "Method"
    )

    plt.title(
        "Observed Mean Geometric Error"
    )

    plt.xticks(
        rotation=20
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        TASK_07_RESULTS,
        "mean_error_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# MEDIAN ERROR PLOT
# ============================================================

def plot_median_error():

    methods = [
        row["method"]
        for row in comparison_data
    ]

    values = [
        row["median_error"]
        for row in comparison_data
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        methods,
        values
    )

    plt.ylabel(
        "Median geometric error (pixels)"
    )

    plt.xlabel(
        "Method"
    )

    plt.title(
        "Observed Median Geometric Error"
    )

    plt.xticks(
        rotation=20
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        TASK_07_RESULTS,
        "median_error_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# 3-PIXEL ACCURACY PLOT
# ============================================================

def plot_three_pixel_accuracy():

    methods = []
    values = []

    for row in comparison_data:

        if row["within_3"] is not None:

            methods.append(
                row["method"]
            )

            values.append(
                row["within_3"]
            )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        methods,
        values
    )

    plt.ylabel(
        "Matches within 3 px (%)"
    )

    plt.xlabel(
        "Method"
    )

    plt.title(
        "Observed 3-Pixel Geometric Accuracy"
    )

    plt.ylim(
        0,
        105
    )

    plt.xticks(
        rotation=20
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        TASK_07_RESULTS,
        "three_pixel_accuracy_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# WRITE REPORT
# ============================================================

def save_report():

    output_path = os.path.join(
        TASK_07_RESULTS,
        "comparison_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 4 - Task 7\n"
        )

        f.write(
            "Comprehensive Feature Matching Comparison\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "METHODS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "1. SIFT\n"
        )

        f.write(
            "2. ORB\n"
        )

        f.write(
            "3. Simplified LoFTR\n"
        )

        f.write(
            "4. Official pretrained LoFTR\n\n"
        )

        f.write(
            "EXPERIMENTAL RESULTS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for row in comparison_data:

            f.write(
                f"\n{row['method']}\n"
            )

            f.write(
                f"Stage: {row['stage']}\n"
            )

            f.write(
                f"Sequence: {row['sequence']}\n"
            )

            f.write(
                f"Experiment: {row['experiment']}\n"
            )

            f.write(
                f"Matches: {row['matches']}\n"
            )

            f.write(
                f"Mean error: "
                f"{row['mean_error']:.4f} px\n"
            )

            f.write(
                f"Median error: "
                f"{row['median_error']:.4f} px\n"
            )

            if row["within_3"] is not None:

                f.write(
                    f"≤3 px: "
                    f"{row['within_3']:.2f}%\n"
                )

            if row["within_5"] is not None:

                f.write(
                    f"≤5 px: "
                    f"{row['within_5']:.2f}%\n"
                )

            if row["within_10"] is not None:

                f.write(
                    f"≤10 px: "
                    f"{row['within_10']:.2f}%\n"
                )

        f.write(
            "\n\nMETHODOLOGICAL COMPARISON\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "SIFT\n"
        )

        f.write(
            "Classical detector-and-descriptor approach. "
            "Keypoints are detected independently and descriptors "
            "are matched between images.\n\n"
        )

        f.write(
            "ORB\n"
        )

        f.write(
            "Classical binary feature approach using FAST-based "
            "keypoint detection and BRIEF-derived binary descriptors.\n\n"
        )

        f.write(
            "Simplified LoFTR\n"
        )

        f.write(
            "Research-oriented simplified implementation developed "
            "during Stage 3. It reproduces selected concepts such "
            "as dense features, Transformer matching, coarse "
            "matching and fine refinement, but it is not equivalent "
            "to the official trained LoFTR model.\n\n"
        )

        f.write(
            "Official pretrained LoFTR\n"
        )

        f.write(
            "Detector-free Transformer-based feature matching using "
            "the official pretrained LoFTR implementation and "
            "checkpoint.\n\n"
        )

        f.write(
            COMPARISON_NOTE
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "Stage 4 - Task 7: "
        "Comprehensive Feature Matching Comparison"
    )
    print("=" * 70)

    print(
        "\nGenerating comparison data..."
    )

    csv_path = save_comparison_csv()

    print(
        f"Saved: {csv_path}"
    )

    match_plot = plot_match_counts()

    print(
        f"Saved: {match_plot}"
    )

    mean_plot = plot_mean_error()

    print(
        f"Saved: {mean_plot}"
    )

    median_plot = plot_median_error()

    print(
        f"Saved: {median_plot}"
    )

    three_pixel_plot = plot_three_pixel_accuracy()

    print(
        f"Saved: {three_pixel_plot}"
    )

    report_path = save_report()

    print(
        f"Saved: {report_path}"
    )

    print("\n")
    print("=" * 70)
    print("Task 7 completed.")
    print("=" * 70)

    print(
        f"\nResults directory:"
    )

    print(
        TASK_07_RESULTS
    )


if __name__ == "__main__":
    main()