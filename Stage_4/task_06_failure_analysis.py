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

TASK_05_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_05"
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "task_06"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_CONFIDENCE_THRESHOLD = 0.30
GEOMETRIC_ERROR_THRESHOLD = 3.0
TOP_FAILURES = 20


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
    [("viewpoint", s) for s in VIEWPOINT_SEQUENCES]
    +
    [("illumination", s) for s in ILLUMINATION_SEQUENCES]
)


# ============================================================
# LOAD SEQUENCE DATA
# ============================================================

def load_sequence(category, sequence):

    sequence_dir = os.path.join(
        TASK_05_DIR,
        sequence
    )

    npz_path = os.path.join(
        sequence_dir,
        "matches_and_errors.npz"
    )

    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)

    return {
        "category": category,
        "sequence": sequence,
        "mkpts0": data["mkpts0"],
        "mkpts1": data["mkpts1"],
        "confidence": data["confidence"],
        "errors": data["errors"],
    }


# ============================================================
# PERCENTILE SAFE FUNCTION
# ============================================================

def percentile(values, percentile):

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            percentile
        )
    )


# ============================================================
# SEQUENCE STATISTICS
# ============================================================

def calculate_sequence_statistics(data):

    errors = data["errors"]
    confidence = data["confidence"]

    n = len(errors)

    if n == 0:
        return None

    return {
        "category": data["category"],
        "sequence": data["sequence"],
        "matches": n,

        "mean_error": float(
            np.mean(errors)
        ),

        "median_error": float(
            np.median(errors)
        ),

        "p90_error": percentile(
            errors,
            90
        ),

        "p95_error": percentile(
            errors,
            95
        ),

        "max_error": float(
            np.max(errors)
        ),

        "within_1_rate": float(
            np.mean(errors <= 1.0)
        ),

        "within_3_rate": float(
            np.mean(errors <= 3.0)
        ),

        "within_5_rate": float(
            np.mean(errors <= 5.0)
        ),

        "within_10_rate": float(
            np.mean(errors <= 10.0)
        ),

        "mean_confidence": float(
            np.mean(confidence)
        ),

        "median_confidence": float(
            np.median(confidence)
        ),

        "high_confidence_count": int(
            np.sum(
                confidence >= HIGH_CONFIDENCE_THRESHOLD
            )
        ),

        "high_confidence_failures": int(
            np.sum(
                (confidence >= HIGH_CONFIDENCE_THRESHOLD)
                &
                (errors > GEOMETRIC_ERROR_THRESHOLD)
            )
        ),
    }


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_all_sequences():

    datasets = []
    statistics = []

    for category, sequence in ALL_SEQUENCES:

        data = load_sequence(
            category,
            sequence
        )

        if data is None:
            continue

        datasets.append(data)

        stats = calculate_sequence_statistics(
            data
        )

        if stats is not None:
            statistics.append(stats)

    return datasets, statistics


# ============================================================
# SAVE SEQUENCE SUMMARY
# ============================================================

def save_sequence_summary(statistics):

    output_path = os.path.join(
        RESULTS_DIR,
        "failure_analysis_summary.csv"
    )

    fieldnames = [
        "category",
        "sequence",
        "matches",
        "mean_error",
        "median_error",
        "p90_error",
        "p95_error",
        "max_error",
        "within_1_rate",
        "within_3_rate",
        "within_5_rate",
        "within_10_rate",
        "mean_confidence",
        "median_confidence",
        "high_confidence_count",
        "high_confidence_failures",
    ]

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for stats in statistics:
            writer.writerow(stats)

    return output_path


# ============================================================
# SAVE TOP FAILURE MATCHES
# ============================================================

def save_top_failures(datasets):

    failures = []

    for data in datasets:

        errors = data["errors"]
        confidence = data["confidence"]
        mkpts0 = data["mkpts0"]
        mkpts1 = data["mkpts1"]

        for i in range(len(errors)):

            failures.append(
                {
                    "category": data["category"],
                    "sequence": data["sequence"],
                    "match_index": i,
                    "error": float(errors[i]),
                    "confidence": float(confidence[i]),
                    "x0": float(mkpts0[i, 0]),
                    "y0": float(mkpts0[i, 1]),
                    "x1": float(mkpts1[i, 0]),
                    "y1": float(mkpts1[i, 1]),
                }
            )

    failures.sort(
        key=lambda x: x["error"],
        reverse=True
    )

    top_failures = failures[:TOP_FAILURES]

    output_path = os.path.join(
        RESULTS_DIR,
        "top_failure_matches.csv"
    )

    fieldnames = [
        "category",
        "sequence",
        "match_index",
        "error",
        "confidence",
        "x0",
        "y0",
        "x1",
        "y1",
    ]

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for failure in top_failures:
            writer.writerow(failure)

    return output_path


# ============================================================
# CONFIDENCE VS ERROR
# ============================================================

def confidence_vs_error_plot(datasets):

    plt.figure(
        figsize=(10, 7)
    )

    for data in datasets:

        confidence = data["confidence"]
        errors = data["errors"]

        plt.scatter(
            confidence,
            errors,
            s=12,
            alpha=0.6,
            label=data["sequence"]
        )

    plt.axhline(
        GEOMETRIC_ERROR_THRESHOLD,
        linestyle="--",
        linewidth=1
    )

    plt.axvline(
        HIGH_CONFIDENCE_THRESHOLD,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "LoFTR confidence"
    )

    plt.ylabel(
        "Geometric error (pixels)"
    )

    plt.title(
        "LoFTR Confidence vs Geometric Error"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "confidence_vs_geometric_error.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

def error_distribution_plot(datasets):

    all_errors = []

    labels = []

    for data in datasets:

        all_errors.append(
            data["errors"]
        )

        labels.append(
            data["sequence"]
        )

    plt.figure(
        figsize=(12, 7)
    )

    plt.boxplot(
        all_errors,
        labels=labels,
        showfliers=True
    )

    plt.axhline(
        GEOMETRIC_ERROR_THRESHOLD,
        linestyle="--",
        linewidth=1
    )

    plt.ylabel(
        "Geometric error (pixels)"
    )

    plt.xlabel(
        "HPatches sequence"
    )

    plt.title(
        "Geometric Error Distribution Across Sequences"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "error_distribution_by_sequence.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# MATCH COUNT PLOT
# ============================================================

def match_count_plot(statistics):

    labels = [
        s["sequence"]
        for s in statistics
    ]

    counts = [
        s["matches"]
        for s in statistics
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        labels,
        counts
    )

    plt.ylabel(
        "Number of matches"
    )

    plt.xlabel(
        "HPatches sequence"
    )

    plt.title(
        "LoFTR Match Count Across Evaluated Sequences"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "match_count_by_sequence.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# 3-PIXEL INLIER RATE
# ============================================================

def three_pixel_rate_plot(statistics):

    labels = [
        s["sequence"]
        for s in statistics
    ]

    rates = [
        s["within_3_rate"] * 100
        for s in statistics
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        labels,
        rates
    )

    plt.axhline(
        50,
        linestyle="--",
        linewidth=1
    )

    plt.ylabel(
        "Matches within 3 px (%)"
    )

    plt.xlabel(
        "HPatches sequence"
    )

    plt.title(
        "3-Pixel Geometric Inlier Rate"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylim(
        0,
        105
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "three_pixel_inlier_rate.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


# ============================================================
# TEXT REPORT
# ============================================================

def save_text_report(
    statistics,
    datasets
):

    output_path = os.path.join(
        RESULTS_DIR,
        "failure_analysis_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Stage 4 - Task 6\n"
        )

        f.write(
            "LoFTR Failure Analysis and Confidence-Geometry Analysis\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"High-confidence threshold: "
            f"{HIGH_CONFIDENCE_THRESHOLD}\n"
        )

        f.write(
            f"Geometric failure threshold: "
            f"{GEOMETRIC_ERROR_THRESHOLD} px\n\n"
        )

        # ----------------------------------------------------
        # Sequence statistics
        # ----------------------------------------------------

        f.write(
            "SEQUENCE STATISTICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for stats in statistics:

            f.write(
                f"\n{stats['sequence']} "
                f"({stats['category']})\n"
            )

            f.write(
                f"Matches: "
                f"{stats['matches']}\n"
            )

            f.write(
                f"Mean error: "
                f"{stats['mean_error']:.4f} px\n"
            )

            f.write(
                f"Median error: "
                f"{stats['median_error']:.4f} px\n"
            )

            f.write(
                f"P90 error: "
                f"{stats['p90_error']:.4f} px\n"
            )

            f.write(
                f"P95 error: "
                f"{stats['p95_error']:.4f} px\n"
            )

            f.write(
                f"Maximum error: "
                f"{stats['max_error']:.4f} px\n"
            )

            f.write(
                f"≤1 px: "
                f"{stats['within_1_rate'] * 100:.2f}%\n"
            )

            f.write(
                f"≤3 px: "
                f"{stats['within_3_rate'] * 100:.2f}%\n"
            )

            f.write(
                f"≤5 px: "
                f"{stats['within_5_rate'] * 100:.2f}%\n"
            )

            f.write(
                f"≤10 px: "
                f"{stats['within_10_rate'] * 100:.2f}%\n"
            )

            f.write(
                f"Mean confidence: "
                f"{stats['mean_confidence']:.4f}\n"
            )

            f.write(
                f"Median confidence: "
                f"{stats['median_confidence']:.4f}\n"
            )

            f.write(
                f"High-confidence matches: "
                f"{stats['high_confidence_count']}\n"
            )

            f.write(
                f"High-confidence failures: "
                f"{stats['high_confidence_failures']}\n"
            )

        # ----------------------------------------------------
        # Overall analysis
        # ----------------------------------------------------

        all_errors = []
        all_confidences = []

        for data in datasets:

            all_errors.extend(
                data["errors"].tolist()
            )

            all_confidences.extend(
                data["confidence"].tolist()
            )

        all_errors = np.asarray(
            all_errors
        )

        all_confidences = np.asarray(
            all_confidences
        )

        if len(all_errors) > 0:

            f.write(
                "\n\nOVERALL MATCH ANALYSIS\n"
            )

            f.write(
                "=" * 70 + "\n"
            )

            f.write(
                f"Total evaluated matches: "
                f"{len(all_errors)}\n"
            )

            f.write(
                f"Mean geometric error: "
                f"{np.mean(all_errors):.4f} px\n"
            )

            f.write(
                f"Median geometric error: "
                f"{np.median(all_errors):.4f} px\n"
            )

            f.write(
                f"P90 geometric error: "
                f"{np.percentile(all_errors, 90):.4f} px\n"
            )

            f.write(
                f"P95 geometric error: "
                f"{np.percentile(all_errors, 95):.4f} px\n"
            )

            f.write(
                f"Overall ≤3 px: "
                f"{np.mean(all_errors <= 3) * 100:.2f}%\n"
            )

            f.write(
                f"Overall ≤5 px: "
                f"{np.mean(all_errors <= 5) * 100:.2f}%\n"
            )

            f.write(
                f"Overall ≤10 px: "
                f"{np.mean(all_errors <= 10) * 100:.2f}%\n"
            )

        # ----------------------------------------------------
        # High confidence failures
        # ----------------------------------------------------

        high_conf_failures = np.sum(
            (all_confidences >= HIGH_CONFIDENCE_THRESHOLD)
            &
            (all_errors > GEOMETRIC_ERROR_THRESHOLD)
        )

        high_conf_matches = np.sum(
            all_confidences >= HIGH_CONFIDENCE_THRESHOLD
        )

        f.write(
            "\n\nHIGH-CONFIDENCE FAILURE ANALYSIS\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            f"High-confidence matches: "
            f"{high_conf_matches}\n"
        )

        f.write(
            f"High-confidence matches with error > "
            f"{GEOMETRIC_ERROR_THRESHOLD} px: "
            f"{high_conf_failures}\n"
        )

        if high_conf_matches > 0:

            f.write(
                f"High-confidence failure rate: "
                f"{high_conf_failures / high_conf_matches * 100:.2f}%\n"
            )

        # ----------------------------------------------------
        # No-match sequences
        # ----------------------------------------------------

        no_match_sequences = []

        for category, sequence in ALL_SEQUENCES:

            sequence_dir = os.path.join(
                TASK_05_DIR,
                sequence
            )

            npz_path = os.path.join(
                sequence_dir,
                "matches_and_errors.npz"
            )

            if not os.path.exists(npz_path):

                no_match_sequences.append(
                    sequence
                )

        f.write(
            "\n\nNO-MATCH SEQUENCES\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        if no_match_sequences:

            for sequence in no_match_sequences:

                f.write(
                    f"{sequence}\n"
                )

        else:

            f.write(
                "No missing sequence result files detected.\n"
            )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "Stage 4 - Task 6: "
        "Failure Analysis"
    )
    print("=" * 70)

    print(
        f"Task 5 results: {TASK_05_DIR}"
    )

    print(
        f"Task 6 results: {RESULTS_DIR}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    datasets, statistics = load_all_sequences()

    print(
        f"\nLoaded successful sequences: "
        f"{len(datasets)}"
    )

    total_matches = sum(
        len(data["errors"])
        for data in datasets
    )

    print(
        f"Total evaluated matches: "
        f"{total_matches}"
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    summary_path = save_sequence_summary(
        statistics
    )

    print(
        f"\nSaved: {summary_path}"
    )

    # --------------------------------------------------------
    # Save worst matches
    # --------------------------------------------------------

    failures_path = save_top_failures(
        datasets
    )

    print(
        f"Saved: {failures_path}"
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    confidence_plot = confidence_vs_error_plot(
        datasets
    )

    print(
        f"Saved: {confidence_plot}"
    )

    error_plot = error_distribution_plot(
        datasets
    )

    print(
        f"Saved: {error_plot}"
    )

    match_plot = match_count_plot(
        statistics
    )

    print(
        f"Saved: {match_plot}"
    )

    three_px_plot = three_pixel_rate_plot(
        statistics
    )

    print(
        f"Saved: {three_px_plot}"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = save_text_report(
        statistics,
        datasets
    )

    print(
        f"Saved: {report_path}"
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Task 6 completed.")
    print("=" * 70)

    print(
        f"Successful sequence files analysed: "
        f"{len(datasets)}"
    )

    print(
        f"Total matches analysed: "
        f"{total_matches}"
    )

    print(
        f"\nResults directory:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()