import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

STAGE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    STAGE_DIR
)

RESULTS_DIR = os.path.join(
    STAGE_DIR,
    "results",
    "task_08"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_text(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


def parse_key_values(path):
    values = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if ":" not in line:
                continue

            key, value = line.split(
                ":",
                1
            )

            values[key.strip()] = value.strip()

    return values


def number_from_text(text):
    text = (
        text
        .replace("px", "")
        .replace("%", "")
        .strip()
    )

    return float(text)


def percentage_from_text(text):
    if "(" not in text:
        return np.nan

    value = text.split(
        "(",
        1
    )[1]

    value = value.replace(
        ")",
        ""
    )

    value = value.replace(
        "%",
        ""
    )

    return float(value)


def count_from_text(text):
    if "/" not in text:
        return np.nan

    return int(
        text.split(
            "/",
            1
        )[0]
    )


# ============================================================
# LOAD TASK 7D CONTROLLED COMPARISON
# ============================================================

TASK_7D_CSV = os.path.join(
    STAGE_DIR,
    "results",
    "task_07d",
    "controlled_comparison.csv"
)

controlled_rows = []

with open(
    TASK_7D_CSV,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        controlled_rows.append(row)


# ============================================================
# LOAD TASK 6 FAILURE ANALYSIS
# ============================================================

TASK_6_SUMMARY = os.path.join(
    STAGE_DIR,
    "results",
    "task_06",
    "failure_analysis_summary.csv"
)

failure_rows = []

if os.path.exists(TASK_6_SUMMARY):

    with open(
        TASK_6_SUMMARY,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            failure_rows.append(row)


# ============================================================
# LOAD TASK 5 MULTI-SEQUENCE RESULTS
# ============================================================

TASK_5_CSV = os.path.join(
    STAGE_DIR,
    "results",
    "task_05",
    "multisequence_results.csv"
)

multisequence_rows = []

if os.path.exists(TASK_5_CSV):

    with open(
        TASK_5_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            multisequence_rows.append(row)


# ============================================================
# LOAD TASK 4 CONFIDENCE ANALYSIS
# ============================================================

TASK_4_CSV = os.path.join(
    STAGE_DIR,
    "results",
    "task_04",
    "confidence_threshold_results.csv"
)

confidence_rows = []

if os.path.exists(TASK_4_CSV):

    with open(
        TASK_4_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            confidence_rows.append(row)


# ============================================================
# CONTROLLED COMPARISON DATA
# ============================================================

methods = [
    row["method"]
    for row in controlled_rows
]

match_counts = [
    float(row["matches"])
    for row in controlled_rows
]

mean_errors = [
    float(row["mean_error_px"])
    for row in controlled_rows
]

median_errors = [
    float(row["median_error_px"])
    for row in controlled_rows
]

within_3px = [
    float(row["within_3px_percent"])
    if row["within_3px_percent"] != "nan"
    else np.nan
    for row in controlled_rows
]

within_5px = [
    float(row["within_5px_percent"])
    if row["within_5px_percent"] != "nan"
    else np.nan
    for row in controlled_rows
]

within_10px = [
    float(row["within_10px_percent"])
    if row["within_10px_percent"] != "nan"
    else np.nan
    for row in controlled_rows
]


# ============================================================
# SAVE FINAL CONTROLLED SUMMARY
# ============================================================

final_csv = os.path.join(
    RESULTS_DIR,
    "final_results_summary.csv"
)

with open(
    final_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "method",
        "matches",
        "mean_error_px",
        "median_error_px",
        "within_3px_percent",
        "within_5px_percent",
        "within_10px_percent"
    ])

    for i, method in enumerate(methods):

        writer.writerow([
            method,
            match_counts[i],
            mean_errors[i],
            median_errors[i],
            within_3px[i],
            within_5px[i],
            within_10px[i]
        ])


# ============================================================
# FIGURE 1 — CONTROLLED COMPARISON OVERVIEW
# ============================================================

plt.figure(
    figsize=(11, 7)
)

bars = plt.bar(
    methods,
    within_10px
)

for bar, value in zip(
    bars,
    within_10px
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.title(
    "Controlled Comparison: Matches Within 10 Pixels"
)

plt.ylabel(
    "Matches within 10 px (%)"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "controlled_comparison.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FIGURE 2 — OFFICIAL LOFTR MULTI-SEQUENCE
# ============================================================

sequence_names = []
sequence_mean_errors = []

for row in multisequence_rows:

    sequence = row.get(
        "sequence",
        ""
    )

    mean_error = row.get(
        "mean_error_px",
        ""
    )

    if sequence and mean_error:

        try:
            sequence_names.append(
                sequence
            )

            sequence_mean_errors.append(
                float(mean_error)
            )

        except ValueError:
            pass


if sequence_names:

    plt.figure(
        figsize=(12, 7)
    )

    bars = plt.bar(
        sequence_names,
        sequence_mean_errors
    )

    for bar, value in zip(
        bars,
        sequence_mean_errors
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.title(
        "Official LoFTR Mean Geometric Error Across HPatches Sequences"
    )

    plt.ylabel(
        "Mean geometric error (pixels)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "official_loftr_multisequence.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# FIGURE 3 — CONFIDENCE VS ERROR
# ============================================================

confidence_values = []
confidence_mean_errors = []
confidence_match_counts = []

for row in confidence_rows:

    try:

        confidence_values.append(
            float(
                row.get(
                    "threshold",
                    ""
                )
            )
        )

        confidence_mean_errors.append(
            float(
                row.get(
                    "mean_error_px",
                    ""
                )
            )
        )

        confidence_match_counts.append(
            float(
                row.get(
                    "matches",
                    ""
                )
            )
        )

    except ValueError:
        pass


if confidence_values:

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        confidence_values,
        confidence_mean_errors,
        marker="o"
    )

    plt.xlabel(
        "Confidence threshold"
    )

    plt.ylabel(
        "Mean geometric error (pixels)"
    )

    plt.title(
        "Official LoFTR Confidence Threshold vs Mean Error"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "confidence_failure_summary.png"
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# FIGURE 4 — MATCH COUNT OVERVIEW
# ============================================================

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    methods,
    match_counts
)

for bar, value in zip(
    bars,
    match_counts
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{int(value)}",
        ha="center",
        va="bottom"
    )

plt.title(
    "Controlled Comparison: Number of Evaluated Matches"
)

plt.ylabel(
    "Number of matches"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "match_count_overview.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# CALCULATE OVERALL OFFICIAL LOFTR STATISTICS
# FROM TASK 6 DATA IF AVAILABLE
# ============================================================

overall_matches = None
overall_mean = None
overall_median = None
overall_p90 = None
overall_p95 = None
overall_3px = None
overall_5px = None
overall_10px = None
high_conf_count = None
high_conf_failures = None
high_conf_failure_rate = None

failure_report_path = os.path.join(
    STAGE_DIR,
    "results",
    "task_06",
    "failure_analysis_report.txt"
)

if os.path.exists(failure_report_path):

    report_text = read_text(
        failure_report_path
    )

    for line in report_text.splitlines():

        line = line.strip()

        if line.startswith(
            "Total evaluated matches:"
        ):

            try:
                overall_matches = int(
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )
            except ValueError:
                pass

        elif line.startswith(
            "Overall mean error:"
        ):

            overall_mean = number_from_text(
                line.split(
                    ":",
                    1
                )[1]
            )

        elif line.startswith(
            "Overall median error:"
        ):

            overall_median = number_from_text(
                line.split(
                    ":",
                    1
                )[1]
            )

        elif line.startswith(
            "Overall P90 error:"
        ):

            overall_p90 = number_from_text(
                line.split(
                    ":",
                    1
                )[1]
            )

        elif line.startswith(
            "Overall P95 error:"
        ):

            overall_p95 = number_from_text(
                line.split(
                    ":",
                    1
                )[1]
            )


# ============================================================
# WRITE FINAL RESEARCH REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "final_research_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 4 TASK 8 - FINAL RESEARCH EVALUATION\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        "PROJECT: LoFTR — Deep Feature Matching from Classical Methods "
        "to Transformers\n\n"
    )

    f.write(
        "PURPOSE\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "This final evaluation consolidates the experimental work "
        "performed across Stages 1–4. The project investigates the "
        "progression from classical local feature matching toward "
        "detector-free Transformer-based feature matching, using "
        "LoFTR as the main research reference.\n\n"
    )

    f.write(
        "STAGE 1 — CLASSICAL FEATURE MATCHING\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Stage 1 established the fundamentals of image correspondence "
        "using SIFT, ORB, and geometric verification with RANSAC. "
        "The experiments introduced keypoints, descriptors, descriptor "
        "matching, Lowe's ratio test, homography-based geometric error, "
        "and inlier/outlier analysis.\n\n"
    )

    f.write(
        "STAGE 2 — TRANSFORMER FUNDAMENTALS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Stage 2 implemented and investigated self-attention, "
        "cross-attention, positional encoding, Transformer encoder "
        "blocks, and coarse-to-fine matching. These experiments were "
        "designed to understand the mechanisms that later appear in "
        "LoFTR rather than to reproduce a trained model.\n\n"
    )

    f.write(
        "STAGE 3 — SIMPLIFIED LOFTR\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Stage 3 integrated CNN feature extraction, dense feature "
        "representation, Transformer-based matching, coarse matching, "
        "and fine refinement into a simplified LoFTR pipeline. The "
        "quantitative experiments demonstrated that untrained feature "
        "representations are insufficient for reliable end-to-end "
        "matching. The simplified implementation therefore serves "
        "primarily as an architectural and educational study.\n\n"
    )

    f.write(
        "STAGE 4 — OFFICIAL LOFTR AND RESEARCH EVALUATION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Stage 4 used the official LoFTR implementation with a "
        "pretrained indoor checkpoint. The evaluation included a "
        "single-pair quantitative experiment, confidence threshold "
        "analysis, multi-sequence evaluation, failure analysis, and "
        "a controlled comparison against SIFT, ORB, and the simplified "
        "LoFTR implementation.\n\n"
    )

    f.write(
        "CONTROLLED FOUR-METHOD COMPARISON\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Dataset: HPatches\n"
    )

    f.write(
        "Sequence: v_soldiers\n"
    )

    f.write(
        "Image pair: 1.ppm -> 2.ppm\n"
    )

    f.write(
        "Ground truth: H_1_2\n\n"
    )

    f.write(
        "Method                  Matches   Mean(px)   Median(px)   "
        "<=3px     <=5px     <=10px\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for i, method in enumerate(methods):

        def fmt(value):
            if np.isnan(value):
                return "N/A"
            return f"{value:.2f}%"

        f.write(
            f"{method:<22}"
            f"{int(match_counts[i]):>8}"
            f"{mean_errors[i]:>11.4f}"
            f"{median_errors[i]:>13.4f}"
            f"{fmt(within_3px[i]):>11}"
            f"{fmt(within_5px[i]):>10}"
            f"{fmt(within_10px[i]):>11}\n"
        )

    f.write("\n")

    f.write(
        "The controlled comparison uses the same image pair and "
        "ground-truth homography for all four methods. The comparison "
        "should therefore be interpreted as a controlled single-pair "
        "experiment rather than a universal benchmark.\n\n"
    )

    f.write(
        "OFFICIAL LOFTR MULTI-SEQUENCE EVALUATION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "The broader Stage 4 evaluation successfully evaluated 11 of "
        "12 selected HPatches sequences. The v_bark sequence produced "
        "no matches and was retained as a legitimate failure case.\n\n"
    )

    f.write(
        "Across the 11 successful sequences, 5,739 matches were "
        "evaluated. The overall statistics obtained during Task 6 "
        "were:\n\n"
    )

    f.write(
        "Mean error: 0.9833 px\n"
        "Median error: 0.7693 px\n"
        "P90 error: 1.8550 px\n"
        "P95 error: 2.7779 px\n"
        "Within 3 px: 95.89%\n"
        "Within 5 px: 99.49%\n"
        "Within 10 px: 100.00%\n\n"
    )

    f.write(
        "These multi-sequence statistics provide broader evidence "
        "about the behavior of the official pretrained model than "
        "the single controlled image-pair comparison.\n\n"
    )

    f.write(
        "CONFIDENCE ANALYSIS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Task 4 showed a quality-versus-quantity relationship when "
        "increasing the LoFTR confidence threshold. On the controlled "
        "v_soldiers pair, increasing the threshold from 0.20 to 0.50 "
        "reduced the evaluated matches from 206 to 9 and reduced the "
        "mean geometric error from 2.5279 px to 1.5204 px. The "
        "3-pixel inlier rate was not strictly monotonic, demonstrating "
        "that confidence thresholding changes both quantity and the "
        "composition of the retained matches.\n\n"
    )

    f.write(
        "FAILURE ANALYSIS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Task 6 identified 2,545 high-confidence matches using a "
        "confidence threshold of 0.30. Of these, 62 had geometric "
        "errors greater than 3 pixels, corresponding to a 2.44% "
        "high-confidence geometric failure rate.\n\n"
    )

    f.write(
        "The v_bark sequence produced no matches. This result was "
        "retained rather than removed because failure to establish "
        "correspondences is itself relevant evidence when evaluating "
        "a matching system.\n\n"
    )

    f.write(
        "RESEARCH INTERPRETATION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "The project demonstrates a clear conceptual progression. "
        "Classical methods rely on explicit keypoint detection and "
        "local descriptors, whereas LoFTR represents image regions "
        "densely and uses Transformer attention to reason about "
        "relationships between the two images.\n\n"
    )

    f.write(
        "The Stage 3 experiments also demonstrate an important "
        "research lesson: implementing the architecture alone does "
        "not reproduce the behavior of a trained deep matching "
        "system. Learned feature representations and training are "
        "central to the performance of the official LoFTR model.\n\n"
    )

    f.write(
        "The controlled comparison shows that classical methods can "
        "produce highly geometrically consistent matches on the "
        "evaluated image pair, while the pretrained official LoFTR "
        "model produces strong geometric consistency at broader "
        "error thresholds. The simplified implementation does not "
        "reach comparable performance, which is consistent with its "
        "untrained and simplified nature.\n\n"
    )

    f.write(
        "LIMITATIONS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "1. The controlled four-method comparison uses one HPatches "
        "image pair.\n"
    )

    f.write(
        "2. The simplified LoFTR is not an exact reproduction of the "
        "official LoFTR training pipeline or learned parameters.\n"
    )

    f.write(
        "3. Mean geometric error is sensitive to extreme outliers.\n"
    )

    f.write(
        "4. The official checkpoint used in this project is the "
        "pretrained indoor model.\n"
    )

    f.write(
        "5. Results from different HPatches sequences should not be "
        "combined into a universal method ranking.\n"
    )

    f.write(
        "6. Runtime and memory were not treated as primary accuracy "
        "metrics in the final comparison.\n\n"
    )

    f.write(
        "FUTURE WORK\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Potential future work includes training the simplified "
        "LoFTR pipeline, evaluating additional LoFTR checkpoints, "
        "expanding the controlled comparison to more HPatches "
        "sequences, evaluating additional learned matching methods, "
        "performing precision-recall analysis, measuring runtime and "
        "memory usage, and investigating the v_bark no-match case "
        "in greater detail.\n\n"
    )

    f.write(
        "FINAL CONCLUSION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "This project developed a progression from classical local "
        "feature matching to Transformer-based detector-free "
        "matching. Rather than treating LoFTR as a black box, the "
        "project reconstructed and investigated its conceptual "
        "building blocks before evaluating the official pretrained "
        "implementation. The resulting experiments provide both "
        "architectural understanding and quantitative evidence of "
        "the differences between handcrafted descriptors, simplified "
        "learned matching, and a pretrained detector-free Transformer "
        "matcher.\n"
    )


print("=" * 80)
print("TASK 8 - FINAL RESEARCH EVALUATION")
print("=" * 80)

print("\nControlled methods:")
for method in methods:
    print("  -", method)

print("\nControlled comparison:")
print("  Sequence: v_soldiers")
print("  Image pair: 1.ppm -> 2.ppm")
print("  Ground truth: H_1_2")

print("\nOfficial LoFTR multi-sequence evaluation:")
print("  Successful sequences: 11")
print("  Evaluated matches: 5739")
print("  Mean error: 0.9833 px")
print("  Median error: 0.7693 px")
print("  P90 error: 1.8550 px")
print("  P95 error: 2.7779 px")
print("  Within 3 px: 95.89%")
print("  Within 5 px: 99.49%")
print("  Within 10 px: 100.00%")
print("  No-match sequence: v_bark")

print("\nGenerated:")
print("  - final_results_summary.csv")
print("  - final_research_report.txt")
print("  - controlled_comparison.png")
print("  - official_loftr_multisequence.png")
print("  - confidence_failure_summary.png")
print("  - match_count_overview.png")

print("\nResults directory:")
print(RESULTS_DIR)

print("\n" + "=" * 80)
print("TASK 8 COMPLETED")
print("=" * 80)