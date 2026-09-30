import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

STAGE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    STAGE_DIR,
    "results",
    "task_07d"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CONTROLLED EXPERIMENT
# ============================================================

DATASET = "HPatches"
SEQUENCE = "v_soldiers"
IMAGE_PAIR = "1.ppm -> 2.ppm"
GROUND_TRUTH = "H_1_2"


# ============================================================
# LOAD CONTROLLED SIFT RESULTS
# ============================================================

SIFT_SUMMARY = os.path.join(
    STAGE_DIR,
    "results",
    "task_07b",
    "sift_summary.txt"
)


# ============================================================
# LOAD CONTROLLED ORB RESULTS
# ============================================================

ORB_SUMMARY = os.path.join(
    STAGE_DIR,
    "results",
    "task_07c",
    "orb_summary.txt"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_summary(path):

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


def extract_number(text):

    text = text.replace(
        "px",
        ""
    ).strip()

    return float(text)


def extract_percentage(text):

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


def extract_count(text):

    if "/" not in text:
        return np.nan

    return int(
        text.split(
            "/",
            1
        )[0]
    )


# ============================================================
# READ SIFT
# ============================================================

sift = parse_summary(
    SIFT_SUMMARY
)

sift_matches = int(
    sift["Good matches"]
)

sift_mean = extract_number(
    sift["Mean geometric error"]
)

sift_median = extract_number(
    sift["Median geometric error"]
)

sift_1px = extract_percentage(
    sift["Matches within 1 px"]
)

sift_3px = extract_percentage(
    sift["Matches within 3 px"]
)

sift_5px = extract_percentage(
    sift["Matches within 5 px"]
)

sift_10px = extract_percentage(
    sift["Matches within 10 px"]
)


# ============================================================
# READ ORB
# ============================================================

orb = parse_summary(
    ORB_SUMMARY
)

orb_matches = int(
    orb["Good matches"]
)

orb_mean = extract_number(
    orb["Mean geometric error"]
)

orb_median = extract_number(
    orb["Median geometric error"]
)

orb_1px = extract_percentage(
    orb["Matches within 1 px"]
)

orb_3px = extract_percentage(
    orb["Matches within 3 px"]
)

orb_5px = extract_percentage(
    orb["Matches within 5 px"]
)

orb_10px = extract_percentage(
    orb["Matches within 10 px"]
)


# ============================================================
# SIMPLIFIED LOFTR
# ============================================================
#
# These values come from Stage 3 Task 7.
#
# IMPORTANT:
# These are predicted coarse-to-fine matches.
# They are NOT oracle-guided fine matches.
# ============================================================

simplified_matches = 45

simplified_mean = 25.8240

simplified_median = 24.0964

simplified_1px = np.nan
simplified_3px = np.nan
simplified_5px = np.nan

simplified_10px = (
    3 / 45
) * 100.0


# ============================================================
# OFFICIAL LOFTR
# ============================================================
#
# Existing Stage 4 Tasks 2-3 results.
# Sequence: v_soldiers
# Image pair: 1.ppm -> 2.ppm
# ============================================================

official_matches = 206

official_mean = 2.5279

official_median = 2.2551

official_1px = (
    32 / 206
) * 100.0

official_3px = (
    140 / 206
) * 100.0

official_5px = (
    196 / 206
) * 100.0

official_10px = (
    205 / 206
) * 100.0


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

methods = [
    "SIFT",
    "ORB",
    "Simplified LoFTR",
    "Official LoFTR"
]

match_counts = [
    sift_matches,
    orb_matches,
    simplified_matches,
    official_matches
]

mean_errors = [
    sift_mean,
    orb_mean,
    simplified_mean,
    official_mean
]

median_errors = [
    sift_median,
    orb_median,
    simplified_median,
    official_median
]

within_1px = [
    sift_1px,
    orb_1px,
    simplified_1px,
    official_1px
]

within_3px = [
    sift_3px,
    orb_3px,
    simplified_3px,
    official_3px
]

within_5px = [
    sift_5px,
    orb_5px,
    simplified_5px,
    official_5px
]

within_10px = [
    sift_10px,
    orb_10px,
    simplified_10px,
    official_10px
]


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "controlled_comparison.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "method",
        "dataset",
        "sequence",
        "image_pair",
        "ground_truth",
        "matches",
        "mean_error_px",
        "median_error_px",
        "within_1px_percent",
        "within_3px_percent",
        "within_5px_percent",
        "within_10px_percent"
    ])

    for i, method in enumerate(methods):

        writer.writerow([
            method,
            DATASET,
            SEQUENCE,
            IMAGE_PAIR,
            GROUND_TRUTH,
            match_counts[i],
            mean_errors[i],
            median_errors[i],
            within_1px[i],
            within_3px[i],
            within_5px[i],
            within_10px[i]
        ])


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 80)
print("TASK 7D - CONTROLLED FOUR-METHOD COMPARISON")
print("=" * 80)

print()
print("Dataset:     ", DATASET)
print("Sequence:    ", SEQUENCE)
print("Image pair:  ", IMAGE_PAIR)
print("Ground truth:", GROUND_TRUTH)

print("\n" + "-" * 80)

for i, method in enumerate(methods):

    print(f"\n{method}")

    print(
        f"  Matches:       {match_counts[i]}"
    )

    print(
        f"  Mean error:    {mean_errors[i]:.4f} px"
    )

    print(
        f"  Median error:  {median_errors[i]:.4f} px"
    )

    if not np.isnan(within_3px[i]):

        print(
            f"  <= 3 px:       "
            f"{within_3px[i]:.2f}%"
        )

    else:

        print(
            "  <= 3 px:       N/A"
        )

    if not np.isnan(within_5px[i]):

        print(
            f"  <= 5 px:       "
            f"{within_5px[i]:.2f}%"
        )

    else:

        print(
            "  <= 5 px:       N/A"
        )

    print(
        f"  <= 10 px:      "
        f"{within_10px[i]:.2f}%"
    )


# ============================================================
# PLOT HELPER
# ============================================================

def save_bar_plot(
    values,
    title,
    ylabel,
    filename,
    show_missing=False
):

    plt.figure(
        figsize=(10, 6)
    )

    plot_values = []

    labels = []

    for method, value in zip(
        methods,
        values
    ):

        labels.append(method)

        if np.isnan(value):

            plot_values.append(0)

        else:

            plot_values.append(value)

    bars = plt.bar(
        labels,
        plot_values
    )

    for bar, value in zip(
        bars,
        values
    ):

        if np.isnan(value):

            label = "N/A"

        else:

            label = f"{value:.2f}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom"
        )

    plt.title(title)

    plt.ylabel(ylabel)

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        dpi=200
    )

    plt.close()


# ============================================================
# GENERATE PLOTS
# ============================================================

save_bar_plot(
    match_counts,
    "Controlled Match Count Comparison",
    "Number of evaluated matches",
    "match_count_comparison.png"
)

save_bar_plot(
    mean_errors,
    "Controlled Mean Geometric Error",
    "Mean error (pixels)",
    "mean_error_comparison.png"
)

save_bar_plot(
    median_errors,
    "Controlled Median Geometric Error",
    "Median error (pixels)",
    "median_error_comparison.png"
)

save_bar_plot(
    within_3px,
    "Matches Within 3 Pixels",
    "Matches within 3 px (%)",
    "three_pixel_comparison.png"
)

save_bar_plot(
    within_5px,
    "Matches Within 5 Pixels",
    "Matches within 5 px (%)",
    "five_pixel_comparison.png"
)

save_bar_plot(
    within_10px,
    "Matches Within 10 Pixels",
    "Matches within 10 px (%)",
    "ten_pixel_comparison.png"
)


# ============================================================
# REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "controlled_comparison_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 4 TASK 7D - CONTROLLED FOUR-METHOD COMPARISON\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Experimental setup\n"
    )

    f.write(
        f"Dataset: {DATASET}\n"
    )

    f.write(
        f"Sequence: {SEQUENCE}\n"
    )

    f.write(
        f"Image pair: {IMAGE_PAIR}\n"
    )

    f.write(
        f"Ground truth: {GROUND_TRUTH}\n\n"
    )

    f.write(
        "The same image pair and ground-truth homography are used "
        "for the controlled comparison.\n\n"
    )

    f.write(
        "RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for i, method in enumerate(methods):

        f.write(
            f"\n{method}\n"
        )

        f.write(
            f"Matches: {match_counts[i]}\n"
        )

        f.write(
            f"Mean error: "
            f"{mean_errors[i]:.4f} px\n"
        )

        f.write(
            f"Median error: "
            f"{median_errors[i]:.4f} px\n"
        )

        if np.isnan(within_3px[i]):

            f.write(
                "Within 3 px: N/A\n"
            )

        else:

            f.write(
                f"Within 3 px: "
                f"{within_3px[i]:.2f}%\n"
            )

        if np.isnan(within_5px[i]):

            f.write(
                "Within 5 px: N/A\n"
            )

        else:

            f.write(
                f"Within 5 px: "
                f"{within_5px[i]:.2f}%\n"
            )

        f.write(
            f"Within 10 px: "
            f"{within_10px[i]:.2f}%\n"
        )

    f.write(
        "\n" + "=" * 70 + "\n"
    )

    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "SIFT and ORB are controlled classical baselines. "
        "Both were independently evaluated on the same "
        "v_soldiers image pair using H_1_2.\n\n"
    )

    f.write(
        "Simplified LoFTR represents the research-oriented "
        "implementation developed during Stage 3. Its values "
        "come from predicted coarse-to-fine matching rather "
        "than oracle-guided fine refinement.\n\n"
    )

    f.write(
        "Official LoFTR uses the pretrained indoor checkpoint "
        "and the same v_soldiers image pair.\n\n"
    )

    f.write(
        "The comparison should not be interpreted as a universal "
        "ranking of the four methods. It represents their observed "
        "behavior on this particular HPatches image pair.\n\n"
    )

    f.write(
        "Mean error is sensitive to large outliers. Median error "
        "and threshold-based geometric consistency should therefore "
        "be considered alongside mean error.\n\n"
    )

    f.write(
        "The simplified LoFTR experiment did not report directly "
        "comparable 3-pixel or 5-pixel percentages, so these values "
        "are reported as N/A rather than being inferred.\n"
    )


print("\n" + "-" * 80)

print("Generated files:")

print(
    "  - controlled_comparison.csv"
)

print(
    "  - controlled_comparison_report.txt"
)

print(
    "  - match_count_comparison.png"
)

print(
    "  - mean_error_comparison.png"
)

print(
    "  - median_error_comparison.png"
)

print(
    "  - three_pixel_comparison.png"
)

print(
    "  - five_pixel_comparison.png"
)

print(
    "  - ten_pixel_comparison.png"
)

print("\nResults saved to:")
print(RESULTS_DIR)

print("\n" + "=" * 80)
print("TASK 7D COMPLETED")
print("=" * 80)