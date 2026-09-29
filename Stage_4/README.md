# Stage 4 — Official LoFTR

Stage 4 transitions from the simplified LoFTR implementation developed in Stage 3 to the official pretrained LoFTR model.

## Task 1 — Official LoFTR Architecture Study

### Objective

The objective of Task 1 is to understand and experimentally demonstrate the major architectural components of LoFTR before using the official pretrained implementation.

This task is an educational architecture study and is **not an implementation of the official pretrained LoFTR model**.

### Dataset

- HPatches sequence: `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- Original image size: `1290 × 968`
- Resized image size: `256 × 256`

### Architecture Demonstration

The implemented pipeline demonstrates:

```text
Image Pair
    ↓
Dense CNN Feature Extraction
    ↓
Coarse Feature Representation
    ↓
2D Positional Encoding
    ↓
Self-Attention
    ↓
Cross-Attention
    ↓
Coarse Similarity Matching
    ↓
Fine-Level Matching Concept
Results

Dense CNN feature maps:

Image 1: [1, 64, 128, 128]
Image 2: [1, 64, 128, 128]

This represents:

128 × 128 = 16,384 spatial feature locations
64 features per spatial location

Coarse representation:

Image 1: [1, 256, 64]
Image 2: [1, 256, 64]

Therefore:

Coarse grid: 16 × 16
Number of coarse tokens: 256
Embedding dimension: 64

Transformer configuration:

Attention heads: 8
Self-attention: [1, 8, 256, 256]
Cross-attention: [1, 8, 256, 256]

Attention validation:

Self-attention row sum:
minimum = 0.99999976
maximum = 1.00000024

Cross-attention row sum:
minimum = 0.99999970
maximum = 1.00000024

The values are effectively 1.0, confirming correct attention normalization.

Coarse Matching

Similarity matrix:

[1, 256, 256]

Similarity statistics:

Minimum: 0.3925
Maximum: 1.0000
Mean:    0.6914

Mutual nearest-neighbor matches:

256
Interpretation

The architecture components operate correctly from a tensor and computational perspective.

However, the CNN and Transformer in this task are randomly initialized and untrained. Therefore, the 256 mutual matches must not be interpreted as 256 correct image correspondences.

The purpose of this task is to understand the flow of dense features through positional encoding, self-attention, cross-attention, and coarse matching.

# Stage 4 — Official Pretrained LoFTR and Research Evaluation

## Task 2 — Official Pretrained LoFTR Inference

### Objective

The objective of this task is to run the official pretrained **LoFTR (Detector-Free Local Feature Matching with Transformers)** implementation on real image pairs.

Unlike the simplified LoFTR implementation developed in Stage 3, this task uses the official ZJU3DV LoFTR implementation together with a pretrained checkpoint.

This establishes a baseline for evaluating the performance of a trained LoFTR model before performing quantitative geometric evaluation.

---

## Official Implementation

The official LoFTR implementation is used from the ZJU3DV LoFTR repository.

The external implementation is kept separately under:

```text
official_loftr/

The pretrained model used in this task is:

indoor_ds.ckpt

The checkpoint is stored locally under:

official_loftr/
└── weights/
    └── indoor_ds.ckpt

The pretrained model and external repository are not included in the project's Git history.

Dataset

The experiment uses the HPatches sequence:

datasets/
└── hpatches-sequences-release/
    └── v_soldiers/
        ├── 1.ppm
        ├── 2.ppm
        └── H_1_2

The image pair is:

Image 0: 1.ppm
Image 1: 2.ppm

The HPatches homography H_1_2 will be used in the following task for quantitative geometric evaluation.

Input Images

Original image dimensions:

Image 0: 1290 × 968
Image 1: 1290 × 968

Because the inference was performed on a CPU system, the images were resized while preserving their aspect ratio.

The maximum image dimension was limited to:

640 pixels

The resulting images were:

Image 0: 640 × 480
Image 1: 640 × 480

The dimensions were adjusted to be divisible by 8, as required by the LoFTR processing pipeline.

Model

The official pretrained LoFTR model was loaded using:

indoor_ds.ckpt

The model was executed in evaluation mode without additional training or fine-tuning.

The inference device was:

CPU
Processing Pipeline

The processing pipeline used in this task is:

HPatches image pair
        │
        ▼
Grayscale conversion
        │
        ▼
Resize to 640 × 480
        │
        ▼
Convert images to tensors
        │
        ▼
Official pretrained LoFTR
        │
        ├── Feature extraction
        ├── Coarse-level Transformer matching
        ├── Fine-level refinement
        └── Confidence estimation
        │
        ▼
Final LoFTR correspondences
Implementation Details

The task script is:

Stage_4/
└── task_02_official_loftr_inference.py

The script:

Loads the official pretrained LoFTR implementation.
Loads indoor_ds.ckpt.
Loads the HPatches v_soldiers image pair.
Converts the images to grayscale.
Resizes the images to a maximum dimension of 640 pixels.
Runs pretrained LoFTR inference.
Extracts:
matched coordinates in image 0
matched coordinates in image 1
matching confidence values
Measures model loading and inference time.
Saves the raw matches.
Generates a qualitative visualization.
Saves a text summary of the experiment.
Results
Model Loading
Model loading time: 0.70 seconds
Inference
Device: CPU
Inference time: 14.06 seconds
Matching Results
Number of matches: 206

Confidence statistics:

Minimum confidence: 0.201274
Maximum confidence: 0.567200
Mean confidence:    0.290910
Median confidence:  0.253034

The confidence values reported here are the raw confidence values returned by the pretrained LoFTR model.

Qualitative Visualization

The generated visualization contains the LoFTR correspondences between the two images.

Output:

results/
└── task_02/
    └── matches_visualization.png

The visualization provides a qualitative view of the correspondence distribution.

The matches are concentrated primarily around textured structures and recognizable regions of the scene.

However, visual inspection alone is not sufficient to determine geometric correctness.

Therefore, the HPatches ground-truth homography will be used in the next task for quantitative evaluation.

## Task 3 — Quantitative Evaluation of Official Pretrained LoFTR

### Objective

The objective of this task is to quantitatively evaluate the geometric accuracy of the correspondences produced by the official pretrained LoFTR model.

Task 2 demonstrated that the official pretrained LoFTR model can generate image correspondences between the HPatches `v_soldiers` image pair.

In Task 3, the predicted LoFTR correspondences are compared against the HPatches ground-truth homography `H_1_2`.

The evaluation measures the geometric reprojection error of each predicted correspondence.

---

## Dataset

The evaluation uses the HPatches sequence:

```text
datasets/
└── hpatches-sequences-release/
    └── v_soldiers/
        ├── 1.ppm
        ├── 2.ppm
        └── H_1_2

The image pair is:

Image 0: 1.ppm
Image 1: 2.ppm

The file H_1_2 contains the ground-truth homography mapping points from image 0 to image 1.

Input Correspondences

The correspondences used for this evaluation are the raw matches generated by Stage 4 Task 2.

Task 2 produced:

Number of LoFTR matches: 206

The matches are stored in:

Stage_4/
└── results/
    └── task_02/
        └── matches.npz

The file contains:

mkpts0
mkpts1
mconf

where:

mkpts0 contains matched coordinates in image 0.
mkpts1 contains matched coordinates in image 1.
mconf contains LoFTR matching confidence values.
Coordinate System

Task 2 performed LoFTR inference on resized images.

Original images:

Image 0: 1290 × 968
Image 1: 1290 × 968

LoFTR input images:

Image 0: 640 × 480
Image 1: 640 × 480

Therefore, the original HPatches homography cannot be applied directly to the resized LoFTR coordinates.

The homography was transformed into the resized coordinate system using:

H_resized = S2 · H_original · S1^-1

where:

S1 is the image-0 resize transformation.
S2 is the image-1 resize transformation.
H_original is the original HPatches homography.
H_resized is the homography used for evaluation.
Original HPatches Homography

The original ground-truth homography was:

[[ 1.3522e+00  2.5037e-02  9.6693e+01]
 [ 2.0588e-01  1.5085e+00 -2.7944e+02]
 [ 4.1800e-04  4.2466e-05  1.0103e+00]]

After transformation to the 640 × 480 coordinate system:

[[ 1.33841433e+00  2.47689475e-02  4.74581235e+01]
 [ 2.03886368e-01  1.49312086e+00 -1.37223497e+02]
 [ 8.34372629e-04  8.47228855e-05  1.00000000e+00]]
Geometric Error

For each LoFTR correspondence:

p0 → p1

the image-0 point is projected into image 1 using the ground-truth homography:

p1_projected = H_resized · p0

after homogeneous-coordinate normalization.

The geometric error is then calculated as:

error = ||p1_projected - p1_LoFTR||

where:

p1_projected is the ground-truth projected position.
p1_LoFTR is the position predicted by LoFTR.

The error is measured in pixels.

Evaluation Thresholds

The following geometric thresholds were evaluated:

1 pixel
3 pixels
5 pixels
10 pixels

A correspondence is considered geometrically consistent with the ground-truth homography when its reprojection error is below the selected threshold.

The 3-pixel threshold is used as the primary strict correctness criterion in this task.

Results
Number of Evaluated Matches
206 / 206

All 206 LoFTR correspondences could be evaluated against the ground-truth homography.

Geometric Error
Mean error:     2.5279 px
Median error:   2.2551 px
Minimum error:  0.2377 px
Maximum error: 11.0571 px
Accuracy by Geometric Threshold
Threshold	Correct Matches	Percentage
≤ 1 px	32 / 206	15.53%
≤ 3 px	140 / 206	67.96%
≤ 5 px	196 / 206	95.15%
≤ 10 px	205 / 206	99.51%
3-Pixel Geometric Inlier Rate

Using a 3-pixel geometric threshold:

Geometric inliers: 140
Total matches:     206

Therefore:

3-pixel geometric inlier rate = 67.96%

This can also be described as the 3-pixel geometric precision of the detected correspondences, because it measures the fraction of LoFTR's predicted matches that are consistent with the ground-truth homography within 3 pixels.

Important Interpretation

The results show that most of the LoFTR correspondences are geometrically close to the HPatches ground truth.

The error distribution is:

206 total matches
│
├── 32  within 1 px
├── 140 within 3 px
├── 196 within 5 px
├── 205 within 10 px
└── 1   above 10 px

The median geometric error is approximately:

2.26 pixels

which means that half of the evaluated correspondences have a geometric error below approximately 2.26 pixels.

The 5-pixel and 10-pixel thresholds show that the majority of detected correspondences are close to their ground-truth locations.

Precision and Recall Terminology

The evaluation does not report conventional recall over all possible image correspondences.

The complete set of possible correct correspondences in the image pair is not explicitly enumerated.

Therefore, the primary correctness measure used here is:

3-pixel geometric inlier rate

or equivalently:

3-pixel geometric precision

The value is:

67.96%

The percentage should not be interpreted as recall over all possible feature correspondences.

Output Files

The results are stored in:

Stage_4/
└── results/
    └── task_03/
        ├── evaluation_summary.txt
        ├── geometric_errors.csv
        ├── confidence_vs_error.csv
        └── projected_points.npz
evaluation_summary.txt

Contains:

image dimensions
transformed homography
number of evaluated matches
mean geometric error
median geometric error
threshold-based accuracy
3-pixel geometric inlier rate
geometric_errors.csv

Contains the geometric error for every evaluated LoFTR correspondence.

Format:

match_index,error_px
confidence_vs_error.csv

Contains the LoFTR confidence and corresponding geometric error for each match.

Format:

match_index,confidence,error_px

This file can be used to investigate the relationship between LoFTR confidence and geometric correctness.

projected_points.npz

Contains:

mkpts0
mkpts1
projected
errors
confidence
H_original
H_resized
Limitations
The evaluation uses a single HPatches image pair.
LoFTR inference was performed at 640 × 480 rather than the original image resolution.
The evaluation measures geometric consistency with the HPatches homography rather than semantic correctness.
The 3-pixel inlier rate should not be interpreted as recall over all possible image correspondences.
The results should not be generalized to all viewpoint changes, illumination conditions, or image sequences from this single experiment.

## Task 4 — LoFTR Confidence Threshold Analysis

### Objective

The objective of this task is to investigate the relationship between LoFTR matching confidence and geometric correspondence quality.

Task 3 evaluated all 206 correspondences produced by the official pretrained LoFTR model.

Task 4 applies different confidence thresholds to the same correspondences and measures how confidence filtering changes:

- the number of retained matches
- mean geometric error
- median geometric error
- geometric accuracy
- 3-pixel geometric inlier rate
- 5-pixel geometric inlier rate
- 10-pixel geometric inlier rate

No LoFTR inference is performed in this task. The task reuses the results generated by Tasks 2 and 3.

---

## Input Data

Task 4 uses:

```text
Stage_4/
├── results/
│   ├── task_02/
│   │   └── matches.npz
│   │
│   └── task_03/
│       └── confidence_vs_error.csv

The original experiment contains:

Total LoFTR matches: 206

The confidence values range from:

Minimum: 0.201274
Maximum: 0.567200
Confidence Thresholds

The following LoFTR confidence thresholds were evaluated:

0.20
0.25
0.30
0.35
0.40
0.45
0.50
0.55

For a threshold T, only matches satisfying:

confidence >= T

are retained.

Evaluation Metrics

For each confidence threshold, the following metrics are calculated:

retained matches
percentage of original matches retained
mean confidence
mean geometric error
median geometric error
matches within 1 pixel
matches within 3 pixels
matches within 5 pixels
matches within 10 pixels

The primary correctness measure is the:

3-pixel geometric inlier rate

This represents the percentage of retained LoFTR matches whose geometric error is at most 3 pixels with respect to the HPatches ground-truth homography.

Results
Confidence Threshold	Retained Matches	Retained	Mean Error (px)	Median Error (px)	≤3 px	≤5 px
0.20	206	100.00%	2.5279	2.2551	67.96%	95.15%
0.25	108	52.43%	2.3332	2.1377	72.22%	96.30%
0.30	69	33.50%	2.2080	2.0549	71.01%	98.55%
0.35	46	22.33%	2.0382	1.7694	78.26%	97.83%
0.40	29	14.08%	1.9649	1.5044	75.86%	96.55%
0.45	17	8.25%	1.6731	1.3475	88.24%	94.12%
0.50	9	4.37%	1.5204	1.2524	88.89%	100.00%
0.55	1	0.49%	2.2479	2.2479	100.00%	100.00%
Interpretation

The results demonstrate a quality-versus-quantity trade-off when applying confidence filtering.

At the lowest threshold:

Confidence >= 0.20

all 206 matches are retained.

The mean geometric error is:

2.5279 pixels

with a 3-pixel geometric inlier rate of:

67.96%

At a confidence threshold of 0.50:

9 matches

remain.

The mean geometric error decreases to:

1.5204 pixels

and the 3-pixel geometric inlier rate becomes:

88.89%

Therefore, increasing the confidence threshold can remove lower-quality correspondences and reduce the average geometric error.

However, this comes at the cost of a substantial reduction in the number of available correspondences.

Non-Monotonic Inlier Rate

The 3-pixel geometric inlier rate does not increase monotonically with confidence.

For example:

0.25 → 72.22%
0.30 → 71.01%
0.35 → 78.26%
0.40 → 75.86%
0.45 → 88.24%
0.50 → 88.89%

This indicates that LoFTR confidence is not a perfect predictor of geometric correctness.

Instead, confidence acts as a useful filtering signal whose effectiveness depends on the selected threshold and the available number of matches.

Important Interpretation of the 0.55 Result

At:

Confidence >= 0.55

only one correspondence remains.

That single correspondence has an error below 3 pixels, resulting in:

100% geometric inlier rate

However, this should not be interpreted as evidence that the 0.55 threshold provides superior matching quality.

The sample contains only one correspondence, so the statistic is not representative of the full matching population.

This illustrates why both correspondence quality and correspondence quantity must be considered together.

Confidence and Geometric Error

The experiment also generates a scatter plot showing:

LoFTR confidence
        vs
geometric error

This provides a direct visual analysis of whether high-confidence matches tend to have lower geometric errors.

The relationship is not expected to be perfectly deterministic because LoFTR confidence and geometric error measure different properties.

Output Files

Results are stored in:

Stage_4/
└── results/
    └── task_04/
        ├── confidence_threshold_results.csv
        ├── confidence_analysis_summary.txt
        ├── confidence_vs_geometric_error.png
        ├── confidence_vs_match_count.png
        ├── confidence_vs_3px_inlier_rate.png
        └── confidence_vs_mean_error.png
confidence_threshold_results.csv

Contains the quantitative results for every confidence threshold.

confidence_analysis_summary.txt

Contains a text summary of the threshold experiment.

confidence_vs_geometric_error.png

Scatter plot showing LoFTR confidence against geometric error.

confidence_vs_match_count.png

Shows how increasing the confidence threshold reduces the number of retained correspondences.

confidence_vs_3px_inlier_rate.png

Shows the relationship between confidence filtering and the 3-pixel geometric inlier rate.

confidence_vs_mean_error.png

Shows how the average geometric error changes as the confidence threshold increases.

Limitations
The experiment uses a single HPatches image pair.
Only 206 LoFTR correspondences are available.
High confidence does not guarantee geometric correctness.
Very high thresholds can leave too few matches for reliable statistical conclusions.
The results should not be generalized to all HPatches sequences based on this single experiment.
The experiment evaluates confidence as a filtering signal rather than as a calibrated probability of correctness.
Research Conclusion

For this v_soldiers image pair, confidence filtering demonstrates a clear trade-off between correspondence quantity and geometric quality.

Increasing the threshold from 0.20 to 0.50 reduces the mean geometric error from:

2.5279 px

to:

1.5204 px

while reducing the number of retained correspondences from:

206

to:

9

The results therefore demonstrate that confidence filtering can improve the geometric quality of the retained matches, but increasingly aggressive filtering substantially reduces correspondence coverage.

The non-monotonic behavior of the 3-pixel inlier rate also shows that confidence should not be treated as a perfect indicator of geometric correctness.


# Stage 4 - Task 5: Multi-Sequence Evaluation

## Overview

This task evaluates the **official pretrained LoFTR model** across multiple image pairs from the HPatches dataset.

The purpose is to determine how the pretrained LoFTR model behaves under different types of image changes, particularly:

- viewpoint changes
- illumination changes

Unlike the previous tasks, which focused primarily on a single image pair, this task evaluates LoFTR across multiple HPatches sequences to examine the consistency and robustness of the matching behavior.

The evaluation uses the **official pretrained LoFTR checkpoint**:

```text
official_loftr/weights/indoor_ds.ckpt

The official LoFTR source repository is kept locally and is excluded from Git tracking.

1. Objectives

The main objectives of this task are:

Evaluate official pretrained LoFTR on multiple HPatches sequences.
Compare matching behavior under viewpoint and illumination changes.
Measure the geometric accuracy of the predicted correspondences.
Analyze the number of matches produced by LoFTR.
Measure confidence statistics of the predicted correspondences.
Identify difficult sequences and failure cases.
Build a dataset-wide evaluation that can later be compared with SIFT, ORB, and the simplified LoFTR implementation.
2. Dataset

The evaluation uses the HPatches image matching dataset.

Each sequence contains related images and a ground-truth homography between image 1 and image 2.

For each sequence, the following files are used:

1.ppm
2.ppm
H_1_2

where:

1.ppm = reference image
2.ppm = transformed image
H_1_2 = ground-truth homography mapping image 1 coordinates to image 2 coordinates

The dataset is stored locally under:

datasets/hpatches-sequences-release/

The dataset is excluded from GitHub using .gitignore.

3. Evaluated Sequences

A total of 12 HPatches sequences were selected.

Viewpoint sequences
v_woman
v_graffiti
v_wall
v_yard
v_london
v_bark
Illumination sequences
i_ajuntament
i_bologna
i_londonbridge
i_santuario
i_school
i_zion

The sequence names correspond to the actual directory names present in the local HPatches dataset.

4. Experimental Setup
Model

The experiment uses the official pretrained LoFTR model.

Model: LoFTR
Checkpoint: indoor_ds.ckpt

The model is loaded once and reused for all sequences.

Device

The evaluation was performed on:

CPU
Image preprocessing

Images are resized while preserving their aspect ratio.

The maximum image dimension is:

640 pixels

The resulting dimensions are adjusted to multiples of 8 to remain compatible with the LoFTR architecture.

5. Homography Transformation

The HPatches homography is defined in the original image coordinate system.

Since the images are resized before LoFTR inference, the ground-truth homography must also be transformed into the resized coordinate system.

The transformation used is:

$$ H_{resized} = S_2 H_{original} S_1^{-1} $$

where:

\(H_{original}\) is the original HPatches homography
\(S_1\) is the scaling matrix for image 1
\(S_2\) is the scaling matrix for image 2
\(H_{resized}\) is the homography used for geometric evaluation

This ensures that the predicted LoFTR coordinates and ground-truth coordinates are expressed in the same coordinate system.

6. LoFTR Matching

For each image pair, LoFTR produces:

mkpts0_f
mkpts1_f
mconf

where:

mkpts0_f = matched coordinates in image 1
mkpts1_f = corresponding coordinates in image 2
mconf = LoFTR confidence values

No additional confidence filtering is applied in this task.

The objective is to evaluate the raw output of the pretrained model.

7. Geometric Error

For every predicted correspondence, the point from image 1 is transformed using the ground-truth homography.

For a correspondence:

$$ p_0 \rightarrow p_1 $$

the ground-truth projected point is:

$$ \hat{p}_1 = H_{resized}p_0 $$

The geometric error is the Euclidean distance:

$$ e = \left\| \hat{p}_1-p_1 \right\|_2 $$

The error is measured in pixels.

A smaller error indicates that the predicted correspondence is closer to the ground-truth geometric transformation.

8. Evaluation Metrics

The following metrics are calculated for every sequence.

Number of matches

Number of correspondences returned by LoFTR.

Mean geometric error

Average geometric error across all evaluated matches.

Median geometric error

Median geometric error.

The median is useful because it is less affected by a small number of large errors.

Minimum and maximum error

The minimum and maximum geometric errors are also recorded.

Accuracy thresholds

The percentage of matches satisfying:

error ≤ 1 pixel
error ≤ 3 pixels
error ≤ 5 pixels
error ≤ 10 pixels

are reported.

Confidence statistics

The following LoFTR confidence statistics are recorded:

mean confidence
median confidence
minimum confidence
maximum confidence
Runtime

Inference time and total sequence runtime are recorded.

9. Results

The complete evaluation produced successful LoFTR correspondences on:

11 / 12 sequences

The sequence v_bark produced no matches.

Viewpoint Results
Sequence	Matches	Mean Error (px)	Median Error (px)	≤1 px	≤3 px	≤5 px	≤10 px
v_woman	16	3.6115	3.7619	6.25%	43.75%	68.75%	100.00%
v_graffiti	224	2.5996	2.3683	14.29%	63.39%	94.20%	100.00%
v_wall	275	1.6731	1.4281	32.00%	88.00%	99.27%	100.00%
v_yard	422	1.8344	1.7423	22.99%	87.91%	100.00%	100.00%
v_london	202	2.4346	2.2689	14.36%	70.30%	96.04%	100.00%
v_bark	0	—	—	—	—	—	—
Viewpoint observations

The viewpoint sequences show substantially different matching behavior.

v_wall and v_yard produced relatively high proportions of geometrically accurate matches, with approximately 88% of matches within 3 pixels.

v_london and v_graffiti produced lower 3-pixel rates of approximately 70% and 63%, respectively.

v_woman produced only 16 matches, of which 7 were within 3 pixels.

The v_bark sequence produced no LoFTR matches and therefore has no geometric error statistics.

The results demonstrate that viewpoint changes can produce significantly different matching difficulty depending on the individual scene.

10. Illumination Results
Sequence	Matches	Mean Error (px)	Median Error (px)	≤1 px	≤3 px	≤5 px	≤10 px
i_ajuntament	949	0.5573	0.5510	97.15%	100.00%	100.00%	100.00%
i_bologna	185	0.5408	0.5232	98.38%	100.00%	100.00%	100.00%
i_londonbridge	251	0.8264	0.8344	74.10%	100.00%	100.00%	100.00%
i_santuario	524	0.8802	0.9000	64.12%	99.81%	99.81%	100.00%
i_school	1329	0.7443	0.7378	79.68%	100.00%	100.00%	100.00%
i_zion	1362	0.7270	0.7221	83.70%	100.00%	100.00%	100.00%
Illumination observations

The illumination sequences produced highly accurate geometric correspondences.

All five illumination sequences with matches achieved:

100% within 10 pixels

Five sequences also achieved approximately 99.8–100% within 3 pixels, with i_santuario producing:

99.81% within 3 pixels

The mean geometric errors for all illumination sequences remained below 1 pixel.

The lowest mean error was obtained on:

i_bologna
0.5408 px

followed by:

i_ajuntament
0.5573 px
11. Failure Case
v_bark

The v_bark sequence produced:

Status: no_matches

No LoFTR correspondences were returned for this image pair.

Consequently, geometric error metrics cannot be calculated for this sequence.

This is retained as part of the evaluation rather than removing the sequence from the dataset.

A no-match case is important for the final research analysis because it demonstrates that the pretrained model does not necessarily produce usable correspondences for every image pair.

12. Confidence Information

The mean LoFTR confidence values varied between sequences.

Examples include:

v_woman          0.2610
v_graffiti       0.2718
v_wall           0.3048
v_yard           0.3005
v_london         0.2930

i_ajuntament     0.3267
i_bologna        0.2809
i_londonbridge   0.2828
i_santuario      0.2989
i_school         0.3286
i_zion           0.3219

Confidence values are retained for later analysis.

They will be used together with geometric error in subsequent research analysis to investigate whether LoFTR confidence can help distinguish reliable and unreliable matches.

13. Results Storage

The aggregate results are stored in:

Stage_4/results/task_05/multisequence_results.csv

The textual summary is stored in:

Stage_4/results/task_05/multisequence_summary.txt

Each successfully evaluated sequence also has its own directory containing:

matches_and_errors.npz
errors.csv

For example:

Stage_4/results/task_05/v_graffiti/
├── matches_and_errors.npz
└── errors.csv

The .npz files contain:

mkpts0
mkpts1
confidence
errors
H_original
H_resized

This allows the individual correspondences to be reanalyzed later without rerunning LoFTR.

14. Interpretation

The multi-sequence experiment provides evidence that the pretrained LoFTR model can produce highly accurate geometric correspondences across the evaluated HPatches illumination sequences.

The viewpoint sequences show greater variation. Some viewpoint sequences achieve high geometric accuracy, while others produce fewer matches or larger errors.

The results therefore provide two important observations for the final research evaluation:

LoFTR performance depends strongly on the characteristics of the image pair.
A model can produce highly accurate correspondences on successful sequences while still producing few or no correspondences on difficult sequences.

These results should not be interpreted as a universal ranking of LoFTR against other feature-matching methods. The experiment evaluates a specific pretrained checkpoint on a selected subset of HPatches sequences using the evaluation protocol described above.

15. Research Significance

This task extends the previous single-sequence LoFTR evaluation into a multi-sequence experiment.

The progression is:

Task 1
Official LoFTR architecture

        ↓

Task 2
Official pretrained LoFTR inference

        ↓

Task 3
Geometric accuracy evaluation

        ↓

Task 4
Confidence analysis

        ↓

Task 5
Multi-sequence evaluation

This creates the foundation for the final research comparison between:

SIFT
ORB
Simplified LoFTR
Official pretrained LoFTR

The multi-sequence results will also support later failure analysis and cross-condition comparisons.

16. Limitations

Several limitations should be considered.

1. CPU inference

The evaluation was performed on CPU, so the measured inference times should not be interpreted as representative of GPU inference performance.

2. Selected HPatches subset

Only 12 sequences were evaluated rather than the complete HPatches dataset.

Therefore, the results represent this selected evaluation subset.

3. Pretrained model

The official LoFTR model was evaluated using the available pretrained checkpoint.

No additional training or fine-tuning was performed.

4. No confidence filtering

The multi-sequence evaluation uses the raw LoFTR correspondences.

Confidence thresholding was investigated separately in Task 4.

5. Geometric metric

The main geometric metric is point-transfer error using the HPatches ground-truth homography.

It does not capture every aspect of correspondence quality.

17. Conclusion

Stage 4 Task 5 successfully evaluated the official pretrained LoFTR model across 12 selected HPatches sequences.

Results were obtained for:

11 / 12 sequences

while:

v_bark

produced no matches.

The illumination sequences showed consistently low geometric errors, while viewpoint sequences showed greater variation in both match count and geometric accuracy.

The resulting per-match data and aggregate statistics provide the foundation for the final multi-condition comparison and failure analysis of the project.