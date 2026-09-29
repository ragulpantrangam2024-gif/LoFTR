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