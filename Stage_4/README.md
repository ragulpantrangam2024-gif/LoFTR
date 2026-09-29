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