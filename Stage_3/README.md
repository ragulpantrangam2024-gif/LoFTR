# Stage 3 — Simplified LoFTR

Stage 3 builds a simplified detector-free feature matching pipeline inspired by LoFTR.

## Task 1 — CNN Feature Extraction

### Objective

Replace the simple random patch projection used in Stage 2 with a CNN-based dense feature extractor.

Pipeline:

Image
→ CNN
→ Dense Spatial Feature Map

### Dataset

- HPatches sequence: `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- Original image size: 1290×968
- Resized size: 256×256

### CNN Architecture

- Conv2D: 1 → 32
- ReLU
- Conv2D: 32 → 64
- ReLU
- MaxPool: 2×2
- Conv2D: 64 → 64
- ReLU

### Results

Final feature map for each image:

`[1, 64, 128, 128]`

This corresponds to:

- 128×128 spatial locations
- 16,384 dense feature locations
- 64-dimensional feature vector per location

The CNN is randomly initialized and not trained. Therefore, the experiment demonstrates dense feature extraction rather than learned correspondence features.

### Key Learning

CNNs provide spatially organized dense features that can later be converted into tokens and processed by the Transformer matching module.

### Results Directory

`Stage_3/results/task_01/`
