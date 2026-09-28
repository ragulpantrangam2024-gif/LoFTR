# Stage 3 — Simplified LoFTR

Stage 3 builds a simplified detector-free feature matching pipeline inspired by LoFTR.

## Task 1 — CNN Feature Extraction

### Objective
Replace the simple random patch projection used in Stage 2 with a CNN-based dense feature extractor.

### Dataset
- HPatches: `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- Original image size: 1290×968
- Resized size: 256×256

### CNN Output
Final feature map:

`[1, 64, 128, 128]`

This provides 16,384 spatial feature locations with a 64-dimensional feature vector at each location.

The CNN is randomly initialized and not trained.

---

## Task 2 — Dense Feature Representation

### Objective
Convert the CNN feature map into a sequence of dense tokens suitable for Transformer processing and add 2D positional information.

### Pipeline

Image  
→ CNN  
→ Dense Feature Map  
→ Flatten Spatial Locations  
→ Dense Tokens  
→ 2D Positional Encoding  
→ Position-Aware Tokens

### Configuration

- Dataset: HPatches `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- Input size: 256×256
- CNN feature map: `[1,64,128,128]`
- Spatial locations: `128×128 = 16,384`
- Feature dimension: 64
- Dense token representation: `[1,16384,64]`

### Results

For both images:

```text
CNN feature map:
[1,64,128,128]

Dense tokens:
[1,16384,64]

Positional encoding:
[128,128,64]

Position-aware tokens:
[1,16384,64]

Mean feature-token norm:

Image 1: 0.3512
Image 2: 0.3472

Mean positional encoding norm:

5.6569

Mean position-aware token norm:

Image 1: 5.7562
Image 2: 5.7548
Observation

The CNN feature map is converted into a dense sequence where every spatial location becomes a feature token.

Unlike SIFT or ORB, the representation does not depend on explicitly detecting keypoints.

2D positional encoding provides the Transformer with information about the spatial location of each feature token.

The CNN is randomly initialized and untrained, so these features do not represent learned semantic correspondences. The task demonstrates the representation mechanism rather than LoFTR-level matching performance.

## Task 3 — Transformer Matching Module

### Objective

Introduce self-attention and cross-attention so that dense features from the two images can exchange contextual information.

### Pipeline

Image  
→ CNN  
→ Coarse Feature Map  
→ Dense Tokens  
→ Positional Encoding  
→ Self-Attention  
→ Cross-Attention  
→ Contextualized Features

### Configuration

- Dataset: HPatches `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- Input size: 256×256
- CNN feature map: `[1,64,128,128]`
- Coarse feature map: `[1,64,16,16]`
- Coarse tokens: `[1,256,64]`
- Embedding dimension: 64
- Attention heads: 8

### Results

Transformer output:

```text
Image 1: [1,256,64]
Image 2: [1,256,64]

Self-attention:

Image 1: [1,8,256,256]
Image 2: [1,8,256,256]

Cross-attention:

Image 1 → Image 2: [1,8,256,256]
Image 2 → Image 1: [1,8,256,256]

Attention row sums were approximately 1.0:

Self-attention:
min = 0.99999976
max = 1.00000024

Cross-attention:
min = 0.99999976
max = 1.00000024
Interpretation

Self-attention allows features within each image to exchange contextual information.

Cross-attention allows features from one image to interact with features from the other image.

The implementation uses a reduced 16×16 coarse representation because full attention over 16,384 tokens would require a very large 16384 × 16384 attention matrix.

The CNN and Transformer are randomly initialized and untrained. Therefore, the attention maps demonstrate the mechanism but should not yet be interpreted as meaningful image correspondences.

## Task 4 — Coarse Matching

### Objective

Use the contextualized Transformer features to establish coarse correspondences between the two images.

### Pipeline

Contextualized Features  
→ L2 Normalization  
→ Cosine Similarity  
→ Similarity Matrix  
→ Mutual Nearest-Neighbor Matching  
→ Confidence Filtering  
→ Geometric Evaluation

### Configuration

- Dataset: HPatches `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- Coarse tokens: 256
- Feature dimension: 64
- Similarity metric: cosine similarity
- Similarity threshold: 0.80
- Ground-truth source: HPatches `H_1_2`
- Geometric evaluation threshold: 10 pixels

### Results

Similarity matrix:

```text
[1,256,256]

Similarity statistics:

Minimum: 0.0893
Maximum: 0.7990
Mean:    0.5031

Using a similarity threshold of 0.80:

Mutual coarse matches: 0
Interpretation

The coarse matching mechanism was successfully implemented, including similarity computation, mutual nearest-neighbor matching, and homography-based geometric evaluation.

However, no matches passed the confidence threshold.

The maximum similarity (0.7990) was slightly below the selected threshold (0.80).

More importantly, the CNN and Transformer are randomly initialized and untrained. Therefore, the resulting features are not expected to provide reliable image correspondences.

This demonstrates an important principle of learned feature matching:

The attention mechanism provides feature interaction, but meaningful correspondence requires learned feature representations.

The HPatches ground-truth homography was successfully loaded and scaled to the 256×256 image resolution.