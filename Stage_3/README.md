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