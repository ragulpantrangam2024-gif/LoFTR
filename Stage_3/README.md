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

## Task 5 — Fine-Level Refinement

### Objective

Demonstrate local fine-level refinement after obtaining an approximate coarse correspondence.

### Pipeline

Ground-Truth-Guided Coarse Location  
→ Fine Feature Map  
→ Local 9×9 Search Window  
→ Cosine Similarity  
→ Best Fine Location  
→ Geometric Error Evaluation

### Configuration

- Dataset: HPatches `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- Fine feature map: `[1,64,128,128]`
- Feature dimension: 64
- Ground-truth-guided coarse points: 50
- Fine search radius: 4 feature pixels
- Fine search window: 9×9

### Results

```text
Refined matches: 50

Mean error: 6.68 px
Median error: 7.64 px

Within 3 px: 12 / 50
Within 5 px: 16 / 50
Within 10 px: 41 / 50

Accuracy within 10 px: 82.0%
Mean local similarity: 0.99977
Important Experimental Note

The coarse locations used in this experiment were generated using the HPatches ground-truth homography.

Therefore, this is not an end-to-end matching evaluation.

The experiment evaluates the fine-level local refinement mechanism when an approximately correct coarse location is already available.

Interpretation

The fine-level refinement produced 50 refined correspondences, with 41 of them within 10 pixels of the ground-truth location.

However, the CNN is randomly initialized and untrained. The very high local similarity (0.99977) should therefore not be interpreted as evidence of learned visual correspondence.

The experiment demonstrates the coarse-to-fine refinement mechanism rather than LoFTR-level performance.

## Task 6 — End-to-End Simplified LoFTR

### Objective

Integrate the CNN feature extractor, dense representation, Transformer matching module, coarse matching, and fine refinement into one simplified end-to-end pipeline.

### Pipeline

Image Pair  
→ CNN Feature Extraction  
→ Dense Feature Representation  
→ Coarse Tokens + Positional Encoding  
→ Self-Attention  
→ Cross-Attention  
→ Coarse Similarity  
→ Mutual Nearest-Neighbor Matching  
→ Fine-Level Refinement

### Configuration

- Dataset: HPatches `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- CNN feature map: `[1,64,128,128]`
- Coarse tokens: `[1,256,64]`
- Transformer output: `[1,256,64]`
- Similarity matrix: `[1,256,256]`
- Similarity threshold: `0.80`
- Fine search radius: 4 feature pixels

### End-to-End Coarse Results

```text
Similarity minimum: 0.0893
Similarity maximum: 0.7990
Similarity mean:    0.5031

Predicted coarse matches: 0

The maximum similarity was slightly below the selected threshold of 0.80.

Therefore, the untrained simplified pipeline produced no confident predicted coarse correspondences.

Oracle-Guided Fine Results

Because the predicted coarse stage produced no matches, a separate oracle-guided experiment was performed.

Ground-truth homography was used to select 50 approximate coarse locations. These locations were then passed to the fine-level refinement stage.

Oracle coarse points: 50
Fine refined points: 50

Mean error: 6.68 px
Median error: 7.64 px

Within 3 px: 12 / 50
Within 5 px: 16 / 50
Within 10 px: 41 / 50

Accuracy within 10 px: 82.0%
Mean fine similarity: 0.99977
Important Interpretation

The 82% fine-stage result is not end-to-end matching accuracy.

The coarse locations for this experiment were obtained from the HPatches ground-truth homography. Therefore, this experiment evaluates the fine refinement mechanism after providing an approximately correct location.

The CNN and Transformer are randomly initialized and untrained. Consequently, the model does not currently learn meaningful visual correspondences.

Overall Observation

The complete simplified LoFTR pipeline has now been implemented:

CNN
 ↓
Dense Features
 ↓
Transformer
 ↓
Coarse Matching
 ↓
Fine Refinement

The experiment demonstrates the architectural concepts of detector-free dense matching and coarse-to-fine correspondence refinement.

It does not reproduce the performance of the official pretrained LoFTR model.

## Task 7 — Quantitative Evaluation

### Objective

Task 7 performs a quantitative evaluation of the complete simplified LoFTR pipeline developed in Stage 3.

The evaluation uses:

- HPatches sequence: `v_soldiers`
- Images: `1.ppm` and `2.ppm`
- Image size after resizing: `256 × 256`
- Coarse grid: `16 × 16`
- Coarse tokens: `256`
- Feature dimension: `64`
- Fine feature map: `128 × 128`
- Fine search radius: `4` feature pixels
- Geometric correctness threshold: `10 px`

The evaluation measures both coarse matching and fine-level refinement.

### Evaluation Pipeline

```text
Image 1
   ↓
CNN Feature Extraction
   ↓
Coarse Feature Representation
   ↓
Transformer Contextualization
   ↓
Cosine Similarity
   ↓
Mutual Nearest-Neighbor Matching
   ↓
Similarity Threshold
   ↓
Coarse Geometric Evaluation
   ↓
Fine Local Refinement
   ↓
Fine Geometric Evaluation
Homography Evaluation

The HPatches ground-truth homography H_1_2 was used to transform source-image coordinates into the target-image coordinate system.

Because the images are resized to 256 × 256, the original homography was converted using:

$$ H_{resized}=S_2H_{original}S_1^{-1} $$

where S1 and S2 represent the original-to-resized coordinate transformations.

Threshold Sensitivity

The similarity thresholds evaluated were:

0.50
0.60
0.70
0.75
0.80

Results:

Threshold	Coarse Matches	Correct ≤10 px	Precision	Recall	Mean Error
0.50	221	10	4.52%	4.78%	35.42 px
0.60	221	10	4.52%	4.78%	35.42 px
0.70	221	10	4.52%	4.78%	35.42 px
0.75	215	10	4.65%	4.78%	35.13 px
0.80	45	10	22.22%	4.78%	24.00 px

At the highest evaluated threshold of 0.80, the number of accepted coarse matches decreased substantially while the number of geometrically correct matches remained unchanged.

Consequently, coarse precision increased from 4.52% at thresholds 0.50–0.70 to 22.22% at 0.80.

However, recall remained 4.78%.

Fine-Level Evaluation

The fine stage was evaluated using the predicted coarse target locations.

At threshold 0.80:

Fine matches:       45
Correct ≤10 px:      3
Fine precision:      6.67%
Mean error:         25.82 px
Median error:       24.10 px
Within 3 px:          0
Within 5 px:          0
Within 10 px:         3

The fine stage therefore did not improve the geometric accuracy in this experiment.

Interpretation

The CNN and Transformer used in Stage 3 are randomly initialized and untrained.

Therefore, these results should not be interpreted as the performance of a trained LoFTR model.

The experiment demonstrates that:

Dense CNN features can be extracted.
Dense features can be converted into coarse tokens.
Transformer self-attention and cross-attention can contextualize the tokens.
A similarity matrix can be constructed.
Mutual nearest-neighbor coarse matching can be performed.
A similarity threshold can control the number of accepted matches.
Predicted coarse matches can be passed to a local fine-search stage.
Homography-based geometric evaluation can quantify correspondence quality.

The low precision and recall demonstrate the limitation of using randomly initialized, untrained feature representations.

At threshold 0.80, coarse precision reached 22.22%, but recall remained only 4.78%. Fine refinement reduced the precision to 6.67%.

This indicates that the fine matching mechanism itself cannot compensate for poor underlying feature representations.

Important Experimental Limitation

Task 5 used ground-truth-guided fine search locations and therefore its 82% within 10 px result should not be interpreted as end-to-end matching performance.

Task 7 instead evaluates fine refinement around predicted coarse locations and is therefore the more appropriate experiment for evaluating the complete pipeline.

Conclusion

Stage 3 successfully implements an educational simplified LoFTR-style pipeline from dense CNN feature extraction through Transformer matching, coarse matching, fine refinement, and quantitative geometric evaluation.

However, the experiment also demonstrates the distinction between implementing an architecture and obtaining useful learned representations.

The current model is untrained. Meaningful correspondence performance requires training or pretrained feature representations.