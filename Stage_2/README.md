# Stage 2 — Transformer Fundamentals

This stage introduces the Transformer concepts required to understand
Detector-Free Local Feature Matching with Transformers (LoFTR).

The goal is to understand the internal mechanisms step by step before
implementing a simplified LoFTR architecture.

---

## Stage 2 Pipeline

```text
Task 1A — Self-Attention with Toy Vectors
            ↓
Task 1B — Self-Attention on a Real Image
            ↓
Task 2 — Cross-Attention
            ↓
Task 3 — Positional Encoding
            ↓
Task 4 — Transformer Encoder
            ↓
Task 5 — Coarse-to-Fine Matching
            ↓
Stage 3 — Simplified LoFTR
Task 1A — Self-Attention
Objective

The objective of Task 1A is to understand the basic mechanism of
scaled dot-product self-attention using a small toy input.

Self-attention allows every token in a sequence to interact with every
other token.

The fundamental equation is:

$$ Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

where:

Q = Query
K = Key
V = Value
d_k = dimensionality of the Key vectors
Self-Attention Pipeline
Input tokens
     ↓
Linear projections
     ↓
Q, K, V
     ↓
QKᵀ
     ↓
Scale by √dₖ
     ↓
Softmax
     ↓
Attention weights
     ↓
Weighted sum of V
     ↓
Attention output
Query, Key and Value

A simple interpretation is:

Query: What information am I looking for?
Key: What information do I contain?
Value: What information should I provide?

The similarity between queries and keys determines how strongly
information from different tokens contributes to the output.

Implementation

The implementation uses PyTorch and manually calculates the
scaled dot-product attention operation.

The experiment prints:

Query matrix
Key matrix
Value matrix
QKᵀ
Scaled attention scores
Attention weights
Final self-attention output
Attention row-sum verification

The attention weights are normalized using softmax.

Each attention row therefore sums approximately to 1.0.

Task 1B — Image Self-Attention
Objective

Task 1B extends the concept of self-attention from toy vectors to
an actual image.

The HPatches image

datasets/hpatches-sequences-release/v_woman/1.ppm

is used as the input.

This experiment demonstrates how an image can be converted into a
sequence of tokens and processed using self-attention.

Image-to-Token Pipeline
HPatches Image
      ↓
Resize to 256 × 256
      ↓
Divide into 16 × 16 patches
      ↓
256 image patches
      ↓
Flatten each patch
      ↓
256-dimensional patch vector
      ↓
Linear embedding
      ↓
64-dimensional token
      ↓
Q, K, V
      ↓
Self-Attention
Image Information

Original image:

Height: 767
Width: 1034

After resizing:

Height: 256
Width: 256
Patch Representation

Patch size:

16 × 16 pixels

Number of patch rows:

16

Number of patch columns:

16

Therefore:

$$ 16 \times 16 = 256 $$

Total number of image tokens:

256

Each patch contains:

$$ 16 \times 16 = 256 $$

pixel values.

Therefore the patch tensor has shape:

(256, 256)
Patch Embedding

Each 256-dimensional patch is projected into a
64-dimensional embedding.

Patch tensor:
(256, 256)

        ↓ Linear Projection

Patch embeddings:
(256, 64)

This converts the image patches into Transformer-compatible tokens.

Q, K and V

The embedded image tokens are projected into:

Q: (256, 64)
K: (256, 64)
V: (256, 64)

The attention scores are calculated using:

$$ S = \frac{QK^T}{\sqrt{d_k}} $$

Since there are 256 tokens:

QKᵀ → (256, 256)

Therefore, the final attention matrix has shape:

256 × 256
Attention Matrix

The attention matrix represents the relationship between image
patches.

Rows → Query patches
Columns → Key patches

Therefore:

Attention[i][j]

represents how much query patch i attends to key patch j.

Because there are 256 patches, every patch can interact with all
256 patches.

This demonstrates the global interaction property of self-attention.

Experimental Results
Tensor Shapes
Patch tensor:
torch.Size([256, 256])

Patch embedding:
torch.Size([256, 64])

Q:
torch.Size([256, 64])

K:
torch.Size([256, 64])

V:
torch.Size([256, 64])

Attention matrix:
torch.Size([256, 256])

Self-attention output:
torch.Size([256, 64])
Attention Row Verification

The minimum attention row sum was:

0.9999997616

The maximum attention row sum was:

1.0000001192

These values are effectively equal to 1.0, confirming that the
softmax attention weights are properly normalized.

Patch 128 Attention

The experiment visualizes the attention distribution for patch 128.

The highest attention weights were:

Rank	Patch	Row	Column	Weight
1	132	8	4	0.003943
2	185	11	9	0.003937
3	105	6	9	0.003936
4	89	5	9	0.003935
5	19	1	3	0.003933
6	88	5	8	0.003932
7	22	1	6	0.003931
8	32	2	0	0.003929
9	199	12	7	0.003928
10	134	8	6	0.003927
Interpretation

The attention map shows that patch 128 distributes its attention
across the other image patches.

However, the attention weights are relatively uniform.

This is expected because the patch embedding and Q/K/V projection
layers in this experiment are randomly initialized and have not been
trained on an image matching task.

Therefore, the attention visualization demonstrates the
self-attention mechanism, but it should not be interpreted as
learned semantic attention.

In a trained Transformer, the learned projections can produce
meaningful relationships between tokens.

Important Observation

Self-attention provides a mechanism through which one image region
can interact with distant image regions.

Unlike a purely local operation, the attention mechanism can directly
connect:

Patch A
   ↕
Patch B
   ↕
Patch C
   ↕
...
All image patches

This global interaction is particularly important for feature
matching, where corresponding structures may be located at different
positions in two images.

Positional Information

This experiment intentionally does not use positional encoding.

Therefore, the model is demonstrating the attention mechanism itself
without explicitly providing the Transformer with the spatial
coordinates of each patch.


# Task 2 — Cross-Attention

## Objective

Task 2 extends self-attention from a single image to interactions
between two different images.

In self-attention:

\[
Q, K, V
\]

come from the same image.

In cross-attention, the Query comes from one image while the Key
and Value come from another image.

```text
Image 1                    Image 2
   ↓                          ↓
   Q                         K, V
    \                         /
     \                       /
      ─── Cross-Attention ───
                ↓
        Updated representation

The objective is to understand how a patch in one image can interact
with patches in another image.

Mathematical Formulation

For Image 1 attending to Image 2:

$$ Q_1 = X_1W_Q $$ $$ K_2 = X_2W_K $$ $$ V_2 = X_2W_V $$

The cross-attention operation is:

$$ Attention(1\rightarrow2) = softmax \left( \frac{Q_1K_2^T}{\sqrt{d_k}} \right)V_2 $$

This produces an updated representation for Image 1 based on
information from Image 2.

The reverse direction can also be calculated:

$$ Attention(2\rightarrow1) = softmax \left( \frac{Q_2K_1^T}{\sqrt{d_k}} \right)V_1 $$
Dataset

The same HPatches v_woman sequence used in Stage 1 and Task 1B
is used:

Image 1:
datasets/hpatches-sequences-release/v_woman/1.ppm

Image 2:
datasets/hpatches-sequences-release/v_woman/2.ppm

The original image dimensions are:

Image 1: 767 × 1034
Image 2: 816 × 1232

Both images are resized to:

256 × 256
Image-to-Token Representation

Both images are divided into:

16 × 16 pixel patches

Therefore:

$$ 16 \times 16 = 256 $$

tokens are generated for each image.

Each patch contains:

$$ 16 \times 16 = 256 $$

pixel values.

The patch representation is therefore:

Image 1 patches: (256, 256)
Image 2 patches: (256, 256)

A linear layer projects each patch into a 64-dimensional embedding:

Image 1 tokens: (256, 64)
Image 2 tokens: (256, 64)
Cross-Attention Dimensions

For Image 1 → Image 2:

Q: (256, 64)
K: (256, 64)
V: (256, 64)

The similarity matrix is:

$$ QK^T $$

Therefore:

(256, 64) × (64, 256)
=
(256, 256)

The resulting cross-attention matrix is:

256 × 256

Each row represents a query patch from Image 1.

Each column represents a key patch from Image 2.

Therefore, every patch in Image 1 can interact with every patch in
Image 2.

Experimental Results
Image 1 → Image 2
Q shape:
torch.Size([256, 64])

K shape:
torch.Size([256, 64])

V shape:
torch.Size([256, 64])

Attention matrix:
torch.Size([256, 256])

Output:
torch.Size([256, 64])
Image 2 → Image 1
Attention matrix:
torch.Size([256, 256])

Output:
torch.Size([256, 64])
Attention Normalization

The attention weights are normalized using softmax.

The measured row sums were:

Minimum row sum:
0.999999821

Maximum row sum:
1.000000238

These values are effectively equal to 1.0.

This verifies that the cross-attention weights are properly
normalized.

Patch 128 Experiment

For Image 1 patch 128, the ten highest attention weights in Image 2
were:

Rank	Image 2 Patch	Row	Column	Weight
1	38	2	6	0.003948
2	66	4	2	0.003947
3	204	12	12	0.003945
4	117	7	5	0.003943
5	169	10	9	0.003941
6	102	6	6	0.003940
7	156	9	12	0.003937
8	106	6	10	0.003934
9	114	7	2	0.003933
10	124	7	12	0.003932
Interpretation

The cross-attention matrix demonstrates that a patch from Image 1
can interact with all patches in Image 2.

For example:

Image 1 Patch 128
        ↓
   ┌───────────────┐
   │ Patch 0       │
   │ Patch 1       │
   │ Patch 2       │
   │ ...           │
   │ Patch 255     │
   └───────────────┘
        Image 2

The attention weights determine how much information from each Image 2
patch contributes to the updated representation of Image 1 patch 128.

Important Limitation

The projection layers in this experiment are randomly initialized.

Therefore, the highest-attention patches should not be
interpreted as verified feature correspondences.

For example, the experiment does not establish that:

Image 1 Patch 128
        ↓
Image 2 Patch 38

is a true geometric correspondence.

The purpose of this task is to demonstrate the cross-attention
mechanism, not to perform trained feature matching.

Meaningful correspondences require learned feature representations
and geometric reasoning, which will be introduced later in the
project.

Self-Attention vs Cross-Attention
Property	Self-Attention	Cross-Attention
Query source	Same image	Image 1
Key source	Same image	Image 2
Value source	Same image	Image 2
Main purpose	Within-image interaction	Between-image interaction
Attention matrix	Token × Token	Image 1 tokens × Image 2 tokens

Conceptually:

Self-Attention:

Image A
   ↓
Q, K, V
   ↓
Relationships within Image A


Cross-Attention:

Image A              Image B
   ↓                    ↓
   Q                   K, V
    \                  /
     \                /
      Cross-Attention
            ↓
   Information from Image B