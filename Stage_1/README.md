Stage 1 — Classical Feature Matching Fundamentals

Stage 1 establishes the classical computer-vision feature-matching pipeline that will later serve as a baseline for the LoFTR project.

The implementation uses a real HPatches image sequence rather than synthetic images. Ground-truth homography data supplied by HPatches is used to evaluate the geometric consistency of feature matches.

Dataset

Dataset: HPatches
Sequence: v_woman
Reference image: 1.ppm
Target image: 2.ppm
Ground-truth homography: H_1_2

The HPatches dataset is stored locally under:

datasets/hpatches-sequences-release/

The dataset is excluded from Git using .gitignore.

Task 1 — Image Correspondences and Homography

Objective

Understand how corresponding image points can be related using a known geometric transformation.

Concepts

Task 1 introduces:

Image coordinates

Point correspondences

Homogeneous coordinates

Homography

Perspective transformation

Forward transformation

Inverse transformation

For a point in image 1,

[
p_1 =
\begin{bmatrix}
x_1\
y_1\
1
\end{bmatrix}
]

the HPatches ground-truth homography maps it to image 2:

[
p_2 \sim H_{1\rightarrow2}p_1
]

After homogeneous normalization:

[
x_2 = \frac{x'}{w'}, \qquad
y_2 = \frac{y'}{w'}
]

Implementation

task_01_image_correspondences.py:

Loads the two HPatches images.

Loads H_1_2.

Defines five sample points in image 1.

Projects the points into image 2 using the homography.

Visualizes the corresponding points.

Uses the inverse homography to perform a round-trip verification.

Sample correspondences

Image 1

Image 2 predicted by H_1_2

(250, 300)

(437.48, 312.58)

(500, 300)

(624.85, 327.76)

(700, 500)

(750.59, 475.63)

(400, 600)

(545.77, 546.79)

(800, 200)

(818.13, 276.29)

Verification

The forward transformation followed by the inverse transformation produced:

Mean round-trip error: 0 pixels
Maximum round-trip error: 0 pixels

This is an implementation consistency check. It does not measure feature-matching accuracy.

Results

Stage_1/results/task_01_ground_truth_correspondences.png
Stage_1/results/task_01_image_pair.png

Learning outcome

A correspondence represents two image locations that refer to the same physical scene point.

The HPatches homography gives us the geometric ground truth required to evaluate feature-matching algorithms in later tasks.

Task 2 — SIFT Feature Matching

Objective

Implement and evaluate a classical local feature-matching pipeline using SIFT (Scale-Invariant Feature Transform).

The purpose is to establish a classical baseline before implementing ORB and eventually comparing both classical methods with LoFTR.

SIFT pipeline

Image
  ↓
SIFT keypoint detection
  ↓
SIFT descriptor extraction
  ↓
128-dimensional descriptors
  ↓
Brute-force descriptor matching
  ↓
Lowe ratio test
  ↓
Candidate correspondences
  ↓
Ground-truth geometric evaluation using H_1_2

Main concepts

Scale-space

SIFT searches for stable image structures across different scales by constructing progressively blurred versions of the image.

Difference of Gaussians

SIFT uses the Difference of Gaussians (DoG) to identify candidate keypoints:

[
D(x,y,\sigma)=L(x,y,k\sigma)-L(x,y,\sigma)
]

Keypoint orientation

A dominant local gradient orientation is assigned to each keypoint. This helps make the feature representation more robust to image rotation.

SIFT descriptor

The classical SIFT descriptor is a 128-dimensional vector constructed from local gradient information.

In simplified form:

4 × 4 spatial cells
×
8 orientation bins
=
128 dimensions

Descriptor matching

The implementation uses a brute-force matcher with the Euclidean/L2 distance.

For each descriptor in image 1, the two nearest descriptors in image 2 are retrieved.

Lowe ratio test

The best and second-best descriptor distances are compared:

[
r = rac{d_1}{d_2}
]

A match is retained when:

[
d_1 < 0.75d_2
]

The threshold used in this experiment is:

0.75

Task 2 — Experimental Results

HPatches v_woman
Image 1: 1.ppm
Image 2: 2.ppm
Ground truth: H_1_2

Feature detection

Image 1 keypoints: 1856
Image 2 keypoints: 7751

Descriptor shapes:

Image 1: (1856, 128)
Image 2: (7751, 128)

Matching

Total KNN matches: 1856
Lowe ratio threshold: 0.75
Good matches after ratio test: 513

Ground-truth evaluation

For every SIFT match, the image-1 point is projected using H_1_2 and compared with the SIFT matched point in image 2.

H_{1\rightarrow2}p_{1,i}^{SIFT}

ight|_2
]

Results:

Number of good matches: 513
Mean error:              17.6902 pixels
Median error:             0.3596 pixels
Minimum error:            0.0169 pixels
Maximum error:          879.9711 pixels

Using a 3-pixel threshold:

Ground-truth-consistent matches: 481 / 513
Ground-truth-consistent match rate: 93.76%

This is a ground-truth-consistent match rate, not a RANSAC inlier percentage.

Error distribution

Geometric error

Number of matches

0–1 px

433

1–3 px

48

3–5 px

3

5–10 px

2

>10 px

27

Result files:

Stage_1/results/task_02_sift_keypoints_image1.png
Stage_1/results/task_02_sift_keypoints_image2.png
Stage_1/results/task_02_sift_good_matches.png
Stage_1/results/task_02_sift_ground_truth_consistent_matches.png
Stage_1/results/task_02_sift_geometric_outliers.png
Stage_1/results/task_02_sift_results.txt

Task 3 — ORB Feature Matching

Objective

Implement and evaluate ORB (Oriented FAST and Rotated BRIEF) using the same HPatches image pair and the same ground-truth homography used for SIFT.

This creates a consistent classical baseline before moving toward RANSAC and eventually LoFTR.

ORB pipeline

Image
  ↓
FAST-based keypoint detection
  ↓
Orientation estimation
  ↓
Rotated BRIEF-style binary descriptor
  ↓
Hamming-distance matching
  ↓
Lowe ratio test
  ↓
Candidate correspondences
  ↓
Ground-truth geometric evaluation using H_1_2

Main concepts

FAST-based detection

ORB builds on FAST-style corner detection to obtain efficient local features.

Orientation

ORB adds an orientation component to the FAST keypoints. This allows the BRIEF sampling pattern to be rotated according to the keypoint orientation.

Binary descriptor

ORB uses a binary BRIEF-style descriptor. In the OpenCV implementation used here, each descriptor has:

32 bytes
=
256 bits

Therefore the descriptor arrays have the form:

(N, 32)

rather than SIFT's:

(N, 128)

Hamming distance

Because ORB descriptors are binary, matching uses Hamming distance rather than the L2 distance used for SIFT.

Lowe ratio test

The same ratio threshold used for SIFT is retained:

0.75

This keeps the evaluation protocol consistent between the two classical baselines.

Task 3 — Experimental Results

The experiment was performed on:

HPatches v_woman
Image 1: 1.ppm
Image 2: 2.ppm
Ground truth: H_1_2

Feature detection

Image 1 keypoints: 4964
Image 2 keypoints: 5000

Descriptor arrays:

Image 1: (4964, 32)
Image 2: (5000, 32)

The second dimension of 32 represents 32 bytes, corresponding to a 256-bit binary descriptor.

Matching

Total KNN matches: 4964
Lowe ratio threshold: 0.75
Good matches after ratio test: 437

Ground-truth geometric evaluation

The same evaluation procedure used for SIFT is applied to ORB:

H_{1\rightarrow2}p_{1,i}^{ORB}

ight|_2
]

Results:

Number of good matches: 437
Mean error:              22.5343 pixels
Median error:             0.9430 pixels
Minimum error:            0.0400 pixels
Maximum error:          765.8625 pixels

Using the same 3-pixel threshold:

Ground-truth-consistent matches: 393 / 437
Ground-truth-consistent match rate: 89.93%

Error distribution

Geometric error

Number of matches

0–1 px

238

1–3 px

155

3–5 px

17

5–10 px

2

>10 px

25

Therefore:

238 + 155 = 393

matches are within the 3-pixel ground-truth threshold.

Task 3 result files

Stage_1/results/task_03_orb_keypoints_image1.png
Stage_1/results/task_03_orb_keypoints_image2.png
Stage_1/results/task_03_orb_good_matches.png
Stage_1/results/task_03_orb_ground_truth_consistent_matches.png
Stage_1/results/task_03_orb_geometric_outliers.png
Stage_1/results/task_03_orb_results.txt

SIFT vs ORB — Current Single-Pair Experiment

Metric

SIFT

ORB

Image 1 keypoints

1,856

4,964

Image 2 keypoints

7,751

5,000

Descriptor representation

128-D float

256-bit binary

Matching distance

L2

Hamming

Good matches

513

437

Median error

0.3596 px

0.9430 px

Matches ≤3 px

481

393

Ground-truth-consistent rate

93.76%

89.93%

These numbers describe one viewpoint pair only. They should not be interpreted as a general benchmark of SIFT and ORB. Later evaluation across multiple HPatches sequences will provide a broader comparison.

Current Stage 1 Status

Task 1 — Image correspondences       COMPLETED
Task 2 — SIFT matching                COMPLETED
Task 3 — ORB matching                 COMPLETED
Task 4 — RANSAC geometric verification NEXT

Why RANSAC is next

So far, the ground-truth homography has been used as an external evaluation reference.

The next task introduces a different idea:

Feature matches
      ↓
Candidate correspondences
      ↓
RANSAC
      ↓
Estimate geometric model
      ↓
Separate inliers and outliers

A real feature-matching system does not normally have access to the HPatches ground-truth homography during operation. RANSAC therefore lets us study robust geometric model estimation independently of the provided ground truth.

Connection to LoFTR

The classical baseline established in Stage 1 is:

SIFT / ORB
     ↓
local keypoints
     ↓
local descriptors
     ↓
descriptor matching
     ↓
geometric verification

The later LoFTR pipeline will investigate a different approach:

Image pair
     ↓
deep feature extraction
     ↓
Transformer self/cross-attention
     ↓
coarse matching
     ↓
fine matching
     ↓
correspondences

Understanding the classical pipeline first makes the motivation for detector-free Transformer-based matching easier to understand.

Stage 1 Learning Goal

By the end of Stage 1, the goal is to understand:

Image geometry

Point correspondences

Homography

Local feature detection

Local feature description

Descriptor matching

Binary vs floating-point descriptors

Hamming vs L2 distance

Ratio-test filtering

Ground-truth geometric evaluation

RANSAC-based geometric verification

The planned progression is:

Image geometry
      ↓
SIFT
      ↓
ORB
      ↓
RANSAC
      ↓
Transformer fundamentals
      ↓
Simplified LoFTR
      ↓
Official pretrained LoFTR
      ↓
SIFT vs ORB vs LoFTR evaluation