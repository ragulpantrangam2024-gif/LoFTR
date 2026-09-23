Stage 1 --- Classical Feature Matching Fundamentals

Stage 1 establishes the classical computer-vision feature-matching
pipeline that will later serve as a baseline for the LoFTR project.

The implementation uses a real HPatches image sequence rather than
synthetic images. Ground-truth homography data supplied by HPatches is
used to evaluate the geometric consistency of feature matches.

Dataset

Dataset: HPatches
Sequence: v_woman
Reference image: 1.ppm
Target image: 2.ppm
Ground-truth homography: H_1_2

The HPatches dataset is stored locally under:

datasets/hpatches-sequences-release/

The dataset is excluded from Git using .gitignore.

Task 1 --- Image Correspondences and Homography

Objective

Understand how corresponding image points can be related using a known
geometric transformation.

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

[ p_1 =

\begin{bmatrix}
x_1\\
y_1\\
1
\end{bmatrix}

]

the HPatches ground-truth homography maps it to image 2:

[ p_2 \sim {=tex}H_{1\rightarrow2{=tex}}p_1 ]

After homogeneous normalization:

[ x_2 = \frac{x'}{w'}{=tex}, \qquad{=tex} y_2 =
\frac{y'}{w'}{=tex} ]

Implementation

task_01_image_correspondences.py:

Loads the two HPatches images.

Loads H_1_2.

Defines five sample points in image 1.

Projects the points into image 2 using the homography.

Visualizes the corresponding points.

Uses the inverse homography to perform a round-trip verification.

Sample correspondences

Image 1        Image 2 predicted by H_1_2

(250, 300)   (437.48, 312.58)
(500, 300)   (624.85, 327.76)
(700, 500)   (750.59, 475.63)
(400, 600)   (545.77, 546.79)
(800, 200)   (818.13, 276.29)

Verification

The forward transformation followed by the inverse transformation
produced:

Mean round-trip error: 0 pixels
Maximum round-trip error: 0 pixels

This is an implementation consistency check. It does not measure
feature-matching accuracy.

Results

Stage_1/results/task_01_ground_truth_correspondences.png
Stage_1/results/task_01_image_pair.png

Learning outcome

A correspondence represents two image locations that refer to the same
physical scene point.

The HPatches homography gives us the geometric ground truth required to
evaluate feature-matching algorithms in later tasks.

Task 2 --- SIFT Feature Matching

Objective

Implement and evaluate a classical local feature-matching pipeline using
SIFT (Scale-Invariant Feature Transform).

The purpose is to establish a classical baseline before implementing ORB
and eventually comparing both classical methods with LoFTR.

SIFT pipeline

The implemented pipeline is:

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

1. Scale-space

SIFT searches for stable image structures across different scales by
constructing progressively blurred versions of the image.

2. Difference of Gaussians

SIFT uses the Difference of Gaussians (DoG) to identify candidate
keypoints:

[ D(x,y,\sigma{=tex}) = L(x,y,k\sigma{=tex})-L(x,y,\sigma{=tex})
]

3. Keypoint orientation

A dominant local gradient orientation is assigned to each keypoint. This
helps make the feature representation more robust to image rotation.

4. SIFT descriptor

The classical SIFT descriptor is a 128-dimensional vector
constructed from local gradient information.

In simplified form:

4 × 4 spatial cells
×
8 orientation bins
=
128 dimensions

5. Descriptor matching

The implementation uses a brute-force matcher with the Euclidean/L2
distance.

For each descriptor in image 1, the two nearest descriptors in image 2
are retrieved.

6. Lowe ratio test

The best and second-best descriptor distances are compared:

[ r = \frac{d_1}{d_2}{=tex} ]

A match is retained when:

[ d_1 < 0.75d_2 ]

The threshold used in this experiment is:

0.75

Task 2 --- Experimental Results

The experiment was performed on:

HPatches v_woman
Image 1: 1.ppm
Image 2: 2.ppm
Ground truth: H_1_2

Feature detection

Image 1 keypoints: 1856
Image 2 keypoints: 7751

The descriptors have the expected SIFT dimensionality:

Image 1: (1856, 128)
Image 2: (7751, 128)

Matching

Total KNN matches: 1856
Lowe ratio threshold: 0.75
Good matches after ratio test: 513

The ratio test therefore reduced the initial candidate set to 513
retained matches.

Ground-Truth Geometric Evaluation

The SIFT matches are evaluated against the HPatches homography H_1_2.

For every SIFT match:

SIFT point in image 1
        ↓
     H_1_2
        ↓
expected point in image 2
        ↓
compare with SIFT's matched point in image 2

The geometric error is:

[ e_i = \left{=tex}| p_{2,i}^{SIFT} -
H_{1\rightarrow2{=tex}}p_{1,i}^{SIFT} \right{=tex}|_2 ]

where homogeneous normalization is applied after the homography
transformation.

Results

Number of good matches: 513

Mean error:              17.6902 pixels
Median error:             0.3596 pixels
Minimum error:            0.0169 pixels
Maximum error:          879.9711 pixels

A match is considered ground-truth-consistent when its geometric
error is at most 3 pixels.

Ground-truth-consistent matches: 481 / 513
Ground-truth-consistent match rate: 93.76%

Important terminology

The 93.76% value is called the:

Ground-truth-consistent match rate

It is not called a RANSAC inlier percentage because RANSAC has not
yet been applied in Stage 1 Task 2.

RANSAC-based geometric verification will be introduced separately.

Error Distribution

The 513 ratio-test matches are distributed as follows:

Geometric error     Number of matches

0--1 px                           433
1--3 px                            48
3--5 px                             3
5--10 px                            2
>10 px                            27

Therefore:

433 + 48 = 481

matches are within the 3-pixel ground-truth threshold.

The median error of approximately 0.36 pixels is much smaller than
the mean error because a relatively small number of matches have very
large geometric errors.

Task 2 Results Files

The following files are generated under:

Stage_1/results/

task_02_sift_keypoints_image1.png
task_02_sift_keypoints_image2.png
task_02_sift_good_matches.png
task_02_sift_ground_truth_consistent_matches.png
task_02_sift_geometric_outliers.png
task_02_sift_results.txt

Keypoint visualizations

These images show the SIFT keypoints detected in each image, including
their scale and orientation information.

Good matches

task_02_sift_good_matches.png visualizes the matches remaining after
the Lowe ratio test.

Ground-truth-consistent matches

task_02_sift_ground_truth_consistent_matches.png contains only matches
whose geometric error with respect to H_1_2 is at most 3 pixels.

Geometric outliers

task_02_sift_geometric_outliers.png contains the matches whose error
exceeds the 3-pixel threshold.

Current Stage 1 Status

Task 1 — Image correspondences       COMPLETED
Task 2 — SIFT matching                COMPLETED
Task 3 — ORB matching                 NEXT
Task 4 — RANSAC geometric verification

The same HPatches image pair and ground-truth homography will be used
for the classical SIFT and ORB baselines so that their results can later
be compared under the same geometric evaluation protocol.

Why SIFT Is Important for the LoFTR Project

SIFT provides a useful classical baseline for understanding local
feature matching.

Its pipeline depends on:

detected keypoints
        +
hand-crafted local descriptors
        +
descriptor distance
        +
explicit matching

LoFTR takes a substantially different approach:

dense image features
        +
Transformer self/cross-attention
        +
coarse-to-fine matching
        +
detector-free correspondence estimation

Understanding SIFT first makes the motivation for LoFTR much clearer.

Stage 1 Learning Goal

By the end of Stage 1, the goal is to understand the classical
feature-matching pipeline before moving toward Transformer-based
detector-free matching.

The progression is:

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