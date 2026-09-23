Before Task 2: README

We should now document Task 1 properly.

Your Task 1 README should explain:

1. Objective

Understand image correspondences using a real HPatches viewpoint sequence.

2. Dataset
HPatches
Sequence: v_woman
Reference: 1.ppm
Target: 2.ppm
Ground truth: H_1_2
3. Theory

Explain:

image coordinates
correspondence
homogeneous coordinates
homography
perspective transformation
forward transformation
inverse transformation
4. Implementation

Explain what the Python program does.

5. Results

Include:

task_01_ground_truth_correspondences.png

and the five point correspondences.

6. Verification

Report:

Mean round-trip error: 0 pixels
Maximum round-trip error: 0 pixels

with the clarification that this is an implementation consistency check, not matching accuracy.

7. Learning outcome

The most important takeaway:

A correspondence is a relationship between image locations representing the same physical scene point. HPatches provides the geometric ground truth needed to evaluate future feature-matching algorithms.