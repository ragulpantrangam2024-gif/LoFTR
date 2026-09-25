import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

torch.manual_seed(42)

NUM_FEATURES = 4
INPUT_DIM = 4
ATTENTION_DIM = 4


# ============================================================
# Create a small feature set
# ============================================================

X = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
)

print("=" * 60)
print("INPUT FEATURES")
print("=" * 60)

print(X)

print("\nInput shape:")
print(X.shape)


# ============================================================
# Linear projections
# ============================================================

W_Q = nn.Linear(INPUT_DIM, ATTENTION_DIM, bias=False)
W_K = nn.Linear(INPUT_DIM, ATTENTION_DIM, bias=False)
W_V = nn.Linear(INPUT_DIM, ATTENTION_DIM, bias=False)


Q = W_Q(X)
K = W_K(X)
V = W_V(X)


print("\n" + "=" * 60)
print("QUERY, KEY AND VALUE")
print("=" * 60)

print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)


# ============================================================
# Scaled dot-product attention
# ============================================================

d_k = K.shape[-1]

scores = torch.matmul(
    Q,
    K.transpose(-2, -1)
)

scaled_scores = scores / torch.sqrt(
    torch.tensor(float(d_k))
)


print("\n" + "=" * 60)
print("ATTENTION SCORES")
print("=" * 60)

print("\nQK^T:")
print(scores)

print("\nScaled scores:")
print(scaled_scores)


# ============================================================
# Softmax
# ============================================================

attention_weights = F.softmax(
    scaled_scores,
    dim=-1
)


print("\n" + "=" * 60)
print("ATTENTION WEIGHTS")
print("=" * 60)

print(attention_weights)


# ============================================================
# Weighted combination of values
# ============================================================

output = torch.matmul(
    attention_weights,
    V
)


print("\n" + "=" * 60)
print("SELF-ATTENTION OUTPUT")
print("=" * 60)

print(output)

print("\nOutput shape:")
print(output.shape)


# ============================================================
# Verify that every attention row sums to 1
# ============================================================

row_sums = attention_weights.sum(dim=-1)

print("\n" + "=" * 60)
print("ATTENTION WEIGHT CHECK")
print("=" * 60)

print("\nRow sums:")
print(row_sums)

print(
    "\nEvery row should be approximately equal to 1."
)