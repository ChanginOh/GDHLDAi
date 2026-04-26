"""
=========================================================================
Improved GDHLDA for Linear Dimensionality Reduction on the Stiefel Manifold
=========================================================================

This script implements an improved version of GDHLDA
(gradient descent-based multiclass harmonic linear discriminant analysis)
for supervised dimensionality reduction. This method computes an
orthonormal basis W on the Stiefel manifold by minimizing a harmonic
trace-ratio objective that combines a harmonic within-class scatter
matrix with pairwise between-class scatter matrices.

The implementation includes the main algorithmic improvements introduced
in the improved GDHLDA framework:
  - random initialization
  - Riemannian gradient normalization
  - adaptive step size with random relaxation
  - orthogonality-based conditional retraction
  - QR/SVD retraction options

Input:
  A CSV file containing feature vectors and class labels

Output:
  W      - projection matrix with orthonormal columns
  Jvals  - objective values across iterations

This code was written to reproduce the improved GDHLDA procedure
described in the following work:

  Oh, M., Oh, C., and Tabassum, E.
  "Covalent: Interpretable and Discriminative Collective Variables
  Reveal Ligand-Dependent Switching in Human Cellular Retinol-Binding
  Protein 2"
  Journal of Chemical Theory and Computation, 2025
  DOI: 10.1021/acs.jctc.5c01402
=========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# User Setting
# ==============================
r = 2                           # Number of eigenvectors
eta = 1e-3                      # Initial step size
q = np.inf                      # Upper bound on step size growth
a = -0.9                        # Lower bound of relaxation parameter
b = 1.5                         # Upper bound of relaxation parameter
stol = np.finfo(float).eps      # Division tolerance
otol = np.finfo(float).eps      # Orthogonality criterion
ftol = np.finfo(float).eps      # Convergence criterion (function value)
gtol = np.finfo(float).eps      # Convergence criterion (gradient)
etamin = np.finfo(float).eps    # Convergence criterion (step size)
miniter = 100                   # Minimum iterations
maxiter = 1000                  # Maximum iterations

# ==============================
# Helper functions
# ==============================
def symm(W):
    """Symmetric matrix"""
    return 0.5 * (W + W.T)

def proj(W, Z):
    """Projection of Z to tangent space at W"""
    return Z - W @ symm(W.T @ Z)

def retr(W, xi, flag="SVD"):
    """Retraction onto the Stiefel manifold"""
    flag = flag.upper()
    if flag == "QR":
        # QR retraction (fast)
        Q, R = np.linalg.qr(W + xi, mode='reduced')
        S = np.sign(np.diag(R))
        S[S == 0] = 1
        return Q @ np.diag(S)
    elif flag == "SVD":
        # SVD retraction (robust)
        U, _, Vh = np.linalg.svd(W + xi, full_matrices=False)
        return U @ Vh
    else:
        raise ValueError(f'Unknown flag: {flag}')

def J(W, nc, ncs, mu, Sw, stol):
    """Objective function J"""
    f = 0
    trSw = np.trace(W.T @ Sw @ W)
    for k in range(nc - 1):
        for l in range(k + 1, nc):
            diff = mu[k, :] - mu[l, :]
            Bkl = np.outer(diff, diff)
            trBkl = np.trace(W.T @ Bkl @ W)
            f += ncs[k] * ncs[l] * trSw / max(trBkl, stol)
    return f

# ==============================
# Dataset Preprocessing
# ==============================
# Load dataset
# df = pd.read_csv('example.csv')
df = pd.read_csv('synthetic_3d_clusters.csv')
X = df.iloc[:, 0:3].values  # features
Y = df.iloc[:, 3].values    # labels

# Find size of dataset (number of samples and number of features)
n, p = X.shape

# Find unique classes and number of classes
c = np.unique(Y)
nc = len(c)

# Set empty arrays for class sizes and class mean vectors
ncs = np.zeros(nc)
mu = np.zeros((nc, p))

# Compute Sw with size of class dataset and mean vectors
Sw = np.zeros((p, p))
for k in range(nc):
    ind = np.where(Y == c[k])[0]
    ncs[k] = len(ind)                       # class sizes
    mu[k, :] = np.mean(X[ind, :], axis=0)   # class mean vectors
    
    Sk = np.zeros((p, p))
    for i in range(int(ncs[k])):
        Xc = X[ind[i], :] - mu[k, :]
        Sk += np.outer(Xc, Xc)

    Sw += np.linalg.pinv(Sk)

Sw = np.linalg.pinv(Sw)

# ==============================
# Riemmanian Gradient Descent
# ==============================
# Initialize W on Stiefel
W = retr(np.random.randn(p, r), np.zeros((p, r)), "QR")

# Preallocate Jvals to record function values
Jvals = np.full(maxiter + 1, np.nan)

# Store initial function value
Jvals[0] = J(W, nc, ncs, mu, Sw, stol)

# Compute initial Euclidean gradient
grad = np.zeros((p, r))
trSw = np.trace(W.T @ Sw @ W)
for k in range(nc - 1):
    for l in range(k + 1, nc):
        diff = mu[k, :] - mu[l, :]
        Bkl = np.outer(diff, diff)
        trBkl = np.trace(W.T @ Bkl @ W)
        grad += 2 * ncs[k] * ncs[l] * (trBkl * (Sw @ W) - trSw * (Bkl @ W)) / max(trBkl**2, stol)

# Compute and normalize initial Riemannian gradient
xi = proj(W, grad)
xih = xi / max(np.linalg.norm(xi, 'fro'), stol)

# Main loop
for counter in range(maxiter):
    # Store old state
    xio = xi.copy()         # old Riemannian gradient
    xiho = xih.copy()       # old normalized Riemannian gradient
    Jo = Jvals[counter]     # old function value
    
    # Update W on tangent space using gradient descent method
    Wtrial = W - eta * xih
    
    # Check orthogonality for retraction
    org = np.linalg.norm(Wtrial.T @ Wtrial - np.eye(r), 'fro') / np.linalg.norm(np.eye(r), 'fro')
    if org > otol:
        W = retr(W, -eta * xih, "QR")
    else:
        W = Wtrial.copy()
        
    # Store current function value
    Jw = J(W, nc, ncs, mu, Sw, stol)
    Jvals[counter + 1] = Jw
    
    # Display progress
    if counter % 50 == 0:
        print(f'Iteration: {counter + 1}, J: {Jw:.15e}, Stepsize: {eta:.5e}')
        
    # Compute new Euclidean gradient
    grad = np.zeros((p, r))
    trSw = np.trace(W.T @ Sw @ W)
    for k in range(nc - 1):
        for l in range(k + 1, nc):
            diff = mu[k, :] - mu[l, :]
            Bkl = np.outer(diff, diff)
            trBkl = np.trace(W.T @ Bkl @ W)
            grad += 2 * ncs[k] * ncs[l] * (trBkl * (Sw @ W) - trSw * (Bkl @ W)) / max(trBkl**2, stol)
            
    # Normalize new Riemannian gradient
    xi = proj(W, grad)
    xih = xi / max(np.linalg.norm(xi, 'fro'), stol)
    
    # Perform (extrinsic) vector transport via projection transport for step adaptation
    xioT = proj(W, xio)     # transport old Riemannian gradient to tangent space
    xihoT = proj(W, xiho)   # transport old normalized Riemannian gradient to tangent space
    
    # Perform random step adaptation with relaxation
    alpha = a + (b - a) * np.random.rand()
    numerator = (1 + alpha) * abs(np.trace(xioT.T @ xihoT))
    denominator = abs(np.trace(xioT.T @ xihoT) - np.trace(xi.T @ xihoT))
    denominator = max(denominator, stol)
    
    if numerator > q * denominator:
        eta = np.sqrt(q) * eta
    else:
        eta = np.sqrt(numerator / denominator) * eta
        
    # Check convergence
    if counter + 1 >= miniter:
        if abs(Jw - Jo) / max(1, abs(Jo)) < ftol:
            break
        elif np.linalg.norm(xi, 'fro') < gtol:
            break
        elif eta < etamin:
            break

# Display final results
print(f'[Iteration] {counter + 1}, [J] {Jw:.15e}, [Stepsize] {eta:.5e}')

# Trim Jvals to actual length
Jvals = Jvals[:counter + 2]

# ==============================
# Results
# ==============================
# Export W
# np.savetxt('W.csv', W, delimiter=',')

# Draw convergence curve
plt.figure(figsize=(8, 5))
plt.plot(Jvals, linewidth=2.5)
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Convergence Curve")
plt.grid(True)
plt.show()