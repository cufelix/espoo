"""Gauge fixing: use EXACT symmetries of a transformer to rewrite its weights
into a basis where they are least 'entangled', before MPO compression.

Symmetries used (function-preserving, zero accuracy cost):
  * FFN hidden-neuron permutation: W_up[P], W_down[:, P] (also for gated MLPs,
    applying P to gate and up together).
  * Attention head permutation (permute whole heads in Q,K,V and O).
  * Residual-stream orthogonal rotation Q (computational invariance, as used by
    SliceGPT / QuaRot) - needs RMSNorm scale folded into adjacent weights.

This file implements the permutation gauge. The core idea is borrowed from
DMRG in quantum chemistry, where *orbital ordering* (choosing which physical
degree of freedom sits on which MPS site) changes the required bond dimension
by orders of magnitude. Here, a neuron's position defines its bit string, so
ordering neurons hierarchically by similarity puts correlated neurons on
nearby leaves of the binary tree -> low entanglement across MPO cuts.
"""
import numpy as np


def _fiedler_split(X):
    """Split rows of X in two equal halves by the Fiedler vector of the
    cosine-similarity graph (spectral bisection)."""
    n = X.shape[0]
    if n <= 2:
        return np.arange(n)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    A = np.abs(Xn @ Xn.T)
    np.fill_diagonal(A, 0.0)
    d = A.sum(1)
    Dm = 1.0 / np.sqrt(d + 1e-12)
    L = np.eye(n) - Dm[:, None] * A * Dm[None, :]
    w, V = np.linalg.eigh(L)
    f = V[:, 1] * Dm
    return np.argsort(f, kind="stable")


def hierarchical_order(X, min_block=2):
    """Recursive spectral bisection -> permutation of rows of X (len power of 2).

    The top-level split decides the most significant bit, the next split the
    next bit, etc. Neurons sharing long bit prefixes end up similar.
    """
    idx = np.arange(X.shape[0])

    def rec(ids):
        if len(ids) <= min_block:
            return list(ids)
        order = _fiedler_split(X[ids])
        h = len(ids) // 2
        a, b = ids[order[:h]], ids[order[h:]]
        return rec(a) + rec(b)

    return np.array(rec(idx))


def gauge_fix_ffn(W_up, W_down, W_gate=None):
    """Return permuted (W_up, W_down, W_gate, perm) computing the same function.

    W_up: (hidden, d_model), W_down: (d_model, hidden), W_gate like W_up.
    Neuron feature = concatenation of its input and output weights.
    """
    feats = [W_up, W_down.T] + ([W_gate] if W_gate is not None else [])
    P = hierarchical_order(np.concatenate(feats, axis=1))
    out_gate = W_gate[P] if W_gate is not None else None
    return W_up[P], W_down[:, P], out_gate, P


def refine_order(W, sites, chi, iters=10, init=None, verbose=False):
    """Alternating 'assign-and-fit' refinement of the neuron permutation.

    Repeat: fit an MPO with bond dim `chi` to the currently ordered rows, then
    re-assign every neuron (row) to the slot whose reconstructed row it matches
    best, via an optimal linear assignment (Hungarian). This is EM / k-means
    over permutations; each step can only keep or lower the fit error of the
    assignment step. Returns permutation P such that W[P] is MPO-friendly.
    """
    from scipy.optimize import linear_sum_assignment
    from .tt import to_mpo, from_mpo

    P = np.arange(W.shape[0]) if init is None else np.asarray(init)
    best_P, best_err = P, np.inf
    for t in range(iters):
        M = W[P]
        A = from_mpo(to_mpo(M, sites, chi=chi)[0])
        err = np.linalg.norm(M - A) / np.linalg.norm(M)
        if verbose:
            print(f"  refine iter {t}: rel err {err:.4f}")
        if err < best_err - 1e-6:
            best_P, best_err = P, err
        elif t > 0:
            break
        # cost[n, s] = ||W[n] - A[s]||^2 (neuron n into slot s)
        cost = (W ** 2).sum(1)[:, None] - 2 * W @ A.T + (A ** 2).sum(1)[None, :]
        rows, cols = linear_sum_assignment(cost)
        newP = np.empty_like(P)
        newP[cols] = rows
        P = newP
    return best_P


def anneal_order(W, sites, chis=(2, 3, 4, 6), iters=40, init=None):
    """Bond-dimension annealing: refine the permutation at small chi first
    (coarse structure), then increase chi. Mirrors how DMRG grows bond
    dimension gradually to avoid local minima."""
    P = np.arange(W.shape[0]) if init is None else np.asarray(init)
    for chi in chis:
        P = refine_order(W, sites, chi, iters=iters, init=P)
    return P
