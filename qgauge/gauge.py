"""Gauge fixing: use EXACT symmetries of a transformer to rewrite its weights
into a basis where they are least 'entangled', before MPO compression.

Symmetries (function-preserving):
  * FFN hidden-neuron permutation P: W_up[P], b_up[P], W_down[:, P]
    (gated MLPs: also W_gate[P]). Works for any elementwise activation.
  * Attention head permutation (whole heads in Q,K,V and O).
  * Residual-stream orthogonal rotation (needs norm folding; SliceGPT/QuaRot).

The idea is borrowed from DMRG in quantum chemistry, where *orbital ordering*
(which degree of freedom sits on which MPS site) changes the required bond
dimension by orders of magnitude. A neuron's position fixes its mixed-radix
digit string, i.e. which 'qudit' states encode it.
"""
import numpy as np
from .tt import to_mpo, from_mpo, factorize


def _fiedler(X):
    n = X.shape[0]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    A = np.abs(Xn @ Xn.T)
    np.fill_diagonal(A, 0.0)
    d = A.sum(1)
    Dm = 1.0 / np.sqrt(d + 1e-12)
    L = np.eye(n) - Dm[:, None] * A * Dm[None, :]
    w, V = np.linalg.eigh(L)
    return V[:, 1] * Dm


def hierarchical_order(X, dims):
    """Recursive spectral partition following the mixed-radix digit tree:
    first split into dims[0] equal groups (most significant digit), each of
    those into dims[1] groups, etc."""
    def rec(ids, level):
        if level >= len(dims) - 1 or len(ids) <= dims[-1]:
            return list(ids)
        f = dims[level]
        order = ids[np.argsort(_fiedler(X[ids]), kind="stable")]
        out = []
        for part in np.array_split(order, f):
            out += rec(part, level + 1)
        return out
    return np.array(rec(np.arange(X.shape[0]), 0))


def _assign(cost):
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost)
    P = np.empty(cost.shape[0], dtype=int)
    P[cols] = rows  # slot s gets neuron P[s]
    return P


def refine_order(mats, sites, chi, P=None, iters=40, row_dims=None, tol=1e-5):
    """Assign-and-fit refinement of a shared row permutation for one or more
    matrices (e.g. [W_up, W_down.T]). Fit MPOs at bond dim chi to the
    currently ordered rows, then re-assign neurons to slots by an optimal
    linear assignment on the summed squared reconstruction cost (EM over
    permutations). Returns the best permutation found and its error."""
    mats = [np.asarray(m, dtype=np.float64) for m in mats]
    n = mats[0].shape[0]
    P = np.arange(n) if P is None else np.asarray(P)
    tot = sum(float((m ** 2).sum()) for m in mats)
    best_P, best = P, np.inf
    for _ in range(iters):
        cost = np.zeros((n, n))
        err2 = 0.0
        for m in mats:
            M = m[P]
            A = from_mpo(to_mpo(M, sites, chi=chi, row_dims=row_dims)[0])
            err2 += float(((M - A) ** 2).sum())
            cost += (m ** 2).sum(1)[:, None] - 2 * m @ A.T + (A ** 2).sum(1)[None, :]
        err = np.sqrt(err2 / tot)
        if err < best - tol:
            best_P, best = P, err
        else:
            break
        P = _assign(cost)
    return best_P, best


def anneal_order(mats, sites, chis=(2, 3, 4, 6), P=None, iters=40, row_dims=None):
    """Bond-dimension annealing (coarse-to-fine), like growing chi in DMRG."""
    for chi in chis:
        P, _ = refine_order(mats, sites, chi, P=P, iters=iters, row_dims=row_dims)
    return P


def gauge_fix_ffn(W_up, W_down, W_gate=None, sites=None, anneal=None):
    """Permute FFN neurons into a low-entanglement order.
    W_up/W_gate: (hidden, d_model); W_down: (d_model, hidden).
    Returns (perm) — apply as W_up[P], b_up[P], W_gate[P], W_down[:, P]."""
    mats = [W_up, W_down.T] + ([W_gate] if W_gate is not None else [])
    dims = factorize(W_up.shape[0], sites or 4)
    P = hierarchical_order(np.concatenate(mats, axis=1), dims)
    if anneal:
        P = anneal_order(mats, sites, chis=anneal, P=P, row_dims=dims)
    return P


def pca_tree_order(X, dims):
    """Scalable hierarchical ordering (no n x n matrices): at each level sort
    the group by its top principal component and cut into dims[level] equal
    chunks. Used for huge index sets such as vocabularies (50k+ tokens)."""
    def top_pc(Y):
        Y = Y - Y.mean(0)
        v = np.random.default_rng(0).standard_normal(Y.shape[1])
        for _ in range(30):
            v = Y.T @ (Y @ v); v /= np.linalg.norm(v) + 1e-12
        return Y @ v

    def rec(ids, level):
        if level >= len(dims) - 1 or len(ids) <= dims[-1]:
            return list(ids)
        f = dims[level]
        order = ids[np.argsort(top_pc(X[ids]), kind="stable")]
        out = []
        for part in np.array_split(order, f):
            out += rec(part, level + 1)
        return out
    return np.array(rec(np.arange(X.shape[0]), 0))
