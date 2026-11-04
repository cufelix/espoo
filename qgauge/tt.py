"""Minimal MPO / tensor-train tools for weight matrices (numpy only).

A matrix W of shape (R, C) with R = prod(row_dims), C = prod(col_dims) is
treated as an operator on a chain of `sites` 'qudits': row index
i = (i_1..i_s) in mixed radix row_dims, column index j = (j_1..j_s).
MPO site k carries (i_k, j_k). The bond dimension chi at cut k is the
Schmidt rank across that cut (entanglement between site groups).
"""
import numpy as np


def _pow2_split(n, sites):
    nb = int(round(np.log2(n)))
    assert 2 ** nb == n
    base, extra = divmod(nb, sites)
    return [2 ** (base + (1 if k < extra else 0)) for k in range(sites)]


def factorize(n, sites):
    """Split n into `sites` near-equal integer factors (largest first)."""
    if n & (n - 1) == 0:
        return _pow2_split(n, sites)
    primes, m, p = [], n, 2
    while m > 1:
        while m % p == 0:
            primes.append(p); m //= p
        p += 1
    dims = [1] * sites
    for q in sorted(primes, reverse=True):
        dims[int(np.argmin(dims))] *= q
    assert len(dims) == sites and np.prod(dims) == n
    return sorted(dims, reverse=True)


def _dims(W, sites, row_dims, col_dims):
    R, C = W.shape
    rd = list(row_dims) if row_dims else factorize(R, sites)
    cd = list(col_dims) if col_dims else factorize(C, sites)
    assert np.prod(rd) == R and np.prod(cd) == C and len(rd) == len(cd)
    return rd, cd


def to_mpo(W, sites, chi=None, rel_tol=None, row_dims=None, col_dims=None):
    """TT-SVD of W into an MPO. Truncate bonds to <= chi and/or to a
    discarded-weight budget rel_tol. Returns (cores, spectra)."""
    rd, cd = _dims(W, sites, row_dims, col_dims)
    s = len(rd)
    T = W.reshape(rd + cd)
    perm = [x for k in range(s) for x in (k, s + k)]
    T = T.transpose(perm)
    phys = [rd[k] * cd[k] for k in range(s)]
    norm2 = float(np.sum(W.astype(np.float64) ** 2))
    budget = (rel_tol ** 2) * norm2 / max(s - 1, 1) if rel_tol else 0.0
    cores, spectra, left = [], [], 1
    M = T.reshape(phys[0], -1)
    for k in range(s - 1):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        keep = len(S)
        if budget > 0:
            tail = np.cumsum((S ** 2)[::-1])[::-1]
            ok = np.where(tail <= budget)[0]
            keep = max(int(ok[0]) if len(ok) else len(S), 1)
        if chi is not None:
            keep = min(keep, chi)
        spectra.append(S)
        U, S, Vt = U[:, :keep], S[:keep], Vt[:keep]
        cores.append(U.reshape(left, rd[k], cd[k], keep))
        left = keep
        M = (S[:, None] * Vt).reshape(left * phys[k + 1], -1)
    cores.append(M.reshape(left, rd[-1], cd[-1], 1))
    return cores, spectra


def from_mpo(cores):
    T = cores[0]
    for G in cores[1:]:
        T = np.tensordot(T, G, axes=([-1], [0]))
    T = T.squeeze(0).squeeze(-1)
    s = len(cores)
    perm = [2 * k for k in range(s)] + [2 * k + 1 for k in range(s)]
    T = T.transpose(perm)
    R = int(np.prod([G.shape[1] for G in cores]))
    C = int(np.prod([G.shape[2] for G in cores]))
    return T.reshape(R, C)


def mpo_approx(W, sites, chi, **kw):
    cores, _ = to_mpo(W, sites, chi=chi, **kw)
    return from_mpo(cores), n_params(cores)


def n_params(cores):
    return int(sum(G.size for G in cores))


def entanglement_entropy(S):
    p = S ** 2
    p = p / p.sum()
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())
