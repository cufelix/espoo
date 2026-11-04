"""Minimal MPO / tensor-train tools for weight matrices (numpy only).

A matrix W of shape (2**n, 2**m) is treated as a quantum operator on qubits:
row index i = (i_1..i_n) bits, column index j = (j_1..j_m) bits.
MPO site k carries (row bit block k, column bit block k). Bond dimension chi
plays the role of the entanglement cut between site groups.
"""
import numpy as np


def _factor(n_bits, sites):
    base, extra = divmod(n_bits, sites)
    return [base + (1 if k < extra else 0) for k in range(sites)]


def to_mpo(W, sites, chi=None, rel_tol=None):
    """TT-SVD of W into an MPO with `sites` cores.

    Truncate each bond to at most `chi`, and/or drop singular values whose
    discarded weight (squared) is below rel_tol**2 * ||W||^2 / (sites-1).
    Returns (cores, spectra) where spectra[k] are singular values at bond k.
    """
    R, C = W.shape
    nr, nc = int(np.log2(R)), int(np.log2(C))
    assert 2 ** nr == R and 2 ** nc == C, "dims must be powers of two"
    rb, cb = _factor(nr, sites), _factor(nc, sites)
    rd, cd = [2 ** b for b in rb], [2 ** b for b in cb]
    # reshape: (r1..rs, c1..cs) -> interleave (r1 c1 r2 c2 ...)
    T = W.reshape(rd + cd)
    perm = [x for k in range(sites) for x in (k, sites + k)]
    T = T.transpose(perm)
    phys = [rd[k] * cd[k] for k in range(sites)]
    T = T.reshape(phys)
    norm2 = float(np.sum(W.astype(np.float64) ** 2))
    budget = (rel_tol ** 2) * norm2 / max(sites - 1, 1) if rel_tol else 0.0
    cores, spectra = [], []
    left = 1
    M = T.reshape(left * phys[0], -1)
    for k in range(sites - 1):
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        keep = len(S)
        if budget > 0:
            tail = np.cumsum((S ** 2)[::-1])[::-1]  # tail[i] = sum_{j>=i} S_j^2
            ok = np.where(tail <= budget)[0]
            keep = int(ok[0]) if len(ok) else len(S)
            keep = max(keep, 1)
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
    T = cores[0]  # (1, r, c, b)
    for G in cores[1:]:
        T = np.tensordot(T, G, axes=([-1], [0]))
    T = T.squeeze(0).squeeze(-1)
    s = len(cores)
    # axes now: r1 c1 r2 c2 ... -> r1..rs c1..cs
    perm = [2 * k for k in range(s)] + [2 * k + 1 for k in range(s)]
    T = T.transpose(perm)
    R = int(np.prod([G.shape[1] for G in cores]))
    C = int(np.prod([G.shape[2] for G in cores]))
    return T.reshape(R, C)


def n_params(cores):
    return int(sum(G.size for G in cores))


def entanglement_entropy(S):
    p = S ** 2
    p = p / p.sum()
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())
