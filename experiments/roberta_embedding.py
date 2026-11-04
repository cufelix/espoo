"""Experiment 3: token-embedding matrix (50265 x 768 = 31% of RoBERTa-base).

Vocabulary order is an exact gauge symmetry (re-index tokens). We pad the
vocab to 51200 rows with zeros and compare MPO error for: BPE order as
shipped, random order, PCA-tree gauge (+ assign-and-fit refinement), and
truncated SVD at equal parameter count.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qgauge.tt import to_mpo, from_mpo, n_params, factorize
from qgauge.gauge import pca_tree_order

SITES = int(os.environ.get("SITES", 5))
sd = np.load(sys.argv[1])
E = sd["curated_encoder.embeddings.inner.word_embeddings.weight"].astype(np.float64)
V, D = E.shape
NP = 51200
rd = factorize(NP, SITES); cd = factorize(D, SITES)
print(f"E {V}x{D}, padded to {NP}; row dims {rd}, col dims {cd}")
rng = np.random.default_rng(0)


Ep = np.vstack([E, np.zeros((NP - V, D))])  # padding rows are zero


def err(M, chi):
    cores, _ = to_mpo(M, SITES, chi=chi, row_dims=rd, col_dims=cd)
    A = from_mpo(cores)
    return np.linalg.norm(M - A) / np.linalg.norm(M), n_params(cores)


t = time.time(); Pg = pca_tree_order(Ep, rd); print(f"gauge search {time.time()-t:.1f}s")
variants = {"bpe": Ep, "random": Ep[rng.permutation(NP)], "gauged": Ep[Pg]}
S = np.linalg.svd(E, compute_uv=False); tot = (S ** 2).sum()
print(f"{'chi':>4} {'%params':>8} | {'bpe':>6} {'random':>6} {'gauged':>6} | {'svd':>6}")
for chi in [16, 32, 64, 128, 256]:
    r = {n: err(m, chi) for n, m in variants.items()}
    p = r["gauged"][1]; k = max(1, p // (V + D))
    print(f"{chi:>4} {100*p/E.size:>7.2f}% | " + " ".join(f"{r[n][0]:6.3f}" for n in variants) + f" | {np.sqrt((S[k:]**2).sum()/tot):6.3f}")
