"""Experiment 1 (synthetic): does permutation gauge fixing recover low
MPO bond dimension after neuron order has been scrambled?

Setup: a 'planted' FFN up-projection W (hidden x d_model) built as a random
MPO with small bond dimension + noise. Training a real network gives neurons in
arbitrary order, which we simulate with a random row shuffle (an exact
symmetry once W_down is shuffled too). We compare, at equal parameter budgets:
  - shuffled  : MPO on the scrambled weights (what you get naively)
  - gauged    : MPO after our hierarchical spectral gauge fixing
  - oracle    : MPO on the planted order (upper bound)
  - low-rank  : plain truncated SVD (classical baseline)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qgauge.tt import to_mpo, from_mpo, n_params, entanglement_entropy
from qgauge.gauge import gauge_fix_ffn, anneal_order

rng = np.random.default_rng(0)
H, D, SITES, CHI_TRUE, NOISE = 1024, 256, 5, 6, 0.02


def random_mpo_matrix(R, C, sites, chi):
    rb = [int(np.log2(R)) // sites + (1 if k < int(np.log2(R)) % sites else 0) for k in range(sites)]
    cb = [int(np.log2(C)) // sites + (1 if k < int(np.log2(C)) % sites else 0) for k in range(sites)]
    cores, l = [], 1
    for k in range(sites):
        r = 1 if k == sites - 1 else chi
        cores.append(rng.standard_normal((l, 2 ** rb[k], 2 ** cb[k], r)))
        l = r
    W = from_mpo(cores)
    return W / np.linalg.norm(W)


W_up = random_mpo_matrix(H, D, SITES, CHI_TRUE)
W_up += NOISE * rng.standard_normal(W_up.shape) / np.sqrt(W_up.size)
W_down = random_mpo_matrix(D, H, SITES, CHI_TRUE).T.copy()  # (H, D) -> we store as (D, H)
W_down = W_down.T.copy() if W_down.shape != (D, H) else W_down
# make W_down correlated with W_up per neuron (as in trained nets): share structure
W_down = (W_up @ rng.standard_normal((D, D)) / np.sqrt(D)).T.copy()

P = rng.permutation(H)
up_s, down_s = W_up[P], W_down[:, P]
Pg = gauge_fix_ffn(up_s, down_s, sites=SITES)                     # spectral gauge
up_g, down_g = up_s[Pg], down_s[:, Pg]
Pa = anneal_order([up_g], SITES, P=np.arange(H), row_dims=None)    # + chi-annealed refinement
up_a, down_a = up_g[Pa], down_g[:, Pa]

# exactness check of the gauge: FFN(x) identical
x = rng.standard_normal((8, D))
relu = lambda z: np.maximum(z, 0)
y0 = relu(x @ up_s.T) @ down_s.T
y1 = relu(x @ up_a.T) @ down_a.T
print(f"gauge exactness max|dy| = {np.abs(y0 - y1).max():.2e}")


def rel_err(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(A)


def lowrank(W, k):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k], k * (W.shape[0] + W.shape[1])


print(f"\nW_up {H}x{D} = {H*D} params, planted chi={CHI_TRUE}, noise={NOISE}")
print(f"{'chi':>4} | {'shuffled':>16} | {'spectral':>16} | {'spectral+anneal':>16} | {'oracle':>16} | {'low-rank (same params)':>22}")
rows = []
for chi in [2, 4, 6, 8, 12, 16, 24, 32]:
    res = []
    for M in (up_s, up_g, up_a, W_up):
        cores, _ = to_mpo(M, SITES, chi=chi)
        res.append((rel_err(M, from_mpo(cores)), n_params(cores)))
    p = res[1][1]
    k = max(1, p // (H + D))
    lr, lp = lowrank(up_g, k)
    rows.append((chi, res, rel_err(up_g, lr), lp))
    print(f"{chi:>4} | " + " | ".join(f"{e:6.3f} @{n:>7}" for e, n in res) + f" | {rel_err(up_g, lr):6.3f} @{lp:>7} (k={k})")

print("\nEntanglement entropy (bits) per MPO cut, full rank:")
for name, M in (("shuffled", up_s), ("spectral", up_g), ("annealed", up_a), ("oracle", W_up)):
    _, spec = to_mpo(M, SITES)
    print(f"  {name:>8}: " + "  ".join(f"{entanglement_entropy(s):5.2f}" for s in spec))
