"""Experiment 2: real pretrained weights (RoBERTa-base, 12 layers, FFN 3072x768).

For FFN matrices we compare, at equal parameter budgets, the relative
reconstruction error of:
  trained   : MPO on the neuron order as it came out of training
  random    : MPO after a random neuron permutation (control)
  gauged    : MPO after Q-Gauge (spectral + chi-annealed assign-and-fit)
  low-rank  : truncated SVD with the same number of parameters
Error is measured jointly on W_up and W_down (shared neuron permutation).

usage: python experiments/roberta_weights.py path/to/roberta.npz [layers]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qgauge.tt import to_mpo, from_mpo, n_params, entanglement_entropy, factorize
from qgauge.gauge import gauge_fix_ffn

SITES = 5
CHIS = [8, 16, 32, 64, 128]

sd = np.load(sys.argv[1])
layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 5, 11]
key = "curated_encoder.layers.{}.ffn.{}.weight"
rng = np.random.default_rng(0)


def joint_err(mats, chi):
    e2 = tot = 0.0; p = 0
    for m in mats:
        cores, _ = to_mpo(m, SITES, chi=chi)
        A = from_mpo(cores); p += n_params(cores)
        e2 += ((m - A) ** 2).sum(); tot += (m ** 2).sum()
    return np.sqrt(e2 / tot), p


def lowrank_err(mats, budget):
    e2 = tot = 0.0; p = 0
    for m in mats:
        k = max(1, int(budget / len(mats)) // sum(m.shape))
        U, S, Vt = np.linalg.svd(m, full_matrices=False)
        e2 += (S[k:] ** 2).sum(); tot += (S ** 2).sum(); p += k * sum(m.shape)
    return np.sqrt(e2 / tot), p


print(f"MPO sites={SITES}, row dims {factorize(3072, SITES)}, col dims {factorize(768, SITES)}")
for L in layers:
    up = sd[key.format(L, "intermediate")].astype(np.float64)   # (3072, 768)
    down = sd[key.format(L, "output")].astype(np.float64)       # (768, 3072)
    full = up.size + down.size
    t = time.time()
    Pg = gauge_fix_ffn(up, down, sites=SITES, anneal=(4, 8, 16, 32))
    tg = time.time() - t
    Pr = rng.permutation(up.shape[0])
    variants = {
        "trained": [up, down.T],
        "random": [up[Pr], down.T[Pr]],
        "gauged": [up[Pg], down.T[Pg]],
    }
    print(f"\n=== layer {L}  (FFN params {full}, gauge search {tg:.0f}s) ===")
    print(f"{'chi':>4} {'%params':>8} | {'trained':>7} {'random':>7} {'gauged':>7} | {'low-rank':>8}")
    for chi in CHIS:
        res = {n: joint_err(m, chi) for n, m in variants.items()}
        p = res["gauged"][1]
        lr, _ = lowrank_err(variants["trained"], p)
        print(f"{chi:>4} {100*p/full:>7.2f}% | " + " ".join(f"{res[n][0]:7.3f}" for n in variants) + f" | {lr:8.3f}")
    for n, m in variants.items():
        _, spec = to_mpo(m[0], SITES)
        print(f"  entropy W_up {n:>8}: " + " ".join(f"{entanglement_entropy(s):5.2f}" for s in spec))
