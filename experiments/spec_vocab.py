"""Experiment 12 (H8): make drafts cheap by restricting their output layer to
the N most frequent words (cf. FR-Spec). The target still uses the full
vocabulary, so speculative sampling stays exactly lossless. The draft
distribution is the softmax over the N kept words only.
Cost of a draft token = its transformer layers + N*d (instead of V*d).
"""
import sys, os, pickle
sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp
from collections import Counter
from memlm.data import load_wt2
from memlm.model import hidden, n_params

R, D = sys.argv[1], sys.argv[2]
tr, va, te, vocab = load_wt2(D); V = len(vocab)
ld = lambda n: pickle.load(open(f"{R}/{n}.pkl", "rb"))
T, S = ld("teacher"), ld("student")
import os
SKD = ld("student_kd") if os.path.exists(f"{R}/student_kd.pkl") and os.environ.get("KD") else None
pT = jax.tree_util.tree_map(jnp.asarray, T["params"]); pS = jax.tree_util.tree_map(jnp.asarray, S["params"])
C = T["cfg"]["ctx"]; lo = C // 2
freq_order = np.array([w for w, _ in Counter(tr.tolist()).most_common()])
starts = np.random.default_rng(0).choice(len(te) - C - 1, 300, replace=False)
X = np.stack([te[s:s + C] for s in starts])


def logits_fn(p, heads):
    return jax.jit(lambda x: hidden(p, x, heads) @ p["emb"].T)


LT = np.asarray(logits_fn(pT, 4)(X)[:, lo:]).reshape(-1, V)
PT = np.exp(LT - LT.max(1, keepdims=True)); PT /= PT.sum(1, keepdims=True)


def flops_layers(p, k=None):
    L = len(p["layers"]); per = (n_params(p) - p["emb"].size - p["pos"].size) / L
    return per * (L if k is None else k)


dT = T["params"]["emb"].shape[1]; full_T = flops_layers(T["params"]) + V * dT
drafts = [("student", pS, S["params"], None)]
if SKD is not None:
    drafts = [("student_kd (distilled)", jax.tree_util.tree_map(jnp.asarray, SKD["params"]), SKD["params"], None)] + drafts
if os.environ.get("KD"):
    drafts = drafts[:2]
drafts += [(f"teacher first {k} layers", dict(pT, layers=pT["layers"][:k]), T["params"], k) for k in (1, 2, 3)]
print(f"{'draft':>24} | {'vocab N':>7} | alpha | cost c | best g | speed-up")
for name, p, raw, k in drafts:
    Ld = np.asarray(logits_fn(p, 4)(X)[:, lo:]).reshape(-1, V)
    d = raw["emb"].shape[1]
    for N in ((V, 1000) if os.environ.get("KD") else (V, 4000, 2000, 1000, 500)):
        keep = freq_order[:N]
        Lk = np.full_like(Ld, -np.inf); Lk[:, keep] = Ld[:, keep]
        Pd = np.exp(Lk - Lk.max(1, keepdims=True)); Pd /= Pd.sum(1, keepdims=True)
        a = float(np.minimum(Pd, PT).sum(1).mean())
        c = (flops_layers(raw, k) + N * d) / full_T
        sp, g = max(((1 - a ** (g + 1)) / (1 - a) / (g * c + 1), g) for g in range(1, 9))
        print(f"{name:>24} | {N:>7} | {a:.3f} | {c:6.3f} | {g:>6} | {sp:.2f}x", flush=True)
