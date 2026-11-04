"""Experiment 14 (H12): self-speculation with a QUANTISED copy of the target.

Draft = the teacher itself with weights rounded to b bits (per-row groups of
64, RTN), optionally with the output layer restricted to the 1k most frequent
words. Target = the full-precision teacher, so the output is exactly the
teacher's. In large-model decoding, time per token ~ bytes of weights read,
so draft cost c ~ (bits / 16) x (fraction of weights read). With nested
bit-planes (MSB-first, MatQuant-style) the draft is a prefix of the target's
own storage: no second model and no extra memory.
"""
import sys, pickle
sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp
from collections import Counter
from memlm.data import load_wt2
from memlm.model import hidden

R, D = sys.argv[1], sys.argv[2]
tr, va, te, vocab = load_wt2(D); V = len(vocab)
T = pickle.load(open(f"{R}/teacher.pkl", "rb")); raw = T["params"]
C = 64; lo = 32
starts = np.random.default_rng(0).choice(len(te) - C - 1, 300, replace=False)
X = np.stack([te[s:s + C] for s in starts])
freq = np.array([w for w, _ in Counter(tr.tolist()).most_common()])


def q(W, bits, gs=64):
    W = np.asarray(W, np.float32); shp = W.shape; W2 = W.reshape(-1, gs) if W.size % gs == 0 else W.reshape(1, -1)
    lo_, hi_ = W2.min(1, keepdims=True), W2.max(1, keepdims=True); s = (hi_ - lo_) / (2 ** bits - 1) + 1e-12
    return (np.round((W2 - lo_) / s) * s + lo_).reshape(shp)


def quant_params(bits):
    p = {k: (q(v, bits) if k in ("emb",) else v) for k, v in raw.items() if k != "layers"}
    p["layers"] = [{k: (q(v, bits) if k in ("qkv", "o", "w1", "w2") else v) for k, v in l.items()} for l in raw["layers"]]
    return jax.tree_util.tree_map(jnp.asarray, p)


def logits(p):
    return np.asarray(jax.jit(lambda x: hidden(p, x, 4) @ p["emb"].T)(X)[:, lo:]).reshape(-1, V)


LT = logits(jax.tree_util.tree_map(jnp.asarray, raw)); PT = np.exp(LT - LT.max(1, keepdims=True)); PT /= PT.sum(1, keepdims=True)
d = raw["emb"].shape[1]; layer_w = sum(np.asarray(v).size for l in raw["layers"] for k, v in l.items() if k in ("qkv", "o", "w1", "w2"))
full_bytes = (layer_w + V * d) * 16
print(f"{'draft':>28} | alpha | cost c (bytes) | best g | speed-up")
for bits in (16, 8, 4, 3, 2):
    Ld = LT if bits == 16 else logits(quant_params(bits))
    for N in (V, 1000):
        Lk = Ld if N == V else np.where(np.isin(np.arange(V), freq[:N])[None], Ld, -np.inf)
        Pd = np.exp(Lk - Lk.max(1, keepdims=True)); Pd /= Pd.sum(1, keepdims=True)
        a = float(np.minimum(Pd, PT).sum(1).mean())
        c = (layer_w + N * d) * bits / full_bytes
        if a > 0.999:
            print(f"{'teacher ' + str(bits) + '-bit, vocab ' + str(N):>28} | {a:.3f} | {c:.3f}          | -      | (identical)"); continue
        sp, g = max(((1 - a ** (g + 1)) / (1 - a) / (g * c + 1), g) for g in range(1, 12))
        print(f"{'teacher ' + str(bits) + '-bit, vocab ' + str(N):>28} | {a:.3f} | {c:.3f}          | {g:>6} | {sp:.2f}x", flush=True)
