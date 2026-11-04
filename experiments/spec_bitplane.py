"""Experiment 15 (H12b): truly nested bit-plane self-speculation.

Store the teacher ONCE as 8-bit codes (per-group asymmetric grid, groups of 64).
Draft = only the top b bit-planes of those same codes (MSB prefix: code >> (8-b),
de-quantised at the bin centre), so the draft needs no extra memory at all.
Target = (a) the 8-bit model itself, or (b) the fp32 teacher.
Draft cost c = b/8 of the target's bytes (target read at 8 bits) or b/16 vs fp16.
"""
import sys, pickle
sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp
from memlm.data import load_wt2
from memlm.model import hidden

R, D = sys.argv[1], sys.argv[2]
tr, va, te, vocab = load_wt2(D); V = len(vocab)
raw = pickle.load(open(f"{R}/teacher.pkl", "rb"))["params"]
C, lo = 64, 32
X = np.stack([te[s:s + C] for s in np.random.default_rng(0).choice(len(te) - C - 1, 300, replace=False)])
MATS = ("qkv", "o", "w1", "w2")


def nested(W, b, gs=64):
    W = np.asarray(W, np.float32); shp = W.shape; W2 = W.reshape(-1, gs)
    mn, mx = W2.min(1, keepdims=True), W2.max(1, keepdims=True); s8 = (mx - mn) / 255 + 1e-12
    c8 = np.clip(np.round((W2 - mn) / s8), 0, 255).astype(np.int32)
    if b == 8:
        return (c8 * s8 + mn).reshape(shp)
    sh = 8 - b; cb = c8 >> sh
    return ((cb * (1 << sh) + ((1 << sh) - 1) / 2) * s8 + mn).reshape(shp)   # bin centre


def params(b):
    if b is None:
        return jax.tree_util.tree_map(jnp.asarray, raw)
    p = {k: (nested(v, b) if k == "emb" else v) for k, v in raw.items() if k != "layers"}
    p["layers"] = [{k: (nested(v, b) if k in MATS else v) for k, v in l.items()} for l in raw["layers"]]
    return jax.tree_util.tree_map(jnp.asarray, p)


def probs(p):
    L = np.asarray(jax.jit(lambda x: hidden(p, x, 4) @ p["emb"].T)(X)[:, lo:]).reshape(-1, V)
    P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)


P32, P8 = probs(params(None)), probs(params(8))
print(f"8-bit stored model vs fp32: alpha {np.minimum(P8, P32).sum(1).mean():.4f}, greedy agreement {(P8.argmax(1) == P32.argmax(1)).mean():.4f}")
print(f"{'draft = top b bit-planes':>26} | target | alpha | c (bytes) | best g | speed-up")
for b in (6, 5, 4, 3, 2):
    Pd = probs(params(b))
    for tname, PT, c in (("8-bit", P8, b / 8), ("fp16", P32, b / 16)):
        a = float(np.minimum(Pd, PT).sum(1).mean())
        sp, g = max(((1 - a ** (g + 1)) / (1 - a) / (g * c + 1), g) for g in range(1, 12))
        print(f"{b:>26} | {tname:>6} | {a:.3f} | {c:.3f}     | {g:>6} | {sp:.2f}x", flush=True)
