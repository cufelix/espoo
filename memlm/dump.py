"""Run a trained LM over a split and dump what the memory experiments need.

For every position i (predicting token i+1), with >= ctx/2 tokens of context
(sliding windows, stride ctx/2):
  keys  : final hidden state (fp16)          -> kNN key
  gold  : the true next token
  lp    : model log-prob of the gold token
  topk  : (optional) top-k next-token ids and probs (the model's 'soft label')

usage: python -m memlm.dump model.pkl split out.npz [--topk 16]
"""
import argparse, pickle
import numpy as np, jax, jax.numpy as jnp
from .data import load_wt2
from .model import hidden

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("split"); ap.add_argument("out")
ap.add_argument("--data", required=True); ap.add_argument("--topk", type=int, default=0)
a = ap.parse_args()
m = pickle.load(open(a.model, "rb")); cfg = m["cfg"]
p = jax.tree_util.tree_map(jnp.asarray, m["params"])
tr, va, te, vocab = load_wt2(a.data)
data = {"train": tr, "valid": va, "test": te}[a.split]
C, H = cfg["ctx"], cfg["heads"]; half = C // 2


@jax.jit
def run(x):
    h = hidden(p, x, H)
    lp = jax.nn.log_softmax(h @ p["emb"].T, -1)
    return h, lp


starts = list(range(0, len(data) - C - 1, half))
N = len(data) - 1
keys = np.zeros((N, p["emb"].shape[1]), np.float16); lpg = np.zeros(N, np.float32)
have = np.zeros(N, bool)
if a.topk:
    tid = np.zeros((N, a.topk), np.int32); tpr = np.zeros((N, a.topk), np.float16)
B = 64
for b in range(0, len(starts), B):
    ss = starts[b:b + B]
    X = np.stack([data[s:s + C] for s in ss]); Y = np.stack([data[s + 1:s + C + 1] for s in ss])
    h, lp = run(X); h, lp = np.asarray(h), np.asarray(lp)
    for j, s in enumerate(ss):
        lo = 0 if s == 0 else half
        pos = np.arange(s + lo, s + C)
        keys[pos] = h[j, lo:]; have[pos] = True
        lpg[pos] = np.take_along_axis(lp[j, lo:], Y[j, lo:, None], -1)[:, 0]
        if a.topk:
            top = np.argpartition(-lp[j, lo:], a.topk, -1)[:, :a.topk]
            tid[pos] = top; tpr[pos] = np.exp(np.take_along_axis(lp[j, lo:], top, -1))
idx = np.where(have)[0]
out = {"pos": idx, "keys": keys[idx], "gold": data[idx + 1].astype(np.int32), "lp": lpg[idx]}
if a.topk:
    out["ids"] = tid[idx]; out["probs"] = tpr[idx]
    # full-length arrays aligned with train positions, for KD training
    out["kd_ids"] = tid; out["kd_probs"] = tpr
np.savez(a.out, **out)
print(f"{a.split}: {len(idx)} positions, ppl {np.exp(-lpg[idx].mean()):.2f}")
