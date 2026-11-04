"""Experiment 16: the recipe on a REAL pretrained model (GPT-2 124M, ONNX zoo).

Text: Moby Dick (Gutenberg). For every 2-D weight matrix (attention, MLP,
embedding/LM head) we build
  stored-b : the model with weights stored at b bits (per-group-of-64 RTN)
  draft-p  : only the top p bit-planes of the 8-bit codes (nested, no extra memory)
and measure perplexity, greedy agreement with fp32, and speculative acceptance
alpha = E[sum_y min(p_draft, p_target)]. Speed-up uses the bandwidth model
(time per token ~ bytes of weights read), with speculative sampling exact
w.r.t. the target.
"""
import sys, json
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

DL, TXT = sys.argv[1], sys.argv[2]
NWIN, T = int(sys.argv[3]) if len(sys.argv) > 3 else 24, 128
import os
GS = int(os.environ.get("GS", 64)); AXIS_IN = os.environ.get("AXIS", "out") == "in"
sys.path.insert(0, ".")
from qgauge.roberta import _bytes_to_unicode, PAT
vocab = json.load(open(f"{DL}/gpt2-vocab.json"))
merges = [tuple(l.split()) for l in open(f"{DL}/gpt2-merges.txt", encoding="utf-8").read().split("\n")[1:] if l.strip()]
ranks = {m: i for i, m in enumerate(merges)}; b2u = _bytes_to_unicode(); cache = {}


def bpe(w):
    if w in cache: return cache[w]
    parts = list(w)
    while len(parts) > 1:
        r, i = min((ranks.get((a, b), 1e18), i) for i, (a, b) in enumerate(zip(parts, parts[1:])))
        if r == 1e18: break
        parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
    cache[w] = parts; return parts


def encode(text):
    return [vocab[p] for tok in PAT.findall(text) for p in bpe("".join(b2u[b] for b in tok.encode("utf-8")))]


text = open(TXT, encoding="utf-8", errors="ignore").read()[200000:200000 + NWIN * T * 6]
ids = np.array(encode(text)[: NWIN * T]).reshape(NWIN, T)
print(f"{NWIN} windows x {T} tokens", flush=True)

base = onnx.load(f"{DL}/gpt2-lm-head-10.onnx")
W = {t.name: numpy_helper.to_array(t) for t in base.graph.initializer}
big = [n for n, a in W.items() if a.ndim == 2 and a.size >= 4096 and a.dtype == np.float32]
nbig = sum(W[n].size for n in big); ntot = sum(a.size for a in W.values())
print(f"grouping: {GS} along {'input' if AXIS_IN else 'output'} axis"); print(f"{len(big)} matrices quantised = {nbig/1e6:.1f}M of {ntot/1e6:.1f}M weights", flush=True)


def grp(a):
    return (a.T if AXIS_IN else a).reshape(-1, GS)


def ungrp(q, shape):
    return q.reshape((shape[1], shape[0]) if AXIS_IN else shape).T if AXIS_IN else q.reshape(shape)


def codes8(a, gs=None):
    a2 = grp(a); mn, mx = a2.min(1, keepdims=True), a2.max(1, keepdims=True); s = (mx - mn) / 255 + 1e-12
    return np.clip(np.round((a2 - mn) / s), 0, 255).astype(np.int32), mn, s


C8 = {n: codes8(W[n]) for n in big}


def variant(kind, b):
    m = onnx.ModelProto(); m.CopyFrom(base)
    for t in m.graph.initializer:
        if t.name not in C8: continue
        a = W[t.name]
        if kind == "rtn":
            a2 = grp(a); mn, mx = a2.min(1, keepdims=True), a2.max(1, keepdims=True); s = (mx - mn) / (2 ** b - 1) + 1e-12
            q = ungrp(np.round((a2 - mn) / s) * s + mn, a.shape)
        else:  # nested prefix of the 8-bit codes
            c, mn, s = C8[t.name]; sh = 8 - b
            q = ungrp(((c >> sh) * (1 << sh) + ((1 << sh) - 1) / 2) * s + mn, a.shape) if sh else ungrp(c * s + mn, a.shape)
        t.CopyFrom(numpy_helper.from_array(q.astype(np.float32), t.name))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    return ort.InferenceSession(m.SerializeToString(), so)


def run(sess):
    out = []
    for w in ids:
        L = sess.run(["output1"], {"input1": w[None, None, :].astype(np.int64)})[0][0, 0]
        L = L - L.max(-1, keepdims=True); P = np.exp(L); P /= P.sum(-1, keepdims=True)
        out.append(P[:-1].astype(np.float32))
    return out


def ppl(P):
    lp = [np.log(np.take_along_axis(p, w[1:, None], -1)[:, 0] + 1e-12) for p, w in zip(P, ids)]
    return float(np.exp(-np.mean(np.concatenate(lp))))


def cmp(Pd, Pt):
    a = float(np.mean(np.concatenate([np.minimum(d, t).sum(-1) for d, t in zip(Pd, Pt)])))
    g = float(np.mean(np.concatenate([(d.argmax(-1) == t.argmax(-1)) for d, t in zip(Pd, Pt)])))
    return a, g


def speedup(a, c):
    return max(((1 - a ** (g + 1)) / (1 - a) / (g * c + 1), g) for g in range(1, 16)) if a < 0.9999 else (1 / c, 0)


P32 = run(ort.InferenceSession(f"{DL}/gpt2-lm-head-10.onnx"))
print(f"fp32 GPT-2: ppl {ppl(P32):.2f}", flush=True)
stored = {}
for b in (8, 4, 3):
    stored[b] = run(variant("rtn", b)); a, g = cmp(stored[b], P32)
    print(f"stored {b}-bit: ppl {ppl(stored[b]):.2f} | vs fp32: alpha {a:.4f}, greedy agreement {g:.4f}", flush=True)
OVH = 2 * 16 / GS  # fp16 scale + zero per group, in bits per weight
print(f"\nnested bit-plane self-drafts (prefix of the 8-bit codes); group overhead {OVH:.2f} bits/weight counted")
print(f"{'draft bits':>10} | {'target':>14} | alpha | draft cost | speed-up vs target | speed-up vs fp16 baseline")
targets = (("fp16", P32, 16.0), ("stored 8-bit", stored[8], 8 + OVH), ("stored 4-bit", stored[4], 4 + OVH))
for p in (6, 5, 4, 3, 2):
    Pd = run(variant("nested", p))
    for tname, Pt, tb in targets:
        if tname == "stored 4-bit" and p >= 4:
            continue
        a, _ = cmp(Pd, Pt); c = (p + OVH) / tb
        sp, g = speedup(a, c)
        print(f"{p:>10} | {tname:>14} | {a:.3f} | {c:.3f}      | {sp:.2f}x (draft len {g})      | {sp * 16 / tb:.2f}x", flush=True)
