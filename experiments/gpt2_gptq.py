"""Experiment 17: GPTQ on real GPT-2 (ONNX) to rescue 3-bit / 4-bit storage.

1. Expose the input activation of every MatMul whose weight is a large
   initializer, run calibration text (Austen), accumulate H = X^T X per weight.
2. GPTQ (Frantar et al. 2022) per weight with group-32 asymmetric grids
   along the output axis (the layout that worked for RTN).
3. Evaluate perplexity / agreement on held-out text (Moby Dick) vs fp32 and RTN.
Embedding (wte, also the LM head) and wpe use RTN at the same bits.
"""
import sys, os, json
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper, helper
sys.path.insert(0, ".")
import importlib.util
DL, CAL, EVAL = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("rec", "experiments/gpt2_recipe.py")
# reuse the tokenizer from gpt2_recipe without running it: copy the minimal parts
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
enc = lambda t: [vocab[p] for tok in PAT.findall(t) for p in bpe("".join(b2u[b] for b in tok.encode("utf-8")))]
T = 128
def windows(path, off, n):
    ids = enc(open(path, encoding="utf-8", errors="ignore").read()[off: off + n * T * 6])[: n * T]
    return np.array(ids).reshape(n, T)
cal, ev = windows(CAL, 10000, 32), windows(EVAL, 200000, 32)

base = onnx.load(f"{DL}/gpt2-lm-head-10.onnx")
W = {t.name: numpy_helper.to_array(t) for t in base.graph.initializer}
mm = {}
for n in base.graph.node:
    if n.op_type in ("MatMul", "Gemm") and len(n.input) > 1 and n.input[1] in W and W[n.input[1]].ndim == 2 and W[n.input[1]].size >= 4096:
        mm[n.input[1]] = n.input[0]
print(f"{len(mm)} MatMul weights get GPTQ", flush=True)
probe = onnx.ModelProto(); probe.CopyFrom(base)
for wname, xname in mm.items():
    probe.graph.output.append(helper.make_tensor_value_info(xname, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(probe.SerializeToString())
outs = [o.name for o in sess.get_outputs()]
H = {w: 0 for w in mm}
for w_ in cal:
    res = dict(zip(outs, sess.run(outs, {"input1": w_[None, None].astype(np.int64)})))
    for wname, xname in mm.items():
        X = res[xname].reshape(-1, res[xname].shape[-1]).astype(np.float64); H[wname] = H[wname] + X.T @ X
del sess, probe
GS = 32


def gptq(Wt, Hm, bits, block=128):
    """Wt: (out, in). GPTQ over input columns; groups of GS along the OUTPUT axis
    means per-column grids of GS consecutive rows."""
    Wt = Wt.astype(np.float64).copy(); n = Wt.shape[1]
    Hm = Hm.copy(); dead = np.diag(Hm) == 0; Hm[dead, dead] = 1; Wt[:, dead] = 0
    Hm += 0.01 * np.mean(np.diag(Hm)) * np.eye(n)
    Hinv = np.linalg.cholesky(np.linalg.inv(Hm)).T
    Q = np.zeros_like(Wt); out = Wt.shape[0]; G = out // GS
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n); W1 = Wt[:, b0:b1].copy(); E1 = np.zeros_like(W1); Hi = Hinv[b0:b1, b0:b1]
        for i in range(b1 - b0):
            w = W1[:, i].reshape(G, GS)
            mn, mx = w.min(1, keepdims=True), w.max(1, keepdims=True); s = (mx - mn) / (2 ** bits - 1) + 1e-12
            q = (np.clip(np.round((w - mn) / s), 0, 2 ** bits - 1) * s + mn).ravel()
            Q[:, b0 + i] = q; e = (W1[:, i] - q) / Hi[i, i]
            W1[:, i:] -= np.outer(e, Hi[i, i:]); E1[:, i] = e
        Wt[:, b1:] -= E1 @ Hinv[b0:b1, b1:]
    return Q


def rtn(a, bits):
    a2 = a.reshape(-1, GS); mn, mx = a2.min(1, keepdims=True), a2.max(1, keepdims=True); s = (mx - mn) / (2 ** bits - 1) + 1e-12
    return (np.round((a2 - mn) / s) * s + mn).reshape(a.shape)


def build(bits, method):
    m = onnx.ModelProto(); m.CopyFrom(base)
    for t in m.graph.initializer:
        a = W[t.name]
        if a.ndim != 2 or a.size < 4096 or a.dtype != np.float32: continue
        if method == "gptq" and t.name in mm and a.shape[1] % GS == 0:
            q = gptq(a.T, H[t.name], bits).T          # ONNX weight is (in, out)
        else:
            q = rtn(a, bits)
        t.CopyFrom(numpy_helper.from_array(q.astype(np.float32), t.name))
    return ort.InferenceSession(m.SerializeToString())


def probs(sess):
    out = []
    for w_ in ev:
        L = sess.run(["output1"], {"input1": w_[None, None].astype(np.int64)})[0][0, 0]
        L = L - L.max(-1, keepdims=True); P = np.exp(L); P /= P.sum(-1, keepdims=True); out.append(P[:-1])
    return out


def ppl(P):
    return float(np.exp(-np.mean(np.concatenate([np.log(np.take_along_axis(p, w_[1:, None], -1)[:, 0] + 1e-12) for p, w_ in zip(P, ev)]))))


P32 = probs(ort.InferenceSession(f"{DL}/gpt2-lm-head-10.onnx")); print(f"fp32 ppl {ppl(P32):.2f}", flush=True)
for bits in (4, 3):
    for method in ("rtn", "gptq"):
        P = probs(build(bits, method))
        a = float(np.mean(np.concatenate([np.minimum(p, q).sum(-1) for p, q in zip(P, P32)])))
        g = float(np.mean(np.concatenate([(p.argmax(-1) == q.argmax(-1)) for p, q in zip(P, P32)])))
        eff = bits + 1.0
        print(f"{bits}-bit {method:>4}: ppl {ppl(P):8.2f} | alpha vs fp32 {a:.3f} | greedy agree {g:.3f} | {16/eff:.2f}x fewer bytes than fp16", flush=True)
