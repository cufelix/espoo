"""Experiment 7: how many bits per weight does RoBERTa's FFN really need?

All 24 FFN matrices quantised (per-row groups of 128 input weights,
asymmetric uniform grid) with
  rtn  : round-to-nearest
  gptq : GPTQ (Frantar et al. 2022) using the calibration input covariance
and then the integer codes are entropy-coded (we report the empirical
Shannon entropy = achievable size with an ideal entropy coder, plus the
per-group scale/zero overhead in fp16).
Metric: cosine of final hidden states vs. the fp32 model on held-out text.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qgauge.roberta import Tokenizer, Roberta, load_corpus

S = sys.argv[1]
GS = 128
tok = Tokenizer(S + "/dl/bpe.pkl"); model = Roberta(np.load(S + "/dl/roberta.npz"))
calib = load_corpus(tok, S + "/text/gutenberg/austen-emma.txt", 16, 128, offset=5000)
evalx = load_corpus(tok, S + "/text/gutenberg/melville-moby_dick.txt", 16, 128, offset=20000)
cov = {}
def hook(l, name, x):
    if name.startswith("ffn"):
        z = x.reshape(-1, x.shape[-1]).astype(np.float64); k = f"layers.{l}.{name}"
        cov[k] = cov.get(k, 0) + z.T @ z
model.forward(calib, hook=hook); ref = model.forward(evalx)


def grid(wg, bits):
    lo, hi = wg.min(1, keepdims=True), wg.max(1, keepdims=True)
    scale = np.maximum(hi - lo, 1e-8) / (2 ** bits - 1)
    return scale, lo


def q_rtn(W, bits):
    Q = np.empty_like(W); codes = np.empty(W.shape, dtype=np.int32)
    for g in range(0, W.shape[1], GS):
        s, z = grid(W[:, g:g + GS], bits)
        c = np.clip(np.round((W[:, g:g + GS] - z) / s), 0, 2 ** bits - 1)
        codes[:, g:g + GS] = c; Q[:, g:g + GS] = c * s + z
    return Q, codes


def q_gptq(W, H, bits, block=128):
    W = W.copy(); n = W.shape[1]
    H = H.copy(); dead = np.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += 0.01 * np.mean(np.diag(H)) * np.eye(n)
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T  # upper
    Q = np.zeros_like(W); codes = np.zeros(W.shape, dtype=np.int32)
    s = z = None
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        W1 = W[:, b0:b1].copy(); E1 = np.zeros_like(W1); Hi = Hinv[b0:b1, b0:b1]
        for i in range(b1 - b0):
            col = b0 + i
            if col % GS == 0:
                s, z = grid(W[:, col:col + GS] if i == 0 and GS <= block else np.concatenate([W1[:, i:], W[:, b1:]], 1)[:, :GS], bits)
            w = W1[:, i]
            c = np.clip(np.round((w - z[:, 0]) / s[:, 0]), 0, 2 ** bits - 1)
            q = c * s[:, 0] + z[:, 0]
            Q[:, col] = q; codes[:, col] = c
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= np.outer(e, Hi[i, i:]); E1[:, i] = e
        W[:, b1:] -= E1 @ Hinv[b0:b1, b1:]
    return Q, codes


def entropy_bits(codes):
    _, cnt = np.unique(codes, return_counts=True); p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def cos(o):
    return float(np.mean(np.sum(o * ref, -1) / np.linalg.norm(o, axis=-1) / np.linalg.norm(ref, axis=-1)))


print(f"{'method':>5} {'bits':>4} | {'cos':>6} | {'raw b/w':>7} {'entropy b/w':>11} | {'x vs fp16':>9}")
for bits in (8, 4, 3, 2):
    for method in ("rtn", "gptq"):
        t = time.time(); over = {}; ent = []; nw = 0
        for l in range(model.L):
            for name in ("ffn.intermediate", "ffn.output"):
                key = f"layers.{l}.{name}"; W = model.w(key + ".weight").astype(np.float64)
                Q, c = q_rtn(W, bits) if method == "rtn" else q_gptq(W, cov[key], bits)
                ent.append(entropy_bits(c) * W.size); nw += W.size
                over[key] = (lambda M: (lambda x: x @ M.T))(Q.astype(np.float32))
        overhead = 2 * 16 / GS  # fp16 scale + zero per group
        e = sum(ent) / nw + overhead
        print(f"{method:>5} {bits:>4} | {cos(model.forward(evalx, override=over)):6.4f} | {bits + overhead:7.2f} {e:11.2f} | {16 / e:8.1f}x  ({time.time()-t:.0f}s)", flush=True)
