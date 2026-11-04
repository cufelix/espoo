"""Experiment 6: dynamic routing / contextual sparsity in FFN (oracle bound).

For every token only a subset of FFN neurons is used:
  topk-neurons : keep the k neurons with largest |GELU activation| per token
  experts      : neurons grouped into 48 'experts' of 64; per token keep the
                 g experts with largest summed |activation| (MoEfication-style).
                 Grouping either random or by Q-Gauge hierarchical clustering.
Oracle = selection uses the true activations, i.e. an upper bound on what a
cheap learned router could achieve. Metric: cosine of final hidden states.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qgauge.roberta import Tokenizer, Roberta, load_corpus
from qgauge.gauge import hierarchical_order

S = sys.argv[1]
tok = Tokenizer(S + "/dl/bpe.pkl"); model = Roberta(np.load(S + "/dl/roberta.npz"))
evalx = load_corpus(tok, S + "/text/gutenberg/melville-moby_dick.txt", 16, 128, offset=20000)
ref = model.forward(evalx)
H, E = 3072, 48
G = H // E


def cos(o):
    return float(np.mean(np.sum(o * ref, -1) / np.linalg.norm(o, axis=-1) / np.linalg.norm(ref, axis=-1)))


def run(select):
    over = {}
    for l in range(model.L):
        Wd = model.w(f"layers.{l}.ffn.output.weight")
        def f(h, Wd=Wd, l=l):
            m = select(l, h)
            return (h * m) @ Wd.T
        over[f"layers.{l}.ffn.output"] = f
    return cos(model.forward(evalx, override=over))


def topk(k):
    def sel(l, h):
        a = np.abs(h); th = -np.partition(-a, k - 1, axis=-1)[..., k - 1:k]
        return (a >= th).astype(h.dtype)
    return sel


groups_rand = [np.random.default_rng(l).permutation(H).reshape(E, G) for l in range(12)]
groups_gauge = []
for l in range(12):
    up = model.w(f"layers.{l}.ffn.intermediate.weight"); dn = model.w(f"layers.{l}.ffn.output.weight")
    P = hierarchical_order(np.concatenate([up, dn.T], 1).astype(np.float64), [48, 64])
    groups_gauge.append(P.reshape(E, G))


def experts(g, groups):
    def sel(l, h):
        a = np.abs(h); idx = groups[l]
        score = a[..., idx].sum(-1)                      # (B,T,E)
        th = -np.partition(-score, g - 1, axis=-1)[..., g - 1:g]
        keepE = score >= th
        m = np.zeros_like(h)
        for e in range(E):
            m[..., idx[e]] = keepE[..., e:e + 1]
        return m
    return sel


print("fraction of FFN neurons active per token -> cosine with full model")
for frac in (0.05, 0.1, 0.2, 0.3, 0.5):
    k = int(frac * H); g = max(1, int(round(frac * E)))
    print(f"{frac:4.0%} | top-k neurons {run(topk(k)):.4f} | experts random {run(experts(g, groups_rand)):.4f} | experts Q-Gauge {run(experts(g, groups_gauge)):.4f}", flush=True)
