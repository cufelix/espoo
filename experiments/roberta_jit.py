"""Experiment 8: just-in-time model specialisation.

Hypothesis: the activations a model uses within ONE context (document /
conversation) live in a much smaller subspace than across all text. If so,
the model can be 'recompiled' per context into a far smaller model.

Compress all 24 FFN matrices with activation-whitened SVD (ASVD), calibrated:
  generic   : on a different book (Austen, Emma)
  in-context: on an earlier, disjoint part of the same book (Moby Dick)
  oracle    : on the evaluation text itself (upper bound)
and evaluate on a later part of Moby Dick. Metric: cosine of final hidden
states vs. the uncompressed model.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qgauge.roberta import Tokenizer, Roberta, load_corpus

S = sys.argv[1]
tok = Tokenizer(S + "/dl/bpe.pkl"); model = Roberta(np.load(S + "/dl/roberta.npz"))
moby = S + "/text/gutenberg/melville-moby_dick.txt"
evalx = load_corpus(tok, moby, 16, 128, offset=400000)
sets = {
    "generic": load_corpus(tok, S + "/text/gutenberg/austen-emma.txt", 16, 128, offset=5000),
    "in-context": load_corpus(tok, moby, 16, 128, offset=300000),
    "oracle": evalx,
}
ref = model.forward(evalx)


def covs(x):
    c = {}
    def hook(l, name, z):
        if name.startswith("ffn"):
            z = z.reshape(-1, z.shape[-1]).astype(np.float64); k = f"layers.{l}.{name}"
            c[k] = c.get(k, 0) + z.T @ z
    model.forward(x, hook=hook); return c


C = {n: covs(x) for n, x in sets.items()}
FACT = {}


def asvd_factors(key, src):
    if (key, src) not in FACT:
        W = model.w(key + ".weight").astype(np.float64); Cv = C[src][key]
        Cv = Cv / np.trace(Cv) * Cv.shape[0]
        L = np.linalg.cholesky(Cv + 1e-4 * np.eye(Cv.shape[0]))
        U, s, Vt = np.linalg.svd(W @ L, full_matrices=False)
        FACT[(key, src)] = (U * s, np.linalg.solve(L.T, Vt.T).T)
    return FACT[(key, src)]


def run(src, frac):
    over = {}
    for l in range(model.L):
        for name in ("ffn.intermediate", "ffn.output"):
            key = f"layers.{l}.{name}"; W = model.w(key + ".weight")
            k = max(1, int(frac * W.size) // sum(W.shape))
            A, B = asvd_factors(key, src)
            over[key] = (lambda M: (lambda z: z @ M.T))((A[:, :k] @ B[:k]).astype(np.float32))
    o = model.forward(evalx, override=over)
    return float(np.mean(np.sum(o * ref, -1) / np.linalg.norm(o, axis=-1) / np.linalg.norm(ref, axis=-1)))


print("FFN size kept | generic | in-context | oracle   (cosine vs full model)")
for frac in (0.05, 0.1, 0.25, 0.5):
    print(f"{frac:12.0%} | " + " | ".join(f"{run(s, frac):7.4f}" for s in sets), flush=True)
