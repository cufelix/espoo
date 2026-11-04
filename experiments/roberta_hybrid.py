"""Experiment 5: hybrid ASVD x MPO.

ASVD gives W ~= A B with A = U_k S_k and B = V_k^T L^-1 (activation-whitened).
One factor is indexed by FFN neurons (A for W_up, B^T for W_down). That
factor is compressed again with an MPO (optionally after the Q-Gauge
permutation of neurons), which frees budget for a larger rank k.
Question: at equal total parameters, does ASVD(k_big) + MPO(neuron factor)
beat plain ASVD(k)?
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qgauge.roberta import Tokenizer, Roberta, load_corpus
from qgauge.tt import to_mpo, from_mpo, n_params, factorize
from qgauge.gauge import hierarchical_order

S = sys.argv[1]
FRAC = float(os.environ.get("FRAC", 0.5))
tok = Tokenizer(S + "/dl/bpe.pkl"); model = Roberta(np.load(S + "/dl/roberta.npz"))
calib = load_corpus(tok, S + "/text/gutenberg/austen-emma.txt", 16, 128, offset=5000)
evalx = load_corpus(tok, S + "/text/gutenberg/melville-moby_dick.txt", 16, 128, offset=20000)
cov = {}
def hook(l, name, x):
    if name.startswith("ffn"):
        z = x.reshape(-1, x.shape[-1]).astype(np.float64); k = f"layers.{l}.{name}"
        cov[k] = cov.get(k, 0) + z.T @ z
model.forward(calib, hook=hook); ref = model.forward(evalx)


def chol(C):
    C = C / np.trace(C) * C.shape[0]
    return np.linalg.cholesky(C + 1e-4 * np.eye(C.shape[0]))


def mpo_fit(M, budget, sites=3):
    lo, hi, best = 1, 4096, None
    while lo <= hi:
        chi = (lo + hi) // 2
        c, _ = to_mpo(M, sites, chi=chi)
        if n_params(c) <= budget: best = (from_mpo(c), n_params(c)); lo = chi + 1
        else: hi = chi - 1
    return best


SVD = {}
def factors(key, W):
    if key not in SVD:
        Lc = chol(cov[key]); U, s, Vt = np.linalg.svd(W @ Lc, full_matrices=False)
        SVD[key] = (U * s, np.linalg.solve(Lc.T, Vt.T).T)  # A (out x r), B (r x in)
    return SVD[key]


def build(kmult, mpo_share, gauge):
    over, used, total = {}, 0, 0
    for l in range(model.L):
        for name in ("ffn.intermediate", "ffn.output"):
            key = f"layers.{l}.{name}"; W = model.w(key + ".weight").astype(np.float64)
            total += W.size; budget = FRAC * W.size
            A, B = factors(key, W)
            k0 = int(budget // sum(W.shape))
            k = min(int(k0 * kmult) // 16 * 16, min(W.shape))
            A, B = A[:, :k], B[:k]
            neuron_up = name == "ffn.intermediate"
            F = A if neuron_up else B.T            # neuron-indexed factor (3072 x k)
            other = B.size if neuron_up else A.size
            fb = budget - other
            if fb >= F.size or mpo_share == 0:
                Fh, p = F, F.size
            else:
                P = hierarchical_order(F, factorize(F.shape[0], 3)) if gauge else np.arange(F.shape[0])
                fac = factorize(k, 3)
                Fh, p = mpo_fit(F[P], fb); Fh = Fh[np.argsort(P)]
            used += p + other
            Wh = Fh @ B if neuron_up else A @ Fh.T
            over[key] = (lambda M: (lambda z: z @ M.T))(Wh.astype(np.float32))
    return over, used / total


def score(over):
    o = model.forward(evalx, override=over)
    return np.linalg.norm(o - ref) / np.linalg.norm(ref), float(np.mean(np.sum(o * ref, -1) / np.linalg.norm(o, axis=-1) / np.linalg.norm(ref, axis=-1)))


print(f"budget {FRAC}")
for kmult, gauge in [(1.0, False), (1.25, False), (1.25, True), (1.5, False), (1.5, True), (2.0, True)]:
    t = time.time(); over, real = build(kmult, 1, gauge); e, c = score(over)
    print(f"k x{kmult:4.2f} gauge={gauge!s:5} | real {real:5.3f} | rel err {e:6.3f} | cos {c:6.3f} | {time.time()-t:4.0f}s", flush=True)
