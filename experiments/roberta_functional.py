"""Experiment 4: does the compressed model still compute the same thing?

All 24 FFN matrices of RoBERTa-base are compressed to a parameter budget
(fraction of the original FFN parameters). We measure, on held-out text
(Moby Dick; calibration on Austen's Emma), the relative error and cosine
similarity of the final hidden states vs. the uncompressed model.

Methods:
  svd        truncated SVD of W
  asvd       activation-whitened SVD (SVD-LLM style): SVD of W L, L = chol(E[x x^T])
  mpo        MPO / tensor-train of W, neuron order as trained (CompactifAI-like, no healing)
  mpo+gauge  MPO after Q-Gauge neuron permutation (exactly function-preserving)
  ampo+gauge activation-aware MPO: MPO of W L (+gauge), then times L^-1 shared per input space
             (L^-1 is stored once per layer input and counted in the budget)
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qgauge.roberta import Tokenizer, Roberta, load_corpus
from qgauge.tt import to_mpo, from_mpo, n_params
from qgauge.gauge import gauge_fix_ffn

S = sys.argv[1]  # dir with dl/roberta.npz, dl/bpe.pkl, text/gutenberg
BUDGETS = [float(x) for x in os.environ.get("BUDGETS", "0.1,0.25,0.5").split(",")]
METHODS = os.environ.get("METHODS", "svd,asvd,mpo,mpo+gauge").split(",")
SITES = int(os.environ.get("SITES", 3))
GAUGE_ANNEAL = tuple(int(x) for x in os.environ.get("ANNEAL", "16").split(","))

tok = Tokenizer(S + "/dl/bpe.pkl")
model = Roberta(np.load(S + "/dl/roberta.npz"))
calib = load_corpus(tok, S + "/text/gutenberg/austen-emma.txt", 16, 128, offset=5000)
evalx = load_corpus(tok, S + "/text/gutenberg/melville-moby_dick.txt", 16, 128, offset=20000)

# --- calibration statistics: input covariance for every FFN matrix
cov = {}
def hook(l, name, x):
    if name.startswith("ffn"):
        z = x.reshape(-1, x.shape[-1]).astype(np.float64)
        k = f"layers.{l}.{name}"
        cov[k] = cov.get(k, 0) + z.T @ z
model.forward(calib, hook=hook)
ref = model.forward(evalx)


def w(key):
    return model.w(key + ".weight").astype(np.float64)


def chol(C):
    C = C / np.trace(C) * C.shape[0]
    lam = 1e-4 * np.trace(C) / C.shape[0]
    return np.linalg.cholesky(C + lam * np.eye(C.shape[0]))


def svd_k(W, k):
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def mpo_at_budget(W, budget):
    lo, hi, best = 1, 2048, None
    while lo <= hi:
        chi = (lo + hi) // 2
        cores, _ = to_mpo(W, SITES, chi=chi)
        if n_params(cores) <= budget:
            best = (from_mpo(cores), n_params(cores)); lo = chi + 1
        else:
            hi = chi - 1
    return best


perms = {}
def get_perm(l):
    if l not in perms:
        up, down = w(f"layers.{l}.ffn.intermediate"), w(f"layers.{l}.ffn.output")
        perms[l] = gauge_fix_ffn(up, down, sites=SITES, anneal=GAUGE_ANNEAL)
    return perms[l]


def compress(method, frac):
    over, used, total = {}, 0, 0
    for l in range(model.L):
        up_k, dn_k = f"layers.{l}.ffn.intermediate", f"layers.{l}.ffn.output"
        Wu, Wd = w(up_k), w(dn_k)  # (3072,768), (768,3072)
        total += Wu.size + Wd.size
        news = {}
        for key, W in ((up_k, Wu), (dn_k, Wd)):
            b = int(frac * W.size)
            if method == "svd":
                k = max(1, b // sum(W.shape)); news[key] = svd_k(W, k); used += k * sum(W.shape)
            elif method == "asvd":
                Lc = chol(cov[key]); k = max(1, b // sum(W.shape))
                news[key] = np.linalg.solve(Lc.T, svd_k(W @ Lc, k).T).T; used += k * sum(W.shape)
            elif method in ("mpo", "mpo+gauge", "dmpo", "dmpo+gauge"):
                # diagonal activation scaling (d*): MPO of W diag(s), s = RMS of inputs;
                # costs only len(s) extra numbers, keeps the MPO structure.
                sc = np.sqrt(np.diag(cov[key]) / len(calib.ravel())) + 1e-6 if method.startswith("d") else np.ones(W.shape[1])
                Ws = W * sc[None, :]
                if method.endswith("+gauge"):
                    P = get_perm(l)
                    Wp = Ws[P] if key == up_k else Ws[:, P].T  # neurons on rows
                else:
                    Wp = Ws if key == up_k else Ws.T
                A, p = mpo_at_budget(Wp, b - (len(sc) if method.startswith("d") else 0)); used += p + (len(sc) if method.startswith("d") else 0)
                if method.endswith("+gauge"):
                    inv = np.argsort(P)
                    A = A[inv] if key == up_k else A[inv].T
                else:
                    A = A if key == up_k else A.T
                news[key] = A / sc[None, :]
        for key, A in news.items():
            Af = A.astype(np.float32)
            over[key] = (lambda M: (lambda z: z @ M.T))(Af)
    return over, used / total


results = []
for frac in BUDGETS:
    for m in METHODS:
        t = time.time()
        over, real = compress(m, frac)
        out = model.forward(evalx, override=over)
        rel = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
        cos = float(np.mean(np.sum(out * ref, -1) / (np.linalg.norm(out, axis=-1) * np.linalg.norm(ref, axis=-1))))
        results.append(dict(budget=frac, method=m, real_frac=real, rel_err=rel, cos=cos, secs=time.time() - t))
        print(f"budget {frac:4.2f} | {m:>10} | real {real:5.3f} | rel err {rel:6.3f} | cos {cos:6.3f} | {time.time()-t:5.0f}s", flush=True)
json.dump(results, open(os.environ.get("OUT", "/dev/null"), "w"), indent=1)
