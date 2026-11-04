"""Experiment 10: lossless speed-up via speculative decoding with our own
teacher (target, 5.7M) and student (draft, 1.7M) on WikiText-2 test text,
and whether a kNN memory makes the draft better.

Speculative sampling (Leviathan et al. 2023) returns samples from EXACTLY the
target distribution. Per position the acceptance probability is
alpha = sum_y min(p_draft(y), p_target(y)). With draft length g and draft
cost c (in target-forward units), expected target-quality tokens per round
E = (1 - a^(g+1)) / (1 - a) (i.i.d. approximation), time per round g*c + 1,
speed-up = E / (g*c + 1). Distributions are computed teacher-forced on real
text (realistic contexts).

Drafts compared: student; student + kNN memory (gold values, student keys,
lam/tau from the memory experiment); teacher-valued memory.
"""
import sys, os, pickle
sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp
from memlm.data import load_wt2
from memlm.model import hidden, n_params
from memlm.knn import build_index

R, D = sys.argv[1], sys.argv[2]
NWIN = int(os.environ.get("NWIN", 300))
tr, va, te, vocab = load_wt2(D); V = len(vocab)
ld = lambda n: pickle.load(open(f"{R}/{n}.pkl", "rb"))
T, S = ld("teacher"), ld("student")
pT = jax.tree_util.tree_map(jnp.asarray, T["params"]); pS = jax.tree_util.tree_map(jnp.asarray, S["params"])
C = T["cfg"]["ctx"]


def dist(p, heads):
    @jax.jit
    def f(x):
        h = hidden(p, x, heads)
        return h, jax.nn.softmax(h @ p["emb"].T, -1)
    return f


fT, fS = dist(pT, T["cfg"]["heads"]), dist(pS, S["cfg"]["heads"])
starts = np.random.default_rng(0).choice(len(te) - C - 1, NWIN, replace=False)
X = np.stack([te[s:s + C] for s in starts])
_, PT = fT(X); hS, PS = fS(X)
lo = C // 2  # only positions with >= 32 tokens of context
PT, PS, hS = np.asarray(PT[:, lo:]).reshape(-1, V), np.asarray(PS[:, lo:]).reshape(-1, V), np.asarray(hS[:, lo:]).reshape(-1, hS.shape[-1])
print(f"{len(PT)} positions", flush=True)

# memory distribution from student neighbours (gold values / teacher values)
Str, Ttr = np.load(f"{R}/S_train.npz"), np.load(f"{R}/T_train.npz")
idx = build_index(Str["keys"]); Dk, Ik = idx.search(hS.astype(np.float32), 64)


def mem_dist(tau, values):
    w = np.exp(-(Dk - Dk.min(1, keepdims=True)) / tau); w /= w.sum(1, keepdims=True)
    M = np.zeros_like(PS)
    rows = np.repeat(np.arange(len(M)), Ik.shape[1])
    if values == "gold":
        np.add.at(M, (rows, Str["gold"][Ik].ravel()), w.ravel())
    else:
        ids, pr = Ttr["ids"][Ik], Ttr["probs"][Ik].astype(np.float32)  # (Q,k,16)
        rows16 = np.repeat(rows, ids.shape[-1])
        np.add.at(M, (rows16, ids.reshape(-1)), (w[..., None] * pr).ravel())
    return M


def alpha(Pd):
    return np.minimum(Pd, PT).sum(1)


# cost ratio: per-token FLOPs ~ 2 * (non-embedding params + V*d for the output layer)
def flops(m):
    p = m["params"]; d = p["emb"].shape[1]
    return n_params(p) - p["emb"].size - p["pos"].size + V * d
c0 = flops(S) / flops(T)
print(f"student/teacher cost ratio c = {c0:.3f}", flush=True)

drafts = {"student": PS}
# self-speculation: the teacher's own first k layers (+ final LN + tied head) as draft
selfc = {}
for k in range(1, len(T["params"]["layers"])):
    pk = dict(pT); pk["layers"] = pT["layers"][:k]
    _, Pk = dist(pk, T["cfg"]["heads"])(X)
    drafts[f"teacher first {k} layers (self-draft)"] = np.asarray(Pk[:, lo:]).reshape(-1, V)
    pt = T["params"]; dd = pt["emb"].shape[1]; per_layer = (n_params(pt) - pt["emb"].size - pt["pos"].size) / len(pt["layers"])
    selfc[f"teacher first {k} layers (self-draft)"] = (k * per_layer + V * dd) / flops(T)
for lam in (0.1, 0.2, 0.3):
    for tau in (10, 30):
        Mg = mem_dist(tau, "gold"); drafts[f"student+mem[gold] lam{lam} tau{tau}"] = (1 - lam) * PS + lam * Mg
Mt = mem_dist(30, "teacher"); drafts["student+mem[teacher] lam0.3 tau30"] = 0.7 * PS + 0.3 * Mt
print(f"{'draft':>36} | alpha | greedy-match | best g | tokens/round | speed-up")
greedy_T = PT.argmax(1)
for name, Pd in drafts.items():
    c = selfc.get(name, c0)
    a = alpha(Pd); am = float(a.mean())
    best = max(((1 - am ** (g + 1)) / (1 - am) / (g * c + 1), g) for g in range(1, 9))
    E = (1 - am ** (best[1] + 1)) / (1 - am)
    print(f"{name:>36} | {am:.3f} | {float((Pd.argmax(1) == greedy_T).mean()):.3f}        | {best[1]}      | {E:.2f}         | {best[0]:.2f}x  (c={c:.2f})", flush=True)
