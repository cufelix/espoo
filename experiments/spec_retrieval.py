"""Experiment 13 (H9): zero-cost drafts copied from memory (cf. REST, prompt lookup).

For a test position, find the nearest train position by the student's key and
propose the g tokens that followed it in the train text. The draft costs no
model compute (only a kNN search). The teacher verifies, so output is exactly
the teacher's. For a deterministic draft token d the acceptance probability is
p_T(d | context). Expected accepted run E = sum_i prod_{j<=i} a_j, using
contexts from the real text (a_j counted only while previous draft tokens equal
the real text: a conservative lower bound). Each round also yields one token
from the teacher itself, so tokens per teacher pass = 1 + E.
Also: 'best of both': student draft when the memory neighbour is far, memory
draft when it is close.
"""
import sys, pickle
sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp
from memlm.data import load_wt2
from memlm.model import hidden
from memlm.knn import build_index

R, D = sys.argv[1], sys.argv[2]
tr, va, te, vocab = load_wt2(D); V = len(vocab)
T, S = (pickle.load(open(f"{R}/{n}.pkl", "rb")) for n in ("teacher", "student"))
pT = jax.tree_util.tree_map(jnp.asarray, T["params"]); pS = jax.tree_util.tree_map(jnp.asarray, S["params"])
C = 64; G = 8
fT = jax.jit(lambda x: jax.nn.softmax(hidden(pT, x, 4) @ pT["emb"].T, -1))
fS = jax.jit(lambda x: hidden(pS, x, 4))
starts = np.random.default_rng(1).choice(len(te) - C - G - 1, 300, replace=False)
X = np.stack([te[s:s + C + G] for s in starts])          # windows with G extra tokens
PT = np.asarray(fT(X[:, :C + G - 1]))                    # (B, C+G-1, V): p_T(next | prefix up to t)
HS = np.asarray(fS(X[:, :C]))[:, C // 2:]                # student keys at positions 32..63
Str = np.load(f"{R}/S_train.npz"); idx = build_index(Str["keys"])
Q = HS.reshape(-1, HS.shape[-1]).astype(np.float32)
Dk, Ik = idx.search(Q, 1)
train_pos = Str["pos"][Ik[:, 0]]                         # neighbour's train position (predicts tr[pos+1])
B, P = len(X), HS.shape[1]
E = np.zeros(B * P); dist = Dk[:, 0]
for q in range(B * P):
    b, t = divmod(q, P); t += C // 2                      # absolute position in window
    draft = tr[train_pos[q] + 1: train_pos[q] + 1 + G]
    run, prod = 0.0, 1.0
    for i, d in enumerate(draft):
        if t + i >= C + G - 1:
            break
        prod *= PT[b, t + i, d]; run += prod
        if d != X[b, t + i + 1]:
            break                                         # context diverges: stop (lower bound)
    E[q] = run
print(f"{B*P} positions; memory draft: mean extra accepted tokens per teacher pass = {E.mean():.3f}")
print(f"tokens per teacher pass (memory draft, cost ~0): {1 + E.mean():.2f}x")
for qt in (10, 25, 50):
    th = np.percentile(dist, qt)
    print(f"  closest {qt}% neighbours (d<{th:.0f}): extra tokens {E[dist < th].mean():.3f}")
