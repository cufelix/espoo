"""kNN memory evaluation: plain kNN-LM vs. 'distillation into memory'.

Datastore (train split): keys from the QUERY model, values either
  gold    : the true next token (kNN-LM, Khandelwal et al. 2020)
  teacher : the teacher's stored top-k next-token distribution at that position
  mix     : alpha * gold + (1 - alpha) * teacher
p(y) = lam * p_mem(y) + (1 - lam) * p_model(y),  p_mem = sum_n softmax(-d_n / tau) v_n(y)
lam and tau are tuned on validation, results reported on test.
"""
import numpy as np, faiss


def build_index(keys, nlist=2048, seed=0):
    d = keys.shape[1]; q = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(q, d, nlist)
    rs = np.random.default_rng(seed).choice(len(keys), min(len(keys), 150000), replace=False)
    idx.train(keys[rs].astype(np.float32)); idx.add(keys.astype(np.float32)); idx.nprobe = 16
    return idx


def neighbours(idx, q, k=64):
    D, I = idx.search(q.astype(np.float32), k)
    return D, I


def mem_prob(D, I, gold_q, values, tau, store, alpha=1.0):
    """probability the memory assigns to each query's gold token."""
    w = np.exp(-(D - D.min(1, keepdims=True)) / tau); w /= w.sum(1, keepdims=True)
    hit_gold = (store["gold"][I] == gold_q[:, None]).astype(np.float32)
    if values == "gold":
        v = hit_gold
    else:
        ids, pr = store["ids"][I], store["probs"][I].astype(np.float32)        # (Q,k,K)
        vt = (pr * (ids == gold_q[:, None, None])).sum(-1)
        v = vt if values == "teacher" else alpha * hit_gold + (1 - alpha) * vt
    return (w * v).sum(1)


def interp_ppl(lp_model, pm, lam):
    p = lam * pm + (1 - lam) * np.exp(lp_model)
    return float(np.exp(-np.log(np.maximum(p, 1e-12)).mean()))


def tune(lp_val, D, I, gold, values, store, alpha=1.0):
    best = (float(np.exp(-lp_val.mean())), 0.0, 1.0)
    for tau in (0.3, 1, 3, 10, 30, 100, 300, 1000):
        pm = mem_prob(D, I, gold, values, tau, store, alpha)
        for lam in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            ppl = interp_ppl(lp_val, pm, lam)
            if ppl < best[0]:
                best = (ppl, lam, tau)
    return best
