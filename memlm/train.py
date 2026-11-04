"""Train a small LM on WikiText-2. Optional knowledge distillation from a
teacher's stored top-k soft labels.

usage: python -m memlm.train --d 128 --L 2 --out student.npz [--kd teacher_topk.npz]
"""
import argparse, time, pickle
import numpy as np, jax, jax.numpy as jnp, optax
from .data import load_wt2
from .model import init, logits, n_params

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--d", type=int, default=128); ap.add_argument("--L", type=int, default=2)
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ctx", type=int, default=64)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--dropout_tok", type=float, default=0.0)
ap.add_argument("--kd", default=None); ap.add_argument("--kd_w", type=float, default=0.5)
ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tr, va, te, vocab = load_wt2(a.data); V = len(vocab)
key = jax.random.PRNGKey(a.seed)
p = init(key, V, a.d, a.L, a.ctx)
print(f"V={V} params={n_params(p)/1e6:.2f}M train tokens={len(tr)}", flush=True)
sched = optax.warmup_cosine_decay_schedule(0, a.lr, min(300, a.steps // 10), a.steps, a.lr * 0.05)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=a.wd))
st = opt.init(p)
kd = np.load(a.kd) if a.kd else None  # teacher top-k at train position i (predicting tr[i+1])
if kd is not None:
    kd_ids, kd_probs = kd["kd_ids"], kd["kd_probs"]


def loss_fn(p, x, y, kid, kpr):
    lg = logits(p, x, a.heads)
    lp = jax.nn.log_softmax(lg, -1)
    nll = -jnp.take_along_axis(lp, y[..., None], -1).mean()
    if kd is None:
        return nll, nll
    soft = -(kpr * jnp.take_along_axis(lp, kid, -1)).sum(-1).mean() / kpr.sum(-1).mean()
    return (1 - a.kd_w) * nll + a.kd_w * soft, nll


@jax.jit
def step(p, st, x, y, kid, kpr):
    (l, nll), g = jax.value_and_grad(loss_fn, has_aux=True)(p, x, y, kid, kpr)
    up, st = opt.update(g, st, p)
    return optax.apply_updates(p, up), st, nll


@jax.jit
def eval_nll(p, x, y):
    lp = jax.nn.log_softmax(logits(p, x, a.heads), -1)
    return -jnp.take_along_axis(lp, y[..., None], -1)


def evaluate(p, data, max_tokens=60000):
    # non-overlapping windows; every token scored with up to ctx-1 context
    n = min(len(data) - 1, max_tokens) // a.ctx * a.ctx
    X = data[:n].reshape(-1, a.ctx); Y = data[1:n + 1].reshape(-1, a.ctx)
    tot = 0.0
    for i in range(0, len(X), 64):
        tot += float(eval_nll(p, X[i:i + 64], Y[i:i + 64]).sum())
    return float(np.exp(tot / n))


rng = np.random.default_rng(a.seed); t0 = time.time()
dummy_id = np.zeros((a.bs, a.ctx, 1), np.int32); dummy_pr = np.zeros((a.bs, a.ctx, 1), np.float32)
for s in range(1, a.steps + 1):
    idx = rng.integers(0, len(tr) - a.ctx - 1, a.bs)
    x = np.stack([tr[i:i + a.ctx] for i in idx]); y = np.stack([tr[i + 1:i + a.ctx + 1] for i in idx])
    if kd is not None:
        pos = idx[:, None] + np.arange(a.ctx)[None]
        kid, kpr = kd_ids[pos], kd_probs[pos].astype(np.float32)
    else:
        kid, kpr = dummy_id, dummy_pr
    p, st, nll = step(p, st, x, y, kid, kpr)
    if s % 500 == 0 or s == a.steps:
        print(f"step {s} train_nll {float(nll):.3f} val_ppl {evaluate(p, va, 20000):.1f} ({time.time()-t0:.0f}s)", flush=True)
vp, tp = evaluate(p, va), evaluate(p, te)
print(f"FINAL val_ppl {vp:.2f} test_ppl {tp:.2f} params {n_params(p)/1e6:.2f}M", flush=True)
pickle.dump({"params": jax.device_get(p), "cfg": vars(a), "val_ppl": vp, "test_ppl": tp}, open(a.out, "wb"))
