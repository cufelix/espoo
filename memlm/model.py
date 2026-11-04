"""Small decoder-only transformer LM in JAX (pre-LN, GELU, tied embeddings)."""
import jax, jax.numpy as jnp
import numpy as np


def init(key, V, d, L, ctx, ff_mult=4):
    ks = iter(jax.random.split(key, 4 + 6 * L))
    n = lambda shape, s=0.02: jax.random.normal(next(ks), shape) * s
    p = {"emb": n((V, d)), "pos": n((ctx, d)), "lnf_g": jnp.ones(d), "lnf_b": jnp.zeros(d), "layers": []}
    for _ in range(L):
        p["layers"].append({
            "ln1_g": jnp.ones(d), "ln1_b": jnp.zeros(d), "ln2_g": jnp.ones(d), "ln2_b": jnp.zeros(d),
            "qkv": n((d, 3 * d)), "o": n((d, d), 0.02 / np.sqrt(2 * L)),
            "w1": n((d, ff_mult * d)), "b1": jnp.zeros(ff_mult * d),
            "w2": n((ff_mult * d, d), 0.02 / np.sqrt(2 * L)), "b2": jnp.zeros(d)})
    return p


def ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-5) * g + b


def hidden(p, x, n_heads):
    """x: (B,T) int -> final hidden states (B,T,d) (after final LN): the kNN key."""
    B, T = x.shape
    h = p["emb"][x] + p["pos"][:T]
    d = h.shape[-1]; dh = d // n_heads
    mask = jnp.tril(jnp.ones((T, T), bool))
    for l in p["layers"]:
        z = ln(h, l["ln1_g"], l["ln1_b"])
        q, k, v = jnp.split(z @ l["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, n_heads, dh).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        s = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh)
        s = jnp.where(mask, s, -1e9)
        a = jax.nn.softmax(s, -1) @ v
        h = h + a.transpose(0, 2, 1, 3).reshape(B, T, d) @ l["o"]
        z = ln(h, l["ln2_g"], l["ln2_b"])
        h = h + jax.nn.gelu(z @ l["w1"] + l["b1"]) @ l["w2"] + l["b2"]
    return ln(h, p["lnf_g"], p["lnf_b"])


def logits(p, x, n_heads):
    return hidden(p, x, n_heads) @ p["emb"].T


def n_params(p):
    return int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(p)))
