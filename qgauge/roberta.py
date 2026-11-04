"""Tiny numpy RoBERTa-base (post-LN, GELU) + GPT-2 byte-level BPE tokenizer,
matching curated-transformers 0.1 (the encoder inside spaCy en_core_web_trf).

Weights: produced by tools/torchzip_to_npz.py. Tokenizer: bpe.pkl with
{'merges': [[a, b], ...], 'vocab': {piece: id}} from the same wheel.
"""
import pickle
import numpy as np
import regex as re
from scipy.special import erf

PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
PFX = "curated_encoder."


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    def __init__(self, path):
        d = pickle.load(open(path, "rb"))
        self.vocab = d["vocab"]
        self.ranks = {tuple(m): i for i, m in enumerate(d["merges"])}
        self.b2u = _bytes_to_unicode()
        self.cache = {}

    def _bpe(self, w):
        if w in self.cache:
            return self.cache[w]
        parts = list(w)
        while len(parts) > 1:
            pairs = [(self.ranks.get((a, b), 1e18), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            r, i = min(pairs)
            if r == 1e18:
                break
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        self.cache[w] = parts
        return parts

    def encode(self, text):
        ids = []
        for tok in PAT.findall(text):
            u = "".join(self.b2u[b] for b in tok.encode("utf-8"))
            ids += [self.vocab.get(p, self.vocab["<unk>"]) for p in self._bpe(u)]
        return ids


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


class Roberta:
    def __init__(self, sd, n_heads=12):
        self.sd = {k: np.asarray(sd[k], dtype=np.float32) for k in sd.files} if hasattr(sd, "files") else dict(sd)
        self.H = n_heads
        self.L = 1 + max(int(k.split(".")[2]) for k in self.sd if ".layers." in k)

    def w(self, name):
        return self.sd[PFX + name]

    def forward(self, ids, hook=None, override=None):
        """ids: (B, T) int with <s> ... </s>, no padding.
        hook(layer, name, x): called with inputs of FFN matrices.
        override: dict param_name -> callable(x) replacing x @ W.T (bias added separately)."""
        override = override or {}
        B, T = ids.shape
        e = "embeddings.inner."
        pos = np.arange(T)[None, :] + 2  # padding_idx(1) + cumsum
        x = self.w(e + "word_embeddings.weight")[ids] + self.w(e + "token_type_embeddings.weight")[0] + self.w(e + "position_embeddings.weight")[pos]
        x = layernorm(x, self.w(e + "layer_norm.weight"), self.w(e + "layer_norm.bias"))

        def lin(name, z, layer):
            key = f"layers.{layer}.{name}"
            if hook:
                hook(layer, name, z)
            f = override.get(key)
            y = f(z) if f else z @ self.w(key + ".weight").T
            return y + self.w(key + ".bias")

        d = x.shape[-1]; dh = d // self.H
        for l in range(self.L):
            p = f"layers.{l}."
            qkv = lin("mha.input", x, l)
            q, k, v = np.split(qkv, 3, axis=-1)
            sh = lambda t: t.reshape(B, T, self.H, dh).transpose(0, 2, 1, 3)
            q, k, v = sh(q), sh(k), sh(v)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            a = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            a = lin("mha.output", a, l)
            x = layernorm(x + a, self.w(p + "attn_output_layernorm.weight"), self.w(p + "attn_output_layernorm.bias"))
            h = gelu(lin("ffn.intermediate", x, l))
            f = lin("ffn.output", h, l)
            x = layernorm(x + f, self.w(p + "ffn_output_layernorm.weight"), self.w(p + "ffn_output_layernorm.bias"))
        return x


def load_corpus(tok, path, n_seq, seq_len, offset=0):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ids = tok.encode(text[offset: offset + n_seq * seq_len * 8])
    need = n_seq * (seq_len - 2)
    assert len(ids) >= need, (len(ids), need)
    ids = np.array(ids[:need]).reshape(n_seq, seq_len - 2)
    bos, eos = tok.vocab["<s>"], tok.vocab["</s>"]
    return np.concatenate([np.full((n_seq, 1), bos), ids, np.full((n_seq, 1), eos)], 1)
