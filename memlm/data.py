"""WikiText-2 word-level data (standard LM benchmark, <eos> per line)."""
import numpy as np


def load_wt2(d, max_vocab=10000):
    def toks(split):
        out = []
        for line in open(f"{d}/{split}.txt", encoding="utf-8"):
            w = line.split()
            if w:
                out += w + ["<eos>"]
        return out
    tr, va, te = toks("train"), toks("valid"), toks("test")
    # keep the max_vocab most frequent training words; the rest -> <unk> (CPU budget)
    from collections import Counter
    keep = [w for w, _ in Counter(tr).most_common() if w not in ("<unk>", "<eos>")][: max_vocab - 2]
    vocab = {w: i for i, w in enumerate(["<unk>", "<eos>"] + sorted(keep))}
    enc = lambda t: np.array([vocab.get(w, vocab["<unk>"]) for w in t], dtype=np.int32)
    return enc(tr), enc(va), enc(te), vocab
