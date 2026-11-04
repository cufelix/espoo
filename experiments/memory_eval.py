"""Experiment 9: 'Distillation into Memory' vs. distillation into weights.

Reads dumps produced by memlm.dump for teacher (T), student (S) and the
knowledge-distilled student (S_kd). Datastore = WikiText-2 train split.
Values: gold token (kNN-LM) or the TEACHER's top-16 soft distribution.
lam, tau tuned on valid; test perplexity reported.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, ".")
from memlm.knn import build_index, neighbours, mem_prob, interp_ppl, tune

R = sys.argv[1]
ld = lambda n: dict(np.load(f"{R}/{n}.npz"))
T_tr = ld("T_train")
teacher_vals = {"ids": T_tr["ids"], "probs": T_tr["probs"]}
rows = []


def evaluate(qname, fracs=(1.0,), values_list=("gold", "teacher", "mix")):
    tr, va, te = ld(f"{qname}_train"), ld(f"{qname}_valid"), ld(f"{qname}_test")
    assert np.array_equal(tr["pos"], T_tr["pos"])
    base = float(np.exp(-te["lp"].mean()))
    rows.append(dict(query=qname, values="-", frac=0, test_ppl=base))
    print(f"{qname:>5} alone: test ppl {base:.2f}", flush=True)
    for frac in fracs:
        n = int(frac * len(tr["keys"]))
        store = {"gold": tr["gold"][:n], "ids": teacher_vals["ids"][:n], "probs": teacher_vals["probs"][:n]}
        t = time.time(); idx = build_index(tr["keys"][:n])
        Dv, Iv = neighbours(idx, va["keys"]); Dt, It = neighbours(idx, te["keys"])
        for values in values_list:
            alphas = (0.3, 0.5, 0.7) if values == "mix" else (1.0,)
            best = None
            for al in alphas:
                vppl, lam, tau = tune(va["lp"], Dv, Iv, va["gold"], values, store, al)
                if best is None or vppl < best[0]:
                    best = (vppl, lam, tau, al)
            vppl, lam, tau, al = best
            pm = mem_prob(Dt, It, te["gold"], values, tau, store, al)
            tppl = interp_ppl(te["lp"], pm, lam)
            rows.append(dict(query=qname, values=values, frac=frac, lam=lam, tau=tau, alpha=al, val_ppl=vppl, test_ppl=tppl))
            print(f"{qname:>5} + mem[{values:>7}] store {frac:4.0%} | test ppl {tppl:7.2f} (lam {lam}, tau {tau}, alpha {al}) [{time.time()-t:.0f}s]", flush=True)


evaluate("T", values_list=("gold",))
evaluate("S", fracs=(0.1, 0.3, 1.0))
evaluate("Skd", fracs=(1.0,))
json.dump(rows, open(f"{R}/memory_results.json", "w"), indent=1)
