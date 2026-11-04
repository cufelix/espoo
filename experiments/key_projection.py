"""Experiment 11 (H4): is the failure of teacher-valued memory a KEY problem?

Hypothesis: memory indexed by the student's representation can't deliver the
teacher's extra knowledge, because the student maps contexts the teacher
distinguishes onto the same point. Tests:
  oracle   : student LM + memory searched with the TEACHER's true keys
             (upper bound; needs the teacher at query time)
  proj     : student LM + memory searched with a linear map W: S-key -> T-key
             fitted on train (ridge), so only the student runs at query time
Values: gold token or teacher soft labels. lam/tau tuned on valid, test reported.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from memlm.knn import build_index, neighbours, mem_prob, interp_ppl, tune

R = sys.argv[1]
ld = lambda n: dict(np.load(f"{R}/{n}.npz"))
Str, Ttr = ld("S_train"), ld("T_train")
Sva, Ste, Tva, Tte = ld("S_valid"), ld("S_test"), ld("T_valid"), ld("T_test")
assert np.array_equal(Sva["pos"], Tva["pos"]) and np.array_equal(Ste["pos"], Tte["pos"])
store = {"gold": Ttr["gold"], "ids": Ttr["ids"], "probs": Ttr["probs"]}

# ridge regression S -> T keys
rs = np.random.default_rng(0).choice(len(Str["keys"]), 400000, replace=False)
Xs = Str["keys"][rs].astype(np.float64); Yt = Ttr["keys"][rs].astype(np.float64)
mx, my = Xs.mean(0), Yt.mean(0)
A = (Xs - mx).T @ (Xs - mx) + 1.0 * np.eye(Xs.shape[1])
W = np.linalg.solve(A, (Xs - mx).T @ (Yt - my))
proj = lambda K: ((K.astype(np.float64) - mx) @ W + my).astype(np.float32)
P_va = proj(Sva["keys"]); Tv64 = Tva["keys"].astype(np.float64)
r2 = 1 - ((P_va - Tv64) ** 2).sum() / ((Tv64 - Tv64.mean(0)) ** 2).sum()  # float64: float16 sums overflow
print(f"linear S->T key map: R^2 on valid = {r2:.3f}", flush=True)

t = time.time(); idx = build_index(Ttr["keys"]); print(f"index {time.time()-t:.0f}s", flush=True)
print(f"student alone test ppl {np.exp(-Ste['lp'].mean()):.2f}", flush=True)
for qname, qva, qte in (("oracle T-keys", Tva["keys"], Tte["keys"]), ("projected S-keys", P_va, proj(Ste["keys"]))):
    Dv, Iv = neighbours(idx, qva); Dt, It = neighbours(idx, qte)
    for values in ("gold", "teacher"):
        vppl, lam, tau = tune(Sva["lp"], Dv, Iv, Sva["gold"], values, store)
        pm = mem_prob(Dt, It, Ste["gold"], values, tau, store)
        print(f"student + mem[{values:>7}] via {qname:>16} | test ppl {interp_ppl(Ste['lp'], pm, lam):6.2f} (lam {lam}, tau {tau})", flush=True)
