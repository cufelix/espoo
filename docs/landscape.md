# The field from above: where scaling is stuck and what can unstick it

Based on the 411-paper catalog in `docs/lit/` (39 topic queries) plus the
primary sources cited in earlier docs. Ids are arXiv numbers.

## 1. The three walls
| wall | evidence | consequence |
|---|---|---|
| **Data** | Human text ≈ 9–27T tokens may be exhausted by 2026–28 (LLMOrbit survey, 2601.14053). Repeating data has diminishing returns (2305.16264, 2605.01640). Synthetic data risks collapse (2404.01413, 2402.07043, 2609.40295). | "More pretraining data" stops being a free lever. |
| **Compute / cost** | Compute-optimal laws (2001.08361, Chinchilla) plus inference-aware laws (2401.00448) push toward smaller, longer-trained models. Test-time compute scaling shifts cost to inference (2601.05593, 2509.04475) and can even hurt (2507.14417). | Every inference FLOP matters, and inference dominates lifetime cost. |
| **Memory bandwidth / energy** | FLOPs grew ~3x per 2 years, DRAM bandwidth ~1.6x ("AI and Memory Wall"). Decoding is bandwidth-bound. Energy per query is the deployment limit (2601.22076, 2311.16863). | Moving weights, not multiplying them, is the cost. |

## 2. What the main lines of work buy (and what they cost)
- **Quantization / pruning / low-rank** (QTIP, AQLM, GPTQ, SqueezeLLM, SVD-LLM, Swift-SVD 2604.01609, …):
  2–4x post-hoc, near-lossless. More only with training. Our RoBERTa
  measurements agree: 4 bits ≈ ok, 2 bits collapses.
- **Tensor networks / "quantum-inspired"** (CompactifAI 2401.14109, TT/MPO
  papers): our experiments found no hidden structure in trained weights;
  gains require retraining.
- **MoE** (ST-MoE 2202.08906, 2501.12370): decouples parameters from FLOPs,
  but all parameters still have to sit in fast memory.
- **Speculative decoding** (2211.17192, Kangaroo 2404.18911, DistillSpec
  2310.08461): exactly lossless 2–3x speed. Bounded by acceptance rate.
- **SSMs / linear attention** (Mamba line, ATLAS 2505.23735): cheaper long
  context, but don't change where knowledge lives.
- **Retrieval / non-parametric memory**: RETRO (25x fewer params for equal Pile
  perplexity), kNN-LM (1911.00172), MassiveDS (2407.12854: datastore scaling
  with no saturation, better at equal training compute), "To memorize or to
  retrieve" (2604.00715: 3-D scaling laws over model, pretraining data and
  retrieval data), memory layers (2412.09764), Mixture-of-Memories
  (2610.25425). **This is the only line that changes where knowledge is stored.**
  Facts move from expensive fast memory (HBM) to cheap storage (disk). The
  datastore also grows without retraining.

## 3. The gap
Retrieval stores **raw text**: what the corpus says. Distillation stores a
big model's **judgement** (its soft predictions), but squeezes it into a small
model's weights, where capacity is the limit. Nobody in the catalog stores the
teacher's judgement in the retrieval memory itself:
- kNN-LM: the model's own keys, gold-token values.
- RetriKT (2310.15594): the LLM generates text knowledge for GLUE-style tasks.
- DistillSpec / KD: the teacher only shapes the student's weights.
- Instruction retrieval (2510.13935): the teacher writes reusable guides.

## 4. The idea: Distillation into Memory (DiM)
Run the big teacher **once, offline** over a corpus and store, per position,
its top-k next-token distribution as the *value* of a kNN memory whose *keys*
are the small student's hidden states. At inference only the small model runs.
It looks up its neighbours and mixes in the teacher's stored judgement.
- Capacity of the "distilled knowledge" scales with **disk**, not with student
  parameters.
- New knowledge or a better teacher: re-run the teacher over text and append.
  The student is never retrained.
- Cost: one teacher pass per stored token (amortised), k × (id, prob) bytes per
  token, plus a nearest-neighbour search per generated token.

Verified at small scale in `docs/results-memory.md` (WikiText-2, JAX on CPU).
