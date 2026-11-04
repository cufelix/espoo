# Brainstorm: what could actually move the field

Grounded in what our experiments showed (docs/results.md):
- Trained weights are **high-rank with no hidden tensor-network structure**
  over neurons. "Quantum-inspired" compression of raw weights loses to plain SVD.
- What matters is the **data/activation subspace**: activation-whitened SVD
  (ASVD) beats every MPO variant by a wide margin.
- Routing alone always loses quality. **Verification** (speculative decoding)
  is the only way to get "dynamic and lossless".

Lesson: compress the *function on real inputs*, not the weights, and get
exactness from verification, not from the approximation.

## Ideas, ranked (impact × feasibility)

### 1. Matryoshka Self-Speculation (MSS)  ★ pick
One model file, many models inside it. Every matrix is stored in an
importance-ordered form (activation-whitened singular directions, most
important first), so **any prefix is a valid smaller model**.
- Draft = rank-k prefix (cheap, in fast memory). Verify = full model.
  Output is exactly the full model's (speculative decoding guarantee).
- k is chosen **per token, dynamically** from the recent acceptance rate:
  easy text → tiny prefix, hard text → bigger prefix.
- Memory tiering: the prefix sits in GPU/RAM, the tail is streamed from
  SSD/CPU and only needed in verification steps, which are batched over many
  tokens. Large models then run on small hardware *without* quality loss.
- Lossless entropy coding of the stored factors (DFloat11-style) on top.
Novelty: self-speculation exists (LayerSkip, Draft&Verify via layer skipping)
and Matryoshka/nested models exist, but a **single activation-ordered nested
factorisation used as a dynamic draft with tiered memory** is, as far as I
found, not an established method. Needs checking against prior work.
Risk: the factor overhead (storing A·B instead of W) eats memory unless the
tail is offloaded. Acceptance rate of low-rank drafts is unknown → **first
experiment**: measure acceptance on GPT-2/Qwen-0.5B for k = 10–50 %.

### 2. Function-space distillation into structured students
Since weights don't have structure but the function on data is
low-dimensional, distil directly into an MPO/low-rank/ternary student using
activations as targets (layer-wise, then end-to-end). This is what
CompactifAI's "healing" implicitly does. Needs a GPU.

### 3. Verified cascades for whole tasks (not tokens)
A router sends a request to the smallest model that can *prove* its answer
(unit tests for code, a checker for math, self-consistency for facts) and
escalates otherwise. Exactness comes from task verifiers, not from a
probability guarantee. Practical, but the router/verifier is task-specific.

### 4. Memory layers instead of FFN
Replace dense FFN with huge sparse key-value memories (product keys): compute
per token is tiny and capacity is enormous, and rows can live on SSD. This is
trained in, not retrofitted.

### 5. Quantum hardware, honestly
The only realistic near-term role: small combinatorial sub-problems (e.g.
choosing which experts/ranks go to which memory tier, a QUBO). No evidence
of advantage over classical solvers. Not where a revolution will come from
in the next years.

## Next concrete step
Experiment for MSS on a decoder model: per-layer activation-whitened SVD,
draft = rank prefix, measure (a) token acceptance rate and (b) expected speedup
vs k. Needs GPT-2 or Qwen2.5-0.5B weights (huggingface.co access).
