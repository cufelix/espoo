# Night experiments: memory, distillation, speculation (WikiText-2, JAX on CPU)

Setup: WikiText-2 word level, vocabulary capped at the 10k most frequent words
(CPU budget), context 64, decoder-only transformers trained from scratch.

| model | params | test ppl |
|---|---|---|
| teacher T (d=256, 4 layers) | 5.73M | 59.93 |
| student S (d=128, 2 layers) | 1.68M | 68.68 |

(Perplexities are scored with ≥32 tokens of context, so they are slightly lower
than the training-script numbers.)

## H1: "Distillation into Memory" (teacher soft labels as kNN values): REJECTED
Datastore = 2.08M train positions, keys = student hidden states.

| student + memory | 10 % store | 30 % store | 100 % store |
|---|---|---|---|
| values = gold next token (kNN-LM) | 68.00 | 67.11 | **65.15** |
| values = teacher top-16 soft labels (DiM) | 68.68 (no gain) | 68.68 | 68.68 |
| mix 0.7·gold + 0.3·teacher | 68.22 | 67.53 | see below |

The teacher-valued memory gives **no improvement at any size**: tuning picks
λ = 0. A debug check confirmed the data is right. Teacher p(gold) at train
positions averages 0.16 and gold is in the top-16 70 % of the time.

**Why (our interpretation):** the memory is searched with the *student's*
representation. The teacher's advantage is exactly its ability to tell apart
contexts that the student maps to the same point. Neighbours found in student
space are contexts where the teacher's soft prediction says roughly what the
student already predicts. Gold-token memory still helps because it adds
*specific memorised continuations* (the classic kNN-LM effect), which grows
with store size (68.0 → 67.1 → 65.2 as the store goes 10 % → 30 % → 100 %).
H4 below tests this explanation directly.

## H4: is it a key problem? Mostly not
| student LM + memory searched with … | values = gold | values = teacher |
|---|---|---|
| student's own keys | 65.15 | 68.68 (no gain) |
| linear map student→teacher keys (R² = 0.66) | 66.58 | 68.68 |
| **teacher's true keys** (oracle, needs the teacher) | **63.09** | 68.68 |

- Better keys help the *gold* memory (65.2 → 63.1), so retrieval quality matters.
- Teacher-valued memory gives **nothing even with the teacher's own keys**. A
  sanity check showed the mechanism is exact: queried at a stored position it
  returns the teacher's own p(gold). The reason is that neighbours are not
  close enough (median squared distance ≈ 150 vs. squared norm ≈ 460), so the
  average of their teacher predictions is a *blurred teacher*. That adds no
  information beyond the student's own smooth prediction.
- **Conclusion: non-parametric memory pays off for memorised specifics
  (what actually followed), not for generalisation (a model's judgement).**
  Judgement has to be computed, not looked up.

## H2/H3: lossless speed-up with speculative decoding
Teacher = target, so output is exactly the teacher's. Acceptance α measured
teacher-forced on 9,600 test positions. Cost c in teacher-forward units (FLOPs).

| draft | α | cost c | best draft length | speed-up |
|---|---|---|---|---|
| **student (3.4x fewer params)** | 0.725 | 0.29 | 2 | **1.42x** |
| student + gold memory | 0.719–0.723 | 0.29 | 2 | 1.41–1.42x |
| student + teacher-valued memory | 0.703 | 0.29 | 2 | 1.39x |
| teacher first 1 layer (self-draft) | 0.527 | 0.59 | 1 | 0.96x |
| teacher first 2 layers | 0.626 | 0.72 | 1 | 0.94x |
| teacher first 3 layers | 0.804 | 0.86 | 1 | 0.97x |

- A separate small draft gives a real, exactly lossless 1.42x.
- Memory does not improve drafts.
- Self-drafts lose here because the 10k-word output layer is >40 % of the
  cost at this tiny width. In large LLMs the output layer is a few % of the
  cost, so this is a small-scale artefact. H8 tests removing it.
