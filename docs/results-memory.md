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

## H8: cheap drafts with a frequency-restricted output layer (cf. FR-Spec): CONFIRMED
The draft only scores the N most frequent words. The target keeps the full
vocabulary, so the output is still exactly the teacher's.

| draft | N = 10k (full) | 4k | 2k | **1k** | 500 |
|---|---|---|---|---|---|
| student: α / cost / speed-up | 0.725 / 0.29 / 1.42x | 0.704 / 0.16 / 1.73x | 0.679 / 0.11 / 1.83x | 0.648 / 0.09 / **1.84x** | 0.613 / 0.08 / 1.79x |
| teacher first 1 layer | 0.96x | 1.15x | 1.22x | 1.27x | 1.28x |
| teacher first 2 layers | 0.94x | 1.11x | 1.17x | 1.19x | 1.19x |
| teacher first 3 layers | 0.97x | 1.11x | 1.15x | 1.16x | 1.15x |

Restricting the draft head to 10 % of the vocabulary costs only 0.08 in
acceptance but cuts draft cost 3x. **Lossless speed-up 1.42x → 1.84x.**

## H9: zero-cost drafts copied from memory (cf. REST / prompt lookup): SMALL GAIN
Draft = the 8 tokens that followed the nearest train position. Conservative
estimate: **1.16 teacher-quality tokens per teacher pass**, rising to 1.32
when the neighbour is very close (closest 10 %). Useful as a free add-on, not
on its own.

## H10: a distilled draft is a better draft (cf. DistillSpec): CONFIRMED
| draft | α (full vocab) | speed-up (full) | α (1k vocab) | speed-up (1k) |
|---|---|---|---|---|
| student | 0.725 | 1.42x | 0.648 | 1.84x |
| **student distilled from teacher** | **0.758** | 1.47x | 0.669 | **1.92x** |

## H12: self-speculation from the target's own bit-planes: CONFIRMED (prior art exists)
Target = teacher. Draft = the same weights rounded to fewer bits. Cost =
bytes read (bandwidth-bound decoding).

| draft | target | α | draft cost | speed-up |
|---|---|---|---|---|
| independent 4-bit RTN copy | fp16 | 0.937 | 0.25 | **2.35x** |
| top 4 of the stored 8-bit codes (nested, no extra memory) | fp16 | 0.937 | 0.25 | 2.34x |
| top 3 of the stored 8-bit codes | 8-bit stored model | 0.870 | 0.375 | 1.55x |

- Nested MSB prefixes are as good as a separately quantised copy, so the
  draft can live *inside* the stored weights.
- The 8-bit stored model itself agrees with fp32 on 99.6 % of greedy tokens
  (α = 0.996). **Combined: 8-bit storage (2x fewer bytes than fp16) × 1.55x
  exact self-speculation ≈ 3.1x faster than fp16 decoding**, with output
  identical to the 8-bit model and 99.6 % greedy-identical to full precision.
- **Prior art:** BitNest (arXiv 2610.02800, Oct 2026), BS-Spec (2610.22877) and
  SPEQ already propose bit-nested self-drafts. Our numbers are an independent,
  small-scale confirmation, not a new method.

## Distillation into weights vs. into memory (final)
| model | test ppl | share of student→teacher gap closed |
|---|---|---|
| student | 68.68 | 0 % |
| student + teacher-valued memory (DiM) | 68.68 | 0 % |
| student + gold memory (kNN-LM) | 65.15 | 40 % |
| student distilled from teacher (KD, into weights) | 67.31 | 16 % |
| KD student + teacher-valued memory | 67.31 | 16 % |
| KD student + mix memory | 64.60 | 47 % |
| **KD student + gold memory** | **63.81** | **56 %** |
| teacher (3.4x more params) | 59.93 | 100 % |

**Best combination: distil the teacher's judgement into the student's weights
and put memorised specifics into the memory.** Each part does what it is good
at. A 3.4x smaller model then closes 56 % of the quality gap.
