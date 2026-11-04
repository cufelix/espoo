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
