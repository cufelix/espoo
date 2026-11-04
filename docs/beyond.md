# Beyond compression: radically different ways to store and run a model

## Tested here: just-in-time specialisation (`experiments/roberta_jit.py`)
Idea: per conversation, "recompile" the model into a much smaller one that
covers only the subspace this context uses.

| FFN kept | generic calibration | same book (in-context) | eval text itself (oracle) |
|---|---|---|---|
| 10 % | 0.44 | 0.49 | 0.60 |
| 25 % | 0.73 | 0.75 | 0.86 |
| 50 % | 0.87 | 0.87 | 0.94 |

Same-document calibration barely helps. Even the oracle needs 50 % of the
FFN for 0.94. The space a model uses shifts **token by token, not document
by document**, so a static per-context mini-model does not give 10x. Negative
result.

## Paradigms that change *where* intelligence lives
1. **Knowledge in a database, reasoning in weights** (retrieval-native models).
   Weights currently store facts *and* skills. Facts can live in a text/vector
   store on cheap disk and be looked up per token. RETRO (DeepMind 2021)
   reported GPT-3-level perplexity with ~25x fewer parameters. This is the
   strongest existing evidence for >10x without quality loss, and it is not
   compression at all. Open problem: making the small reasoner as good at
   reasoning as a large one.
2. **Never compute the same thing twice** (global activation memoisation).
   Huge parts of real traffic are near-duplicates (system prompts, documents,
   code). Prefix caching already reuses the KV cache. A step further: a shared
   semantic cache of intermediate states, so new requests reuse computation
   from similar old ones. The gain depends on traffic, could be large, and
   exactness needs verification.
3. **Compute where the weights are** (in-memory / analog / optical compute).
   Most energy goes to *moving* weights, not multiplying them. Analog
   in-memory chips claim 10–100x energy efficiency. Accuracy suffers from analog
   noise, and that is the open research problem. Hardware, not software.
4. **Program-generated weights** (weights from seeds + a generator, cf. SeedLM).
   Our data says trained weights look high-rank and noise-like, and noise has
   no short program. Likely a dead end for large factors.
5. **Grow, don't store** (models that fetch and assemble experts on demand from
   a large library, like loading code modules). Mixture-of-experts at the
   scale of whole sub-networks over the network. Compute per request is small.
   Quality depends on routing, which our routing experiment showed is the
   hard part.

## Ranking
Retrieval-native models (1) > memoisation (2) > hardware (3) > the rest.
Only (1) has published evidence of >10x parameter reduction at similar
quality. If we pursue something "borderless", it should be (1), because it changes
what a model *is* instead of squeezing what it already is.
