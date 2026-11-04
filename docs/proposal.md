# Q-Gauge: návrh metody

## Hypotéza

Natrénované váhy LLM mají nízký „entanglement“ (= jsou dobře stlačitelné do
MPO s malou bond dimension χ). Tato struktura je ale **skrytá za libovolnou
volbou báze**: pořadím FFN neuronů, pořadím attention hlav a rotací residual
streamu. Když tuto volbu („gauge“) zafixujeme tak, aby minimalizovala
entanglement přes řezy MPO, dostaneme při stejné přesnosti výrazně menší
model, než dává MPO komprese bez ní (CompactifAI-styl).

## Pipeline

1. **Gauge fixing (bez ztráty přesnosti)**
   - Permutace FFN neuronů P: `W_up[P], W_gate[P], W_down[:, P]`. Implementováno.
   - Permutace attention hlav (celé bloky Q/K/V/O). TODO.
   - Rotace residual streamu Q (computational invariance, SliceGPT/QuaRot):
     místo náhodné Hadamardovy matice se hledá Q, které minimalizuje entropii
     MPO řezů (optimalizace na Stiefelově varietě). TODO, nejambicióznější krok.
2. **Hledání pořadí**: spektrální bisekce → přiřazení a fit (Hungarian) →
   žíhání χ. Implementováno. Další možnosti: simulované žíhání a nakonec
   QUBO formulace pro kvantové žíhání / QAOA.
3. **MPO rozklad** s adaptivní χ podle spektra (rozpočet chyby po vrstvách
   řízený citlivostí, např. Fisherovou diagonálou z kalibračních dat).
4. **Řídká korekce** (jako Saten) pro odlehlé hodnoty.
5. **Kvantizace jader MPO** na 2–4 bity (QTIP/AQLM-styl uvnitř jader).
6. **Krátké dotrénování** (distillace z původního modelu, jen pár set milionů tokenů).

## Výsledky zatím

Experiment 1 (syntetická data, `experiments/synthetic_gauge.out.txt`):
správné pořadí je rozdíl mezi chybou 2 % a 99 % při 0,6 % parametrů. Naše
heuristika zatím snižuje chybu z 0,99 na 0,82 (χ=6) a z 0,85 na 0,49 (χ=32).
Gauge je ověřený jako přesný (rozdíl výstupů FFN ~1e-17).

## Plán experimentů

| # | Experiment | Otázka | Kritérium úspěchu |
|---|---|---|---|
| 1 | Syntetické MPO (hotovo) | Dá se skryté pořadí najít? | Částečně: ano |
| 2 | GPT-2 small, FFN vrstvy | Mají reálné váhy po gauge fixingu nižší entropii řezů? | ≥ 20 % nižší entropie / chyba při stejné χ oproti náhodnému pořadí |
| 3 | GPT-2 / Qwen-0.5B celý model | Perplexita vs. velikost: Q-Gauge vs. MPO bez gauge vs. low-rank vs. 4bit | Lepší Pareto křivka než MPO bez gauge |
| 4 | Llama-3.x 8B | Škálování | Model < 2 GB s ≤ 3 % ztrátou na benchmarcích |
| 5 | QUBO pro řazení | Pomůže kvantové žíhání / QAOA? | Srovnání se simulovaným žíháním při stejném čase |

**Riziko:** u reálných vah žádná skrytá nízko-entanglementová struktura být
nemusí, a gauge fixing pak pomůže jen málo. Experiment 2 je proto rozhodující
a levný. Měl by být první.

## Další nápady z brainstormingu (zatím nerozpracované)

- **MPO pro KV cache**: komprese kontextu přes vrstvy (navazuje na xKV, ten ale
  používá SVD).
- **Entanglement jako diagnostika**: profil entropie po vrstvách jako ukazatel,
  které vrstvy jsou redundantní (pro pruning).
- **Kvantový generátor parametrů**: malý kvantový obvod generuje jádra MPO
  (navazuje na arXiv 2410.09846). Spíš vědecky zajímavé než praktické.
