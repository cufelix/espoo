# Výsledky experimentů (listopad 2026)

Model: **RoBERTa-base** (125M parametrů, 12 vrstev) ve verzi ze spaCy
`en_core_web_trf` 3.8.0, stažený z GitHub releases a spuštěný ve vlastní
numpy implementaci (`qgauge/roberta.py`). Tokenizer jsem ověřil na známých
ID tokenů. Kalibrační text: Austen, *Emma*. Testovací text: Melville, *Moby Dick*
(16 sekvencí × 128 tokenů). **Nic se nedotrénovávalo** (bez „healingu“).

## Exp. 1: syntetická data (`experiments/synthetic_gauge.py`)
Skrytá struktura MPO se zamíchaným pořadím neuronů. Q-Gauge vrátí velkou
část zisku (chyba 0,99 → 0,66 při χ=6, 0,85 → 0,25 při χ=32). Ideální pořadí
dává 0,02. Na pořadí tedy může záležet obrovsky, **pokud v datech skrytá
struktura je**.

## Exp. 2: skutečné váhy FFN (`roberta_weights.py`, vrstva 0)
| χ | % parametrů | pořadí po tréninku | náhodné | Q-Gauge | SVD |
|---|---|---|---|---|---|
| 32 | 2 % | 0,972 | 0,972 | 0,944 | **0,917** |
| 128 | 16 % | 0,859 | 0,859 | 0,814 | **0,753** |

- Pořadí po tréninku je stejně nestrukturované jako náhodné (entropie řezů se
  shoduje na dvě desetinná místa).
- Q-Gauge sníží chybu o 3–5 %, ale **obyčejná SVD je lepší než jakákoli MPO**.
  Ověřeno i pro 3, 4 a 5 sitů.

## Exp. 3: matice embeddingů (`roberta_embedding.py`, 31 % modelu)
| χ | % parametrů | pořadí BPE | náhodné | Q-Gauge | SVD |
|---|---|---|---|---|---|
| 32 | 0,26 % | 0,984 | 0,987 | **0,960** | 0,983 |
| 64 | 0,74 % | 0,977 | 0,983 | **0,943** | 0,957 |
| 256 | 7 % | 0,930 | 0,937 | 0,890 | **0,857** |

Jediné místo, kde Q-Gauge + MPO porazí SVD: velmi malé rozpočty. Chyby jsou
tam ale tak velké (~0,95), že prakticky nepoužitelné.

## Exp. 4: funkční test celého modelu (`roberta_functional.py`)
Všech 24 matic FFN zmenšených na daný rozpočet. Metrika: kosinová podobnost
výstupních reprezentací s původním modelem (1 = stejné, ~0,1 = rozbité).

| metoda | 50 % parametrů FFN | 75 % parametrů FFN |
|---|---|---|
| SVD | 0,17 | 0,22 |
| **ASVD (SVD s ohledem na aktivace)** | **0,86** | **0,92** |
| MPO (styl CompactifAI, bez dotrénování) | 0,19 | 0,38 |
| MPO + diagonální škálování | 0,32 | 0,56 |
| MPO + Q-Gauge | 0,16 | 0,12 |
| MPO + škálování + Q-Gauge | 0,17 | 0,48 |

Kontrola citlivosti: 1 % šumu ve všech vahách FFN → cos 0,9997, 10 % → 0,97.
Model tedy není přehnaně citlivý a čísla výše platí.

## Exp. 5: hybrid ASVD × MPO (`roberta_hybrid.py`)
Faktor indexovaný neurony z ASVD zmáčknutý ještě pomocí MPO, aby se uvolnilo
místo pro vyšší rank. Při 50 %: čistá ASVD 0,86 → hybrid 0,17–0,22.
**Hybrid jednoznačně škodí.**

## Závěry (poctivě)
1. **Hypotéza Q-Gauge se na skutečných vahách nepotvrdila.** Natrénované váhy
   nemají skrytou nízko-entanglementovou strukturu přes indexy neuronů,
   kterou by stačilo „najít“ přeskládáním.
2. **Menší chyba vah ≠ menší chyba modelu.** Q-Gauge snižuje chybu vah, ale
   funkčně spíš škodí. Podobné neurony skončí ve stejném bloku MPO a jejich
   chyby se korelují.
3. **Rozhoduje znalost aktivací (dat), ne kvantová struktura.** ASVD
   s whiteningem drtí všechny varianty MPO.
4. Výsledky CompactifAI (−80 až −93 %) proto nejspíš stojí hlavně na
   **dotrénování po kompresi** a na kvantizaci, ne na tom, že by váhy byly
   přirozeně „tensor-network-friendly“. Jejich čísla jsem nereplikoval,
   protože tady nemám GPU na dotrénování.

## Kam dál (nejslibnější směry podle dat)
- **MPO od začátku tréninku / destilace do MPO.** Struktura se tak vynutí
  tréninkem, místo aby se hledala. Analogie k BitNet, kde se ternární váhy
  trénují od nuly.
- **ASVD + kvantizace + krátký healing.** Zatím nejsilnější klasická cesta.
- Ověřit totéž na dekodérových modelech (GPT-2, Qwen, Llama), až bude přístup
  k Hugging Face. RoBERTa je enkodér a může se chovat jinak.
