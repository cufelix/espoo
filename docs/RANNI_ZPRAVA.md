# Ranní zpráva: co jsem přes noc zjistil

## Na rovinu hned na začátku
**Revoluci jsem nenašel.** Hledal jsem ji poctivě: 428 vědeckých prací a 12
hypotéz otestovaných na skutečně natrénovaných modelech. Většina toho, co
funguje, už někdo publikoval. To, co by bylo nové, pokusy nepotvrdily.
Výsledkem je **ověřený recept**, jak pustit velký model levněji se stejným
výstupem, a pár **nových poznatků**, proč některé slibné cesty nefungují.

## Co jsem udělal
1. **Výzkum:** 428 prací ve 39 tématech (katalog v `docs/lit/`) a pohled na
   obor shora (`docs/landscape.md`). Škálování AI brzdí tři zdi: docházejí
   data, výpočty jsou drahé a hlavně je pomalé čtení vah z paměti.
2. **Vlastní modely:** natrénoval jsem v JAX na CPU dva jazykové modely na
   Wikipedii (WikiText-2): „učitele“ (5,7 mil. parametrů) a „studenta“
   (1,7 mil.).
3. **12 hypotéz**, každá s měřením (`docs/results-memory.md`, `docs/results.md`).

## Výsledky hypotéz
| # | hypotéza | výsledek |
|---|---|---|
| H1 | Destilace do paměti: malý model se dívá do databáze úsudků velkého | ❌ nulový přínos |
| H2 | Malý model navrhuje, velký ověřuje (spekulativní dekódování) | ✅ 1,42× rychleji, stejný výstup |
| H3 | Velký model navrhuje sám sobě pomocí prvních vrstev | ❌ u malých modelů ne |
| H4 | Je chyba H1 ve špatném vyhledávání? | ❌ ne, problém je hlubší (viz níže) |
| H5 | Větší databáze = lepší malý model | ✅ dožene ~40 % rozdílu proti učiteli |
| H8 | Návrh počítá jen 1000 nejčastějších slov | ✅ 1,42× → 1,84× |
| H9 | Návrh zkopírovaný z databáze (zdarma) | ➖ jen 1,16× |
| H10 | Destilovaný malý model navrhuje lépe | ✅ 1,92× |
| H11 | Úsudek do vah (destilace) + fakta do paměti | ✅ 3,4× menší model dožene 56 % rozdílu kvality (68,7 → 63,8, učitel 59,9) |
| H12 | Velký model navrhuje sám sobě ze svých „horních bitů“ | ✅ **2,34× bez druhého modelu a bez paměti navíc** |
| dřív | Kvantové přeskládání vah (Q-Gauge), JIT zmenšení, hybridy | ❌ |

## Nejlepší ověřený recept, změřený na SKUTEČNÉM modelu (GPT-2, 124M)
Ráno jsem recept ještě ověřil na skutečném předtrénovaném GPT-2 (váhy
z ONNX Model Zoo na GitHubu, text Moby Dick, `experiments/gpt2_recipe.py`).
Výsledky jsou střízlivější než na mých malých modelech:

| varianta | kvalita (perplexita, originál 81,6) | méně bajtů / rychleji než 16 bitů |
|---|---|---|
| uložení v 8 bitech (skupiny po 32) | **81,6**, 98,3 % slov stejných | 1,8× |
| 8 bitů + model si navrhuje z horních 6 bitů | výstup **přesně** jako 8bitový model | **~2,0×** |
| 16 bitů + navrhovatel z 6bitové kopie | výstup **přesně** jako originál | 1,7× |
| uložení ve 4 bitech | 84,6 (o 3,7 % horší) | 3,2× |

- **Bez ztráty kvality: ~2× rychleji a skoro 2× méně paměti.**
- **S malou ztrátou (−3,7 %): ~3,2×.**
- Vnořené návrhy pod 6 bitů na skutečném modelu nefungují. Oříznutí bitů
  poškodí extrémní hodnoty vah, na kterých GPT-2 hodně závisí. Na
  miniaturních modelech to vypadalo líp (2,3×), takže skutečný model byl
  důležitá kontrola.

**Proč ne 10×:** bez přetrénování se dá bezztrátově ušetřit jen ~30 % paměti
a ověřování dá 1,5–2×. Změřeno na GPT-2 to dohromady dává ~2× bez ztráty.
10× vyžaduje přetrénování (např. 2bitové modely trénované od začátku jako
BitNet) nebo modely, které jsou přirozeně „řídké“.

**Pozor, není to nové:** podobné metody už existují (BitNest a BS-Spec
z října 2026, QSpec, SPEQ, FR-Spec, DistillSpec). Já je nezávisle ověřil,
změřil na skutečném modelu a spojil.

## Co je opravdu nový poznatek
**Paměť (databáze) pomáhá AI jen se zapamatovanými konkrétními fakty, ne
s úsudkem.** Když malý model nahlíží do databáze toho, „co by si myslel velký
model“, nepomůže mu to vůbec. Nepomůže to ani při dokonalém vyhledávání
pomocí reprezentace velkého modelu. Mechanismus jsem ověřil, je přesný.
Podobné situace v databázi prostě nejsou dost podobné a zprůměrovaný úsudek
je rozmazaný. Úsudek se musí spočítat, nedá se vyhledat. Nejlépe proto funguje rozdělení práce: **úsudek patří do vah (destilace), fakta do paměti**. Malý model tak dožene 56 % rozdílu proti 3,4× většímu. Tím se vyvrací
jeden z intuitivních směrů „zmenšování AI pomocí paměti“. Tohle jsem
v literatuře nenašel.

## Omezení (ať se nespálíš)
- Většina pokusů běžela na malých modelech na CPU. Recept je ověřený
  i na skutečném GPT-2 (124M), ale ne na modelech s miliardami parametrů.
- Zrychlení je spočítané z míry přijetí a z počtu přečtených bajtů, ne
  změřené stopkami na grafické kartě.
- Hugging Face byl v této session zablokovaný. GPT-2 jsem sehnal přes
  ONNX Model Zoo, větší modely (Qwen, Llama) tudy nejdou.

## Co dál (konkrétně)
1. Spusť **novou session** s povoleným `huggingface.co`. Pak změřím recept na
   Qwen2.5-0.5B/1.5B a na skutečném textu.
2. Na **počítači s grafickou kartou** změřit skutečné zrychlení stopkami.
3. Pro 10×: zkusit krátké **dotrénování** na 2–3 bity a k tomu recept výše.
