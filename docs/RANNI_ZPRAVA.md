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
| H12 | Velký model navrhuje sám sobě ze svých „horních bitů“ | ✅ **2,34× bez druhého modelu a bez paměti navíc** |
| dřív | Kvantové přeskládání vah (Q-Gauge), JIT zmenšení, hybridy | ❌ |

## Nejlepší ověřený recept: velký model lokálně levněji
1. **Ulož model v 8 bitech** místo 16: 2× méně paměti. Výstup se shoduje
   s originálem na 99,6 % slov.
2. **Ať model navrhuje sám sobě z horních 3–4 bitů svých vah** a pak se
   ověří celými 8 bity. Výstup je přesně stejný jako u 8bitového modelu a
   generování je 1,5× rychlejší. Nepotřebuješ druhý model ani paměť navíc.
3. **Dohromady: ~3× rychleji a 2× méně paměti než standard, prakticky stejný
   výstup.**
4. Pokud chceš výstup **bit po bitu stejný** jako u plného modelu:
   ~2,3× rychleji za +25 % paměti (4bitová kopie jako navrhovatel).

**Proč ne 10×:** bez přetrénování se dá bezztrátově ušetřit jen ~30 % paměti
a ověřování dá 2–3×. 10× vyžaduje přetrénování (např. 2bitové modely
trénované od začátku jako BitNet) nebo modely, které jsou přirozeně „řídké“.

**Pozor, není to nové:** podobné metody už existují (BitNest a BS-Spec
z října 2026, QSpec, SPEQ, FR-Spec, DistillSpec). Já je nezávisle ověřil
a spojil.

## Co je opravdu nový poznatek
**Paměť (databáze) pomáhá AI jen se zapamatovanými konkrétními fakty, ne
s úsudkem.** Když malý model nahlíží do databáze toho, „co by si myslel velký
model“, nepomůže mu to vůbec. Nepomůže to ani při dokonalém vyhledávání
pomocí reprezentace velkého modelu. Mechanismus jsem ověřil, je přesný.
Podobné situace v databázi prostě nejsou dost podobné a zprůměrovaný úsudek
je rozmazaný. Úsudek se musí spočítat, nedá se vyhledat. Tím se vyvrací
jeden z intuitivních směrů „zmenšování AI pomocí paměti“. Tohle jsem
v literatuře nenašel.

## Omezení (ať se nespálíš)
- Modely jsou malé (miliony parametrů, ne miliardy) a běžely na CPU.
- Zrychlení je spočítané z míry přijetí a z počtu přečtených bajtů, ne
  změřené stopkami na grafické kartě.
- Hugging Face byl v této session zablokovaný, takže jsem nemohl vzít
  skutečný velký open model.

## Co dál (konkrétně)
1. Spusť **novou session** s povoleným `huggingface.co`. Pak změřím recept na
   Qwen2.5-0.5B/1.5B a na skutečném textu.
2. Na **počítači s grafickou kartou** změřit skutečné zrychlení stopkami.
3. Pro 10×: zkusit krátké **dotrénování** na 2–3 bity a k tomu recept výše.
