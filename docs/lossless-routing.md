# Návrh: dynamické routování bez ztráty výkonu

Požadavek: model, který se dynamicky „routuje“ (pro lehké tokeny/úlohy použije
méně výpočtu) a přitom **neztratí žádnou kvalitu**.

## Co je opravdu bezztrátové (matematicky zaručené)
1. **Spekulativní dekódování** (Leviathan a kol. 2023, Chen a kol. 2023):
   levný „draft“ navrhne několik tokenů a plný model je ověří jedním paralelním
   průchodem. Rozdělení výstupů je **identické** s plným modelem, typicky 2–3×
   rychlejší generování.
2. **DFloat11** (arXiv 2504.11651, NeurIPS 2025): entropické kódování vah BF16.
   Model zabere ~70 % paměti a výstupy jsou **bit po bitu stejné**.

## Návrh „Routed Self-Speculation“ (RSS)
Spojení routování a spekulace:
- **Draft = stejný model, ale dynamicky ořezaný:** router vybere ~20–30 %
  neuronů FFN na token (contextual sparsity, Deja Vu), případně se přeskočí
  pozdní vrstvy (LayerSkip). Žádný druhý model není potřeba.
- **Verify = plný model**, se zrychlením díky ověřování k tokenů naráz.
- **Bezztrátové:** každý token, který plný model neschválí, se nahradí jeho
  vlastní volbou.
- **Úsporné na paměť:** váhy uložené jednou v DFloat11, draft je jen podmnožina
  téhož.
- **Dynamické:** míra ořezání draftu se řídí podle úspěšnosti přijetí. U
  snadného textu draft agresivně ořezává, u těžkého méně.

Zrychlení závisí na tom, jak často plný model draft přijme. To je potřeba
změřit na dekodérovém modelu (GPT-2 / Qwen / Llama). RoBERTa je enkodér a
negeneruje text, takže na ní to změřit nejde.

## Co zaručit nejde
Menší **paměť i výpočet zároveň bez jakékoli ztráty**, nad rámec bezztrátového
kódování (~30 %). Každé ztrátové zmenšení (kvantizace, low-rank, MPO, MoE)
něco stojí. Dá se jen minimalizovat a změřit.
