# Přehled výzkumu: kvantové / kvantově inspirované AI a komprese LLM

Stav k listopadu 2026. Čísla jsou převzatá z abstraktů a tiskových zpráv
autorů, pokud není uvedeno jinak. Nezávislé replikace jsou vzácné.

## 1. Kvantově *inspirované* tensor networks (funguje už dnes, běží na GPU)

- **CompactifAI** (Multiverse Computing, arXiv 2401.14109, ESANN 2025).
  Váhy se rozloží do Matrix Product Operators (MPO) a krátce se dotrénují.
  Na LLaMA-2 7B autoři uvádějí −93 % paměti a −70 % parametrů s ~2–3 % ztrátou
  přesnosti, pokud se to zkombinuje s kvantizací. V roce 2025 firma vydala
  „Slim“ verze Llama 3.1 8B / 3.3 70B (−80 %) a získala 215 mil. USD.
  Čísla pocházejí od firmy samotné.
  https://arxiv.org/abs/2401.14109
- **Saten** (arXiv 2505.14871, EMNLP 2025). Tensor-train rozklad plus řídká
  korekce, bez dotrénování. Konstatuje, že čistě low-rank tensorová
  komprese předtrénovaných LLM zatím moc nefunguje.
  https://arxiv.org/abs/2505.14871
- **TensorGPT** (arXiv 2307.00526). TT/MPS komprese embedding vrstvy.
  https://arxiv.org/abs/2307.00526
- MPO pro lehký fine-tuning (Gao a kol., „Enabling Lightweight Fine-tuning …
  based on Matrix Product Operators“) a TT na FPGA (arXiv 2501.19135).

**Mezera, kterou Q-Gauge cílí:** žádná z těchto prací (podle toho, co jsem
našel) systematicky neoptimalizuje **pořadí / bázi** neuronů před rozkladem,
i když jde o přesnou symetrii modelu. Ve fyzice je přitom optimalizace pořadí
orbitalů pro DMRG standardní a zásadní krok.

## 2. Extrémní kvantizace (klasická, dnešní state of the art)

- **QuIP#** (E8 mřížka + Hadamardova „incoherence processing“), **AQLM**
  (aditivní vícekódové kvantování), **QTIP** (trellis-coded quantization,
  NeurIPS 2024). Kolem 2 bitů na váhu s rozumnou kvalitou.
  https://arxiv.org/abs/2406.11235 · https://arxiv.org/abs/2401.06118
- Nezávislé srovnání na 11B modelu (arXiv 2603.04162, 2026): efektivně
  ~3,3–3,6 bpw po započtení režie kodebooků.
- **BitNet b1.58** (Microsoft): ternární váhy {−1, 0, 1}, trénováno od nuly.
  Model 2B4T (arXiv 2504.12285) je open-weight, jeho jádro bez embeddingů má
  ~0,4 GB. Nejde ale o kompresi existujícího modelu, model se trénuje znovu.
  https://arxiv.org/abs/2504.12285

**Pozor na napětí s Q-Gauge:** QuIP#/QTIP *záměrně* rotují váhy náhodnou
Hadamardovou maticí, aby rozprostřely informaci. Pro kvantizaci je to dobře,
ale entanglement se tím maximalizuje, což je pro MPO špatně. Q-Gauge dělá
opak (minimalizuje entanglement), takže pořadí kroků musí být promyšlené: nejdřív
MPO, pak incoherence processing *uvnitř* jader MPO.

## 3. Skutečné kvantové obvody v jazykových modelech (rané, malé)

- **HyQuT** (arXiv 2511.10653). Hybridní kvantově-klasický LLM s 150M
  parametry, ve kterém 10 qubitů / 80 hradel nahradí ~10 % parametrů. Je to
  první demonstrace a autoři ji tak sami popisují.
- **Quantum NLI** (arXiv 2510.15972). Metrika „information gain per parameter“
  vychází kvantovým modelům řádově lépe, ale nejde o absolutní přesnost ani
  výpočetní výhodu.
- **Quantum parameter adaptation** (arXiv 2410.09846). Kvantový obvod
  *generuje* parametry pro fine-tuning (obdoba LoRA), samotný LLM běží klasicky.

## 4. Proč nejde „převést 70B model do kvantového počítače“

- **Načítání dat / QRAM.** Zakódovat N klasických čísel do kvantového stavu
  stojí obecně O(N) operací a QRAM je spekulativní hardware.
  https://arxiv.org/abs/2305.10310
- **Dekvantizace** (Ewin Tang a kol.). Mnohé kvantové ML algoritmy
  s exponenciálním zrychlením mají klasické protějšky se stejným typem
  přístupu k datům. https://arxiv.org/abs/1811.00414
- **Měření.** Z n qubitů se jedním měřením dostane jen n bitů. Odhadnout
  logity přes slovník o 100k tokenech chce obrovské množství opakování.
- **Šum a velikost.** Dnešní zařízení mají stovky až nízké tisíce fyzických
  qubitů s omezenou koherencí, takže hluboké obvody o miliardách parametrů
  nejsou na pořadu dne.

**Kde má kvantový hardware realistickou roli:** kombinatorická optimalizace
v malém měřítku. Hledání pořadí neuronů (Q-Gauge, krok 2) je problém typu
QAP/QUBO, takže kvantové žíhání nebo QAOA jsou legitimní experiment. Kvantová
výhoda pro takové problémy ale **není prokázaná** a silné klasické heuristiky
(simulované žíhání, tabu search) jsou výchozí srovnání.
