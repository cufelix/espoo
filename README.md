# espoo — Q-Gauge

**Kvantově inspirovaná komprese velkých jazykových modelů pomocí „gauge fixingu“.**

> Hlavní myšlenka: dnešní komprese LLM pomocí tensor networks (MPO / tensor-train,
> např. CompactifAI) bere váhy modelu tak, jak vyšly z tréninku. Jenže pořadí
> neuronů v síti je **libovolné** a jeho změna vůbec nemění chování modelu.
> Ve fyzice (DMRG v kvantové chemii) se ví, že správné pořadí „orbitalů“ na
> řetězci MPS sníží potřebnou bond dimension o řády. Q-Gauge využívá **přesné
> symetrie transformeru** (permutace neuronů a hlav, rotace residual streamu)
> k přepsání vah do báze s minimálním „provázáním“ (entanglementem), a teprve
> pak je komprimuje do MPO a kvantizuje.

Symetrie nic nestojí, protože model počítá přesně totéž. Za kompresi se pak
platí jen tím, kolik informace se opravdu zahodí.

## Proč ne „opravdový“ kvantový počítač?

Krátce: běžet 7B+ model na kvantovém hardwaru dnes ani v dohledné době nejde.
Načtení miliard vah do qubitů (QRAM) je nevyřešený a možná principiálně drahý
problém. Výstup z kvantového stavu vyžaduje mnoho opakovaných měření a
„dekvantizační“ výsledky (Ewin Tang a další) ukazují, že řadu slibovaných
kvantových zrychlení pro ML lze dosáhnout i klasicky. Kde kvantová fyzika
reálně pomáhá už teď, je **matematika tensor networks**, která běží na
obyčejných GPU. Kvantový hardware může později pomoct s jedním kombinatorickým
podproblémem: hledáním optimálního pořadí neuronů. Podrobnosti najdeš v
[docs/research-survey.md](docs/research-survey.md).

## Co je v repu

| Cesta | Obsah |
|---|---|
| `docs/research-survey.md` | Přehled výzkumu: co funguje, co je hype, odkazy na papery |
| `docs/proposal.md` | Návrh metody Q-Gauge, hypotézy, plán experimentů |
| `qgauge/tt.py` | TT-SVD / MPO rozklad matic, entanglement entropie (numpy) |
| `qgauge/gauge.py` | Gauge fixing: spektrální řazení neuronů a zjemnění s žíháním bond dimension |
| `experiments/synthetic_gauge.py` | Experiment 1 na syntetických vahách |

## Výsledky (stručně)

Podrobně v [docs/results.md](docs/results.md).

- **Syntetická data:** na pořadí neuronů může záležet obrovsky a Q-Gauge
  velkou část zisku najde.
- **Skutečný model (RoBERTa-base):** hypotéza se **nepotvrdila**. Natrénované
  váhy skrytou strukturu nemají. MPO (i s Q-Gauge) prohrává se SVD
  a hlavně s ASVD, která bere ohled na aktivace.
  Při zmenšení FFN na 50 % bez dotrénování: ASVD cos 0,86, nejlepší MPO 0,32,
  MPO + Q-Gauge 0,17.
- **Poučení:** u komprese rozhodují data (aktivace) a dotrénování, ne
  „kvantová“ struktura vah. Další směr: trénovat přímo v MPO formě.

Spuštění:

```bash
pip install -r requirements.txt msgpack regex
python experiments/synthetic_gauge.py
```

Experimenty s RoBERTa potřebují váhy: `tools/torchzip_to_npz.py` je vytáhne
z wheelu spaCy `en_core_web_trf` (GitHub releases).
