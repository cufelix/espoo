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

## První výsledek (syntetická data)

Matice FFN 1024×256 se skrytou strukturou MPO (χ=6) a zamíchaným pořadím
neuronů, jako po tréninku. Hodnota je relativní chyba rekonstrukce při stejném
počtu parametrů (nižší je lepší):

| χ | parametrů | zamíchané | spektrální gauge | + žíhání χ | oracle pořadí | low-rank SVD |
|---|---|---|---|---|---|---|
| 6 | 1 584 (0,6 %) | 0,989 | 0,976 | **0,824** | 0,020 | 0,987 |
| 16 | 9 536 (3,6 %) | 0,939 | 0,894 | **0,619** | 0,020 | 0,920 |
| 32 | 26 944 (10 %) | 0,848 | 0,760 | **0,493** | 0,019 | 0,807 |

Co z toho plyne:
1. **Na pořadí neuronů záleží obrovsky.** Se správným pořadím stačí 0,6 %
   parametrů s chybou 2 %, se zamíchaným pořadím je i 10 % parametrů skoro
   k ničemu. Naivní MPO komprese tuhle možnost úplně nevyužívá.
2. Naše heuristika zatím obnoví jen část zisku (0,99 → 0,82 při χ=6).
   Hledání pořadí je těžký kombinatorický problém a je to hlavní otevřená
   otázka projektu.
3. **Zatím neověřeno na skutečných modelech.** Syntetická data mají strukturu
   uměle vloženou. Jestli ji mají i natrénované váhy (GPT-2, Llama, Qwen), je
   první věc, kterou je potřeba změřit (viz plán v `docs/proposal.md`).

Spuštění:

```bash
pip install -r requirements.txt
python experiments/synthetic_gauge.py
```
