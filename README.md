# Aα-NSGA-II — Co-Evolving the Confidence Level in Fuzzy Multi-Objective Optimization

Reference implementation, experimental pipeline, and manuscript sources for the paper:

> **Co-Evolving the Confidence Level in Fuzzy Multi-Objective Optimization: A Computable Criterion and Its Empirical Test**
> Yves Ndikuriyo, Yinggui Zhang
> *IEEE Transactions on Fuzzy Systems* (under review, 2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE-figures.md)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Cite this repository](https://img.shields.io/badge/cite-CITATION.cff-brightgreen.svg)](CITATION.cff)

---

## Overview

Aα-NSGA-II is a fuzzy multi-objective evolutionary algorithm that **co-evolves the confidence level α** of fuzzy chance constraints as an additional decision variable, in parallel with a **success-rate-adaptive polynomial mutation schedule** (discrete 1/5 rule). The paper establishes seven theoretical properties and culminates in the **α-sensitivity condition**, which reduces to the closed-form test

$$
f_i^M = \tfrac{1}{2}\bigl(f_i^L + f_i^U\bigr) \quad \text{along the Pareto front}
$$

and predicts when confidence-level co-evolution is worth activating.

The paper reports a **2×2 factorial ablation** on 49 fuzzy benchmark problems (10 runs each) against SPEA2, MOEA/D, NSGA-III, and cMLSGA, plus a **15-problem F2 diagnostic family** designed as a positive control for α-sensitivity.

---

## Key Findings

- **Adaptive mutation is the dominant mechanism**, raising normalized HV from 0.8949 to 0.9750.
- **Co-evolving α on top adds no significant gain** on the approximately α-invariant suite (`p = 0.851`, Cohen's `d = −0.027`).
- **Co-evolving α in isolation is worse** than fixed-α baselines.
- The two adaptive mechanisms win **nearly disjoint problem sets** (26 and 24 of 64).
- The F2 diagnostic family turned out **less α-sensitive** than F1 ($\widehat{\Sigma} = 0.0108$ vs. $0.0326$), so the positive direction of the α-sensitivity proposition remains untested.

---

## Repository Structure

```
aa-nsga2/
├── code/
│   └── aa_nsga2_pipeline.py     # Complete pipeline (v8.8)
├── manuscript/
│   ├── Manuscript-Aalpha-NSGAII.tex
│   ├── Supplementary material.tex
│   ├── title page.tex
│   ├── Authors details.tex
│   └── figures/                 # pareto_*.png, rank_distribution.png, etc.
├── results/
│   ├── results.csv              # Per-(problem, algorithm) aggregate metrics
│   ├── sigma_hat.csv            # Empirical α-sensitivity proxy Σ̂
│   └── key_figures/             # Selected figures for quick inspection
├── README.md
├── CITATION.cff
├── LICENSE                      # MIT — applies to code/ only
└── LICENSE-figures.md           # CC-BY-4.0 — manuscript, figures, tables
```

> **Licensing note.** The repository uses **two licenses**. Source code under
> `code/` is MIT (see `LICENSE`). Manuscript sources, LaTeX files, figures,
> result tables, and documentation are CC-BY-4.0 (see `LICENSE-figures.md`).

---

## Requirements

- **Python** 3.9+
- **Packages**: `numpy`, `pandas`, `scipy`, `matplotlib`

Install with:

```bash
pip install numpy pandas scipy matplotlib
```

No GPU is required. The full experiment (64 problems × 13 algorithms × 10 runs)
takes several hours on a multi-core CPU.

---

## Reproducing the Experiments

### 1. Configure the output directory

Open `code/aa_nsga2_pipeline.py` and set `OUTPUT_ROOT` to a writable folder on
your system. The checked-in version points to a Windows path from the authors'
machine; replace it with a portable default, e.g.:

```python
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"
)
```

### 2. Run the full pipeline

```bash
# Serial (stable, slower) — recommended for reproducibility
python code/aa_nsga2_pipeline.py --serial

# Parallel — uses half the available cores
python code/aa_nsga2_pipeline.py

# Test mode (5 problems, 2 runs, 30×50 budget) — ~1 minute
python code/aa_nsga2_pipeline.py --test --serial
```

### 3. Useful flags

| Flag | Effect |
|---|---|
| `--serial` | Force serial execution (recommended for reproducibility) |
| `--workers N` | Parallel worker count |
| `--runs N` | Override number of independent runs (default 10) |
| `--resume` | Resume from `checkpoint.pkl` if present |
| `--pareto-only` | Generate **only** Pareto-front figures (skip experiments) |
| `--no-sigma` | Skip Σ̂ computation |
| `--no-igd-gd` | Skip IGD/GD post-processing pass |
| `--external-mode {strict,fair,both}` | External-comparison condition |

### 4. Outputs

All results, figures, and LaTeX tables are written under `OUTPUT_ROOT/`:

- `results/` — `checkpoint.pkl`, `results.csv`, `igd_gd_checkpoint.pkl`
- `figures/` — heatmaps, rank distributions, Pareto-front plots
- `latex/` — generated LaTeX tables for the main paper
- `supplementary_latex/` — per-problem tables for the supplement
- `sigma_hat/` — Σ̂ CSV and LaTeX table
- `external/` — external-comparison CSV

These regenerated outputs are ignored by `.gitignore`; only the aggregated
`results/*.csv` and a small set of `key_figures/` are versioned.

---

## Statistical Protocol

- **10 independent runs** per (problem, algorithm) with seeds `2026–2035`.
- **Friedman test** for overall differences (rank data are non-normal on all families).
- **Wilcoxon signed-rank** for pairwise comparisons against Aα-NSGA-II.
- **Bonferroni correction** at `α' = 0.0125` (four internal comparisons) and `α' = 0.0167` (six external comparisons).
- **Effect sizes**: Cohen's `d` and Cliff's `δ`.
- **Bootstrap 95% CIs** with 5,000 resamples.

---

## Reproducibility Notes

- The implementation is fully deterministic given the seed. No random elements
  outside the seeded NumPy generator are used.
- IGD/GD reference fronts are built from a **memory-safe, per-algorithm-balanced
  empirical pool** (capped at 150 points per algorithm, 100 output points),
  stored in `igd_gd_checkpoint.pkl` independently of the algorithm under test.
- `FScal5000` and `FScal10000` are **excluded from IGD/GD** (unstable reference
  fronts at `N = 60`), so IGD/GD covers 62 of the 64 problems.
- The four external baselines (SPEA2, MOEA/D, NSGA-III, cMLSGA) are
  **reimplemented within the same evaluation framework** as the internal
  configurations, so budget, operators, seeds, and reference α are identical.

---

## Citation

If you use this code or the benchmark suite, please cite:

```bibtex
@article{ndikuriyo2026alpha,
  title   = {Co-Evolving the Confidence Level in Fuzzy Multi-Objective
             Optimization: A Computable Criterion and Its Empirical Test},
  author  = {Ndikuriyo, Yves and Zhang, Yinggui},
  journal = {IEEE Transactions on Fuzzy Systems},
  year    = {2026},
  note    = {Under review}
}
```

A machine-readable `CITATION.cff` is included; GitHub renders a
**"Cite this repository"** button from it.

---

## Contact

- **Yves Ndikuriyo** (corresponding author) — `yvesndikuriyo@csu.edu.cn`
  School of Traffic and Transportation Engineering, Central South University
  ORCID: [0009-0006-9324-7265](https://orcid.org/0009-0006-9324-7265)
- **Yinggui Zhang** — `ygzhang@csu.edu.cn`
  ORCID: [0000-0002-5790-0638](https://orcid.org/0000-0002-5790-0638)

---

## Acknowledgments

This research was supported by the National Natural Science Foundation of China
(Grant No. 71971220) and the Natural Science Foundation of Hunan Province,
China (Grant Nos. 2023JJ30710 and 2022JJ31020).
5. **Fixed the citation block** to match the author order and title in your manuscript (`Ndikuriyo, Yves and Zhang, Yinggui`), and added the `note = {Under review}` field.
6. **Added the second author's email and ORCID** from `Authors details.tex`, so the Contact section is complete.
7. **Added the badges at the top** — the MIT and CC-BY-4.0 badges link directly to `LICENSE` and `LICENSE-figures.md`, which is the correct way to indicate a dual-license repo.
