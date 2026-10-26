# SA-ALNS for DSC-CTRP

**Stability-Aware Adaptive Large Neighborhood Search**
for the **Dynamic Stability-Constrained Container-Tanker Routing Problem**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PuLP](https://img.shields.io/badge/PuLP-2.6.0-red.svg)](https://coin-or.github.io/pulp/)

---

## 📖 Overview

Liquid bulk transportation via multi-compartment tankers presents a critical operational challenge: balancing economic efficiency with physical safety throughout dynamic delivery routes. Existing vehicle routing models either ignore loading constraints entirely or treat them as static post-optimization checks, leading to solutions that are economically optimal but operationally infeasible or unsafe.

This repository provides the complete implementation of the **Stability-Aware Adaptive Large Neighborhood Search (SA-ALNS)** algorithm for solving the **Dynamic Stability-Constrained Container-Tanker Routing Problem (DSC-CTRP)**.

### Key Contributions

| Contribution | Description |
|--------------|-------------|
| **Unified Formulation** | DSC-CTRP integrating routing decisions with dynamic stability constraints at every delivery prefix |
| **Three-Tier Constraint Hierarchy** | Geometric/capacity, static LDD, and dynamic anti-sloshing constraints in one MILP |
| **SA-ALNS Algorithm** | Master–slave metaheuristic with embedded stability verification |
| **Sequence-Aware Cache** | Feasibility cache keyed on the ordered visit sequence (not the customer set) |
| **Benchmark Suite** | 100 synthetic instances (3–5 compartments); 30 used for the SOTA comparison and ablation |
| **Open Null Result** | Adaptive operator selection and stability-aware repair show no measurable effect on this benchmark |

---

## 📊 Experimental Results

### State-of-the-Art Comparison (30 Instances, 20–25 Customers)

| Algorithm | Avg Penalised Cost ($) | Fleet | LDD Violation (%) | CG Compliance (%) | Wall-clock (s) |
|-----------|------------------------|-------|-------------------|-------------------|----------------|
| **SA-ALNS** | 1430.32 | 4.30 | 35.17 | 100.00 | 0.43 |
| SA-ALNS Cost-Only | 1435.02 | 4.33 | 36.00 | 100.00 | 0.38 |
| Standard ALNS | 1416.51 | 4.30 | 30.83 | 100.00 | 0.51 |
| FDAHS | 1550.84 | 4.20 | 22.61 | 100.00 | 1.54 |
| GCOF | 1777.46 | 4.37 | 37.89 | 100.00 | 0.22 |
| HSA | 1620.27 | 4.37 | 32.94 | 100.00 | 0.01 |
| ACO-2opt | 1516.81 | 4.17 | 22.33 | 100.00 | 1.06 |

> **Key Results (from the manuscript).** SA-ALNS attains **5.70–19.53% lower penalised cost** than the four traditional metaheuristics (FDAHS, GCOF, HSA, ACO-2opt), with large effect sizes against GCOF (*p* = 1.9×10⁻⁹, Cohen's *d* = −0.844) and HSA (*p* = 9.3×10⁻⁹, *d* = −0.690). SA-ALNS is **statistically comparable to Standard ALNS** on cost (*p* = 0.152) but returns routes with lower residual penalty. Relative to a single-product-per-route baseline, the multi-compartment configuration reduces cost by **27.44%** and fleet size by **23.68%**. Integrated routing and loading outperforms sequential decomposition by **37.7%**. All solutions achieve **100% centre-of-gravity compliance**; residual LDD violations of **35.17%** are penalised and bounded along the delivery sequence.

> **Implementation note.** The LP fallback in SACA is retained for completeness but is **never invoked** on this benchmark: the greedy split-allocation procedure succeeds on every feasible route, so MILP diagnostics are uniformly zero and wall-clock time is dominated by routing search. We disclose this explicitly rather than presenting the LP as a functional component.

---

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- IBM ILOG CPLEX 22.1.0 or higher (optional — used only by the SACA-MILP fallback, which is not exercised on the reported benchmark)
- MiKTeX or TeX Live (for manuscript compilation)

### Setup

```bash
# Clone the repository
git clone https://github.com/YvesNDIKURIYO-2022/SA-ALNS-4-DSC-CTRP.git
cd SA-ALNS-4-DSC-CTRP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
click>=8.0.0
tqdm>=4.62.0
pyyaml>=5.4.0
joblib>=1.1.0
pulp>=2.6.0
```

---

## 🚀 Quick Start

```python
from src.alns import SAALNS
from src.benchmarks import load_instance

# Load a benchmark instance
instance = load_instance("P20_1")

# Create solver
solver = SAALNS(instance)

# Solve
solution = solver.solve()

# Results
print(f"Total Cost: {solution.cost:.2f}")
print(f"CG Compliance: {solution.cg_compliance:.2f}%")
print(f"LDD Violations: {solution.ldd_violations}")
print(f"Stability Margin: {solution.stability_margin:.2f}%")
print(f"CPU Time: {solution.cpu_time:.3f}s")
```

If you import from an older script, a backward-compatibility alias is provided: `from src.alns import IALNS` still resolves to `SAALNS`.

---

## 📁 Repository Structure

```
SA-ALNS-4-DSC-CTRP/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── alns/                    # ALNS Master Module
│   │   ├── master.py            # Main ALNS loop
│   │   ├── destroy_operators.py # Random, Worst-Cost, Compatibility, Imbalance
│   │   └── repair_operators.py  # Greedy, Regret, Stability-Aware
│   ├── stability/               # SACA Slave Module
│   │   ├── saca_milp.py         # Exact MILP solver (|R_k| ≤ 9)
│   │   ├── saca_heuristic.py    # Heuristic solver (|R_k| > 9)
│   │   └── ldd_verification.py  # LDD constraint checker
│   ├── cache/                   # Feasibility Cache
│   │   └── feasibility_cache.py # LRU cache with sequence-aware keys
│   └── utils/                   # Utilities
│       ├── instance_loader.py   # JSON instance loader
│       └── metrics.py           # Performance metrics
│
├── benchmarks/                  # Benchmark instances
│   ├── paixao/                  # 15 instances (20–30 customers, 3 commodities)
│   │   ├── P20_1--P20_5         # 20 customers
│   │   ├── P25_1--P25_5         # 25 customers
│   │   └── P30_1--P30_5         # 30 customers
│   └── mirzaei/                 # 85 instances (20–100 customers, 2–6 commodities)
│       ├── vrpnc1--vrpnc15      # Mirzaei-layer instances
│       └── extra33--extra87     # Extended instances
│
├── manuscript/                  # LaTeX manuscript
│   ├── Manuscript-SA-ALNS.tex
│   ├── references.bib
│   └── figures/                 # All figures (PNG, PDF)
│
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Documentation
```

---

## 📚 Benchmark Instances

The study evaluates **100 synthetic benchmark instances** spanning 20–100 customers, 3–5 compartments, and 2–6 products. The instances are newly generated for this study, mirroring the structural parameters of established MCVRP literature. 

| Layer | Instances | Customers | Compartments | Products | Purpose |
|-------|-----------|-----------|--------------|----------|---------|
| Paixão-based (Paixão et al., 2026) | 15 | 20–30 | 3 | 3 | LDD & stability validation |
| Mirzaei-based (Mirzaei & Wøhlk, 2019) | 85 | 20–100 | 3–5 | 2–6 | Scalability testing |
| **Total** | **100** | **20–100** | **3–5** | **2–6** | **Full evaluation** |

Each customer demands exactly one product, with demand magnitudes calibrated to 60–90% of the average compartment volume. Compartment capacities range from 120 to 200 units. All instances are stored in standardized JSON format and are publicly available in the `benchmarks/` directory.

---

## 🧠 Algorithm Architecture

The SA-ALNS framework follows a master–slave architectural pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: Generate Initial                      │
│                   Stability-Aware Solution                      │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Master Module: ALNS Routing Search                 │
│  ┌────────────────────────┐   ┌──────────────────────────────┐ │
│  │   Destroy Operators    │   │    Repair Operators          │ │
│  │  • Random              │ → │  • Greedy                    │ │
│  │  • Worst-Cost          │   │  • Regret-k                  │ │
│  │  • Compatibility       │   │  • Stability-Aware Insertion │ │
│  │  • Imbalance           │   │  • Balance                   │ │
│  │  • Shaw                │   │                              │ │
│  └────────────────────────┘   └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Sequence-Aware Feasibility Cache Check                │
│        (key = ordered customer tuple, not the set)              │
│              YES ────────────────────────┐                      │
│               NO                         │                      │
│               ▼                          ▼                      │
│  ┌───────────────────────────┐ ┌────────────────────────────┐   │
│  │  Slave Module: SACA       │ │  Update Best Solution      │   │
│  │  Two-Stage Verification   │ │                            │   │
│  │  Stage 1: Capacity        │ └────────────────────────────┘   │
│  │  & Volume Screening       │                                  │
│  │  Stage 2: Greedy Split-   │                                  │
│  │  Allocation (LP fallback  │                                  │
│  │  retained, not invoked)   │                                  │
│  └───────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
          ┌─────────────────────────────────────┐
          │       Termination Criterion?        │
          │    YES → Output Final Route Scheme  │
          │    NO → Return to ALNS Main Loop    │
          └─────────────────────────────────────┘
```

---

## 📝 Constraint Hierarchy

| Tier | Constraints | Verification Type |
|------|-------------|-------------------|
| **1** | Geometric & Capacity | Routing flow, compartment capacity, product compatibility |
| **2** | Static LDD Stability | Axle load bounds, centre-of-gravity limits |
| **3** | Dynamic Anti-Sloshing | Fill-ratio avoidance [0.35, 0.55] + physical validation |

---

## 📊 Performance Metrics

| Category | Metrics |
|----------|---------|
| **Economic Efficiency** | Total Distance, Transportation Cost, Fleet Utilization, Volume Utilization |
| **Safety Compliance** | LDD Violation Rate, CG Compliance Rate, Stability Margin, Slosh Risk Score |
| **Computational Performance** | CPU Time, Route-Penalty Cache Hit Rate, Iterations to Convergence |

A *verified route* is one that passes the sequence-aware feasibility check: cumulative volume at every delivery prefix ≤ vehicle capacity, and product count ≤ compartment count.

---

## 📄 Citation

If you use this code, data, or benchmarks in your research, please cite:

```bibtex
@article{Ndikuriyo2026,
  title   = {A Stability-Aware Adaptive Large Neighborhood Search for
             Multi-Compartment Container-Tanker Routing with Dynamic
             Stability Constraints},
  author  = {Ndikuriyo, Yves and Zhang, Yinggui},
  journal = {Computers \& Operations Research},
  year    = {2026},
  note    = {Under review}
}
```

The repository was previously published under the identifier `IALNS-4-DSC-CTRP` and has been renamed to `SA-ALNS-4-DSC-CTRP` for consistency with the manuscript. GitHub redirects the previous URL to the current one.

---

## 📬 Contact

| Author | Role | Email |
|--------|------|-------|
| Yves Ndikuriyo | First Author | yvesndikuriyo@csu.edu.cn |
| Yinggui Zhang | Corresponding Author | ygzhang@csu.edu.cn |

**Affiliation:**
School of Traffic and Transportation Engineering
Central South University
Changsha, Hunan 410075, China

---

## 🙏 Acknowledgments

This work was supported by:

- National Natural Science Foundation of China (Grant No. 71971220)
- Natural Science Foundation of Hunan Province (Grant Nos. 2023JJ30710 and 2022JJ31020)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please submit issues and pull requests through the GitHub repository.

---

**⭐ Star this repository if you find it useful!**
