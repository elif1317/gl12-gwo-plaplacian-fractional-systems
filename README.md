# GL1-2-GWO solver for p-Laplacian type fractional systems

This repository contains the Python code used for the numerical experiments in the manuscript:

**A Hybrid Numerical Framework for Solving High Dimensional p-Laplacian Type Fractional Systems with Caputo-Katugampola Derivatives via GL1-2 Discretization and Grey Wolf Optimization**

Authors: Elif Demir, Yusuf Zeren, and Alpaslan Demirci.

## Purpose

The code implements a hybrid numerical framework that combines a GL1-2 discretization of the Caputo-Katugampola fractional derivative with a Grey Wolf Optimization (GWO) residual-minimization procedure for a nonlinear p-Laplacian type fractional boundary value problem.

The default parameters reproduce the main numerical setting reported in the paper:

- `N = 100`
- `mu = 0.8`
- `rho = 0.5`
- `gamma = 1.2`
- `nu = 0.75`
- `p = 2`
- `n_wolves = 200`
- `max_iter = 100`
- `tol = 1e-5`
- `seed = 42`

## Repository contents

```text
gl12_gwo_plaplacian.py                         Main reproducibility script
requirements.txt                               Python dependencies
CITATION.cff                                   Citation metadata for GitHub and Zenodo
.zenodo.json                                   Metadata used by Zenodo archiving
LICENSE                                        MIT license
CODE_AVAILABILITY_STATEMENT.txt                Manuscript code availability statement template
original_updated_gwo_gl12_plaplace_sonnnn.py   Original author script
solution.csv                                   Representative numerical solution output
convergence.csv                                Representative residual convergence output
regional_amplitudes.csv                        Representative regional amplitude output
summary.json                                   Summary of the representative run
figure2_regional_amplitude.png                 Regional amplitude figure
norm_evolution.png                             Norm evolution figure
```

## Installation

Create a fresh Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code was written for Python 3.10 or newer.

## Quick software check

Run a reduced smoke test:

```bash
python gl12_gwo_plaplacian.py --quick
```

This creates a `results_quick/` directory with a short convergence run and saved figures.

## Full reproduction of the main run

Run the default experiment:

```bash
python gl12_gwo_plaplacian.py --output results_main
```

This saves:

- `summary.json`
- `solution.csv`
- `convergence.csv`
- `regional_amplitudes.csv`
- `figure2_regional_amplitude.png`
- `figure3_solution_and_convergence.png`
- `norm_evolution.png`

The full default setting is computationally expensive because the fractional-memory operator is evaluated inside a population-based optimizer. The runtime depends on the CPU and may take a long time.

## Population-size sensitivity experiment

To reproduce the population sensitivity analysis used for Figure 4, run:

```bash
python gl12_gwo_plaplacian.py --output results_sensitivity --run-sensitivity
```

This additionally saves:

- `figure4_population_sensitivity.png`
- `population_sensitivity.json`

## Notes on reproducibility

The GWO component is stochastic. The default run uses `seed = 42`. Small numerical differences may occur across Python, NumPy, SciPy, and hardware versions, but the residual-reduction trend and qualitative solution behavior should remain consistent.

## Citation

After archiving the repository on Zenodo, update `CITATION.cff`, `.zenodo.json`, and `CODE_AVAILABILITY_STATEMENT.txt` with the Zenodo DOI.
