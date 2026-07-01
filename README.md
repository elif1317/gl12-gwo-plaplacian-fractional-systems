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

The GWO implementation is an alpha-guided variant of the Grey Wolf Optimizer, consistent with the algorithmic description in the manuscript.

## Repository contents

```text
gl12_gwo_plaplacian.py          Main reproducibility script
requirements.txt                Python dependencies
CITATION.cff                    Citation metadata
.zenodo.json                    Metadata for optional Zenodo archiving
LICENSE                         MIT license
CODE_AVAILABILITY_STATEMENT.txt Code availability statement template
solution.csv                    Representative quick-check solution output
convergence.csv                 Representative quick-check convergence output
regional_amplitudes.csv         Representative quick-check regional amplitude output
summary.json                    Representative quick-check summary file
figure2_regional_amplitude.png  Representative regional amplitude figure
figure3_solution_and_convergence.png Representative solution and convergence figure
norm_evolution.png              Representative norm evolution figure
