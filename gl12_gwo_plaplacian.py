# -*- coding: utf-8 -*-
"""
Reproducibility code for:
A Hybrid Numerical Framework for Solving High Dimensional p-Laplacian Type
Fractional Systems with Caputo-Katugampola Derivatives via GL1-2 Discretization
and Grey Wolf Optimization.

This script implements the hybrid GL1-2 + Grey Wolf Optimization solver used in
Example 4.1 / the numerical section of the manuscript. It can run the full
experiment reported in the paper or a small smoke test for checking the software
environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as Gamma


def parameters(n_eq: int, N: int):
    """Numerical parameters used in the manuscript."""
    mu = 0.8
    gamma_n = 1.2
    nu = 0.75
    rho_i = 0.5
    p = 2
    q = 2
    alpha_n = 1.0 / N
    lam = 1.0
    sigma = 0.0
    return mu, gamma_n, nu, rho_i, p, q, alpha_n, lam, sigma


def laplace_initial_guess(N: int, s_vals: np.ndarray) -> np.ndarray:
    """Smooth initial approximation satisfying the boundary constraints."""
    y = s_vals**2 - s_vals**3
    y[0] = 0.0
    if N >= 2:
        y[-1] = y[-2]
    if N > 3:
        y[1] = y[0]
        y[2] = 0.5 * (y[1] + y[3])
    return y


def tempered_norm(y_vals: np.ndarray, alpha_vals: np.ndarray, p: float) -> float:
    """Truncated tempered l_p^alpha norm."""
    return float((np.sum((alpha_vals * np.abs(y_vals)) ** p)) ** (1.0 / p))


def gl12_caputo_katugampola(y_vals: np.ndarray, t_vals: np.ndarray, alpha: float, mu: float) -> np.ndarray:
    """GL1-2 approximation for the Caputo-Katugampola derivative, 0 < alpha < 1."""
    N = len(y_vals)
    t_mu = t_vals**mu
    h = (t_mu[-1] - t_mu[0]) / (N - 1)
    D = np.zeros(N, dtype=float)

    def a_coeff(kj: int) -> float:
        return (kj + 1) ** (1 - alpha) - kj ** (1 - alpha)

    def b_coeff(kj: int) -> float:
        return (1 / (2 - alpha)) * ((kj + 1) ** (2 - alpha) - kj ** (2 - alpha)) - 0.5 * (
            (kj + 1) ** (1 - alpha) + kj ** (1 - alpha)
        )

    for k in range(2, N):
        diff1 = (y_vals[1 : k + 1] - y_vals[0:k]) / h
        diff2 = (y_vals[2 : k + 1] - 2 * y_vals[1:k] + y_vals[0 : k - 1]) / (h**2)

        a_idx = np.array([a_coeff(k - j) for j in range(1, k + 1)])
        b_idx = np.array([b_coeff(k - j) for j in range(2, k + 1)])

        sum_a = np.dot(a_idx, diff1)
        sum_b = np.dot(b_idx, diff2)

        D[k] = (mu**alpha) * (h ** (1 - alpha)) / Gamma(2 - alpha) * sum_a + (
            mu**alpha
        ) * (h ** (2 - alpha)) / Gamma(2 - alpha) * sum_b
    return D


def caputo_katugampola_general(y_vals: np.ndarray, t_vals: np.ndarray, rho: float, mu: float) -> np.ndarray:
    """General order wrapper: rho = m + beta."""
    m = int(np.floor(rho))
    beta = rho - m

    y_m = np.array(y_vals, dtype=float)
    for _ in range(m):
        y_m = np.gradient(y_m, t_vals)

    if beta < 1e-12:
        return y_m

    return gl12_caputo_katugampola(y_m, t_vals, beta, mu)


def g_n(s: float, y: np.ndarray, z: np.ndarray, n: int = 1, epsilon: float = 1e-12) -> float:
    """Nonlinear source term adapted from the reference example."""
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    term1 = (s * np.cos(n * np.pi * s)) / (((s + n) ** 6) * (n**2))

    total = 0.0
    K = len(y)
    for k in range(1, K + 1):
        denom = max((k + s) ** 9, epsilon)
        total += np.cos(k * np.pi * s) * (y[k - 1] + z[k - 1]) / denom

    term2 = (s**0.5) * total / (4.0 * (5.0 - s) ** 2 * (s + n) ** 2)
    return float(term1 + term2)


def Q_kernel(s: float, r: float, mu: float, gamma_val: float, nu: float, rho: float, lam: float, sigma: float) -> float:
    """Green kernel used in the fixed-point operator."""
    s_mu, r_mu, lam_mu = s**mu, r**mu, lam**mu
    power = nu + gamma_val - 1.0

    Lambda = (Gamma(nu + gamma_val) * Gamma(nu + 2.0)) / (Gamma(nu + gamma_val + 2.0) * Gamma(nu))
    Lambda *= 1.0 - sigma * lam ** (mu * (nu + 1.0))

    if abs(Lambda) < 1e-14:
        return 0.0

    def base_core(x_mu: float, y_mu: float) -> float:
        if x_mu >= y_mu:
            return float((mu ** (nu + gamma_val)) * (x_mu - y_mu) ** power / Gamma(nu + gamma_val))
        return 0.0

    def boundary_corr(s_mu_local: float, r_mu_local: float) -> float:
        return float(
            (s_mu_local ** (nu + 1.0))
            / Lambda
            * (sigma * max(lam_mu - r_mu_local, 0.0) ** power - max(1.0 - r_mu_local, 0.0) ** power)
        )

    if s <= lam:
        if 0.0 <= r <= s:
            return base_core(s_mu, r_mu) + boundary_corr(s_mu, r_mu)
        if s < r <= lam:
            return boundary_corr(s_mu, r_mu)
        return float(-(s_mu ** (nu + 1.0)) / Lambda * max(1.0 - r_mu, 0.0) ** power)

    if 0.0 <= r <= s:
        return base_core(s_mu, r_mu) + boundary_corr(s_mu, r_mu)
    if s < r <= lam:
        return base_core(s_mu, r_mu) - float((sigma * s_mu ** (nu + 1.0)) / Lambda * max(1.0 - r_mu, 0.0) ** power)
    return float(-(s_mu ** (nu + 1.0)) / Lambda * max(1.0 - r_mu, 0.0) ** power)


def F_p_operator(
    s_vals: np.ndarray,
    y_vals: np.ndarray,
    mu: float,
    gamma_val: float,
    nu: float,
    rho: float,
    lam: float,
    sigma: float,
    p: float,
    n_eq: int,
) -> np.ndarray:
    """Fixed-point operator F[y](s) = int_0^1 Q(s,r) g_n(r,y,D_rho y) dr."""
    N = len(s_vals)
    F_vals = np.zeros_like(y_vals, dtype=float)

    z_vals = caputo_katugampola_general(y_vals, s_vals, rho, mu)

    dx = np.diff(s_vals)
    w = np.zeros_like(s_vals, dtype=float)
    if N > 1:
        w[0] = dx[0] / 2.0
        w[-1] = dx[-1] / 2.0
    if N > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2.0

    g_vec = np.array([g_n(r, y_vals, z_vals, n=n_eq) for r in s_vals], dtype=float)

    for i, s in enumerate(s_vals):
        Q_row = np.array([Q_kernel(s, r, mu, gamma_val, nu, rho, lam, sigma) for r in s_vals], dtype=float)
        F_vals[i] = np.sum(Q_row * g_vec * w)

    return F_vals


def gwo_p_laplacian(
    dim: int,
    n_eq: int = 1,
    n_wolves: int = 200,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int | None = 42,
):
    """Simplified alpha-guided GWO residual minimization."""
    if seed is not None:
        np.random.seed(seed)

    s_vals = np.linspace(0, 1, dim)
    mu, gamma_n, nu, rho_i, p, _q, alpha_n, lam, sigma = parameters(n_eq, dim)
    alpha_vals = np.full(dim, alpha_n)

    wolves = np.array([laplace_initial_guess(dim, s_vals) for _ in range(n_wolves)], dtype=float)
    wolves += np.random.normal(scale=0.01, size=wolves.shape)
    wolves[:, 0] = 0.0
    wolves[:, -1] = wolves[:, -2]

    alpha_pos = wolves[0].copy()
    alpha_score = np.inf
    convergence_curve = []
    solutions_over_time = []

    for t in range(max_iter):
        for i in range(n_wolves):
            try:
                Fw = F_p_operator(s_vals, wolves[i], mu, gamma_n, nu, rho_i, lam, sigma, p, n_eq)
                fitness = tempered_norm(wolves[i] - Fw, alpha_vals, p)
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha_pos = wolves[i].copy()
            except Exception:
                continue

        convergence_curve.append(float(alpha_score))
        solutions_over_time.append(alpha_pos.copy())

        a_param = 2 - t * (2 / max_iter)
        r1 = np.random.rand(n_wolves, dim)
        r2 = np.random.rand(n_wolves, dim)
        A = 2 * a_param * r1 - a_param
        C = 2 * r2
        D = np.abs(C * alpha_pos - wolves)
        wolves = alpha_pos - A * D
        wolves[:, 0] = 0.0
        wolves[:, -1] = wolves[:, -2]

        if alpha_score < tol:
            break

    return alpha_pos, convergence_curve, solutions_over_time, s_vals, wolves


def compute_regional_amplitudes(s_vals: np.ndarray, profile: np.ndarray):
    """Compute regional amplitude values from a numerical profile, e.g. z(u)=D_rho w(u)."""
    intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    amplitudes = []
    for a, b in intervals:
        if b < 1.0:
            mask = (s_vals >= a) & (s_vals < b)
        else:
            mask = (s_vals >= a) & (s_vals <= b)
        amplitudes.append(float(np.max(np.abs(profile[mask])) if np.any(mask) else 0.0))
    labels = [f"[{a:.1f}, {b:.1f}]" for a, b in intervals]
    return labels, amplitudes


def save_main_figures(output_dir: Path, s_vals, solution, convergence, solutions_over_time, elapsed_time):
    """Save figures corresponding to the manuscript's numerical section."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    stride = max(1, len(solutions_over_time) // 10)
    for idx in range(0, len(solutions_over_time), stride):
        plt.plot(s_vals, solutions_over_time[idx], label=f"Iter {idx}", alpha=0.5)
    plt.plot(s_vals, solution, "r--", linewidth=2.5, label="Final Solution")
    plt.title("GWO - Evolution of the Solution by Iteration")
    plt.xlabel("s")
    plt.ylabel("w(s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(convergence) + 1), convergence, marker="o", label="Residual (Norm)")
    plt.axhline(y=1e-5, color="gray", linestyle="--", label="Convergence Threshold")
    plt.title("GWO - Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Residual (Norm)")
    plt.legend()
    plt.grid(True)
    plt.suptitle(f"Solution of p-Laplacian with GWO | Duration: {elapsed_time:.2f} s", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / "figure3_solution_and_convergence.png", dpi=300)
    plt.close()

    N = len(s_vals)
    mu, gamma_n, nu, rho_i, p, q, alpha_n, lam, sigma = parameters(0, N)
    z_vals = caputo_katugampola_general(solution, s_vals, rho_i, mu)
    labels, amplitudes = compute_regional_amplitudes(s_vals, z_vals)
    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, amplitudes, alpha=0.85)
    for bar, amp in zip(bars, amplitudes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{amp:.4f}", ha="center", va="bottom", fontsize=9)
    plt.title("Regional Amplitude Analysis of z(u)=D_rho w(u)")
    plt.xlabel("Subinterval u in [0,1]")
    plt.ylabel("Amplitude")
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_regional_amplitude.png", dpi=300)
    plt.close()

    norm_curve = []
    for sol in solutions_over_time:
        norm_curve.append(tempered_norm(sol, np.full(N, alpha_n), p=p))
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(norm_curve) + 1), norm_curve, marker="s")
    plt.title("GWO - Norm Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "norm_evolution.png", dpi=300)
    plt.close()

    return labels, amplitudes, norm_curve


def run_experiment(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    solution, convergence, solutions_over_time, s_vals, _all_wolves = gwo_p_laplacian(
        dim=args.N,
        n_eq=args.n_eq,
        n_wolves=args.n_wolves,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
    )
    elapsed_time = time.time() - start_time

    mu, gamma_n, nu, rho_i, p, q, alpha_n, lam, sigma = parameters(args.n_eq, args.N)
    residual_final = convergence[-1] if convergence else None
    solution_norm = tempered_norm(solution, np.full(args.N, alpha_n), p=p)

    labels, amplitudes, norm_curve = save_main_figures(output_dir, s_vals, solution, convergence, solutions_over_time, elapsed_time)

    np.savetxt(output_dir / "solution.csv", np.column_stack([s_vals, solution]), delimiter=",", header="s,w", comments="")
    np.savetxt(output_dir / "convergence.csv", np.column_stack([np.arange(1, len(convergence) + 1), convergence]), delimiter=",", header="iteration,residual_norm", comments="")
    np.savetxt(output_dir / "regional_amplitudes.csv", np.column_stack([labels, amplitudes]), delimiter=",", fmt="%s", header="subinterval,amplitude", comments="")

    summary = {
        "title": "Hybrid GL1-2-GWO p-Laplacian fractional system experiment",
        "N": args.N,
        "n_eq": args.n_eq,
        "n_wolves": args.n_wolves,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
        "mu": mu,
        "gamma_n": gamma_n,
        "nu": nu,
        "rho_i": rho_i,
        "p": p,
        "q": q,
        "alpha_n": alpha_n,
        "lambda": lam,
        "sigma": sigma,
        "w_0": float(solution[0]),
        "w_1": float(solution[-1]),
        "solution_norm": solution_norm,
        "final_residual_norm": float(residual_final) if residual_final is not None else None,
        "runtime_seconds": elapsed_time,
        "iterations_completed": len(convergence),
        "regional_amplitudes": dict(zip(labels, amplitudes)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.run_sensitivity:
        plt.figure(figsize=(10, 6))
        sensitivity = {}
        for nw in args.sensitivity_wolves:
            print(f"Running population sensitivity: n_wolves = {nw}")
            _, conv, _, _, _ = gwo_p_laplacian(dim=args.N, n_eq=args.n_eq, n_wolves=nw, max_iter=args.max_iter, tol=args.tol, seed=args.seed + nw if args.seed is not None else None)
            plt.plot(range(1, len(conv) + 1), conv, label=f"$n_{{wolves}}$ = {nw}")
            sensitivity[str(nw)] = conv
        plt.title("Convergence history for different Grey Wolf population sizes")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (Norm)")
        plt.axhline(y=args.tol, color="gray", linestyle="--", label="Convergence Threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "figure4_population_sensitivity.png", dpi=300)
        plt.close()
        (output_dir / "population_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the GL1-2-GWO p-Laplacian fractional system experiment.")
    parser.add_argument("--N", type=int, default=100, help="Number of discretization points.")
    parser.add_argument("--n-eq", type=int, default=1, help="Equation/component index n.")
    parser.add_argument("--n-wolves", type=int, default=200, help="Grey Wolf population size.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum GWO iterations.")
    parser.add_argument("--tol", type=float, default=1e-5, help="Residual tolerance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, default="results", help="Output directory for figures and numerical files.")
    parser.add_argument("--run-sensitivity", action="store_true", help="Also run population-size sensitivity experiment for Figure 4.")
    parser.add_argument("--sensitivity-wolves", type=int, nargs="+", default=[50, 100, 150, 200], help="Population sizes used in sensitivity analysis.")
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke test with reduced N, population size, and iterations.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.N = 30
        args.n_wolves = 8
        args.max_iter = 5
        args.run_sensitivity = False
        args.output = "results_quick"
    run_experiment(args)


if __name__ == "__main__":
    main()
