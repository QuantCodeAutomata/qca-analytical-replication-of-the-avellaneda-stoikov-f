"""
Experiment 4: Optional infinite-horizon analytical extension of the
Avellaneda-Stoikov model.

Reproduces the paper's separate infinite-horizon formulation as an analytical
extension. Focuses on the discounted infinite-horizon market-making problem,
stationarity conditions, and the role of the discount parameter omega.

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_infinite_horizon_params(
    gamma: float,
    sigma: float,
    omega: float,
    q_max: int,
) -> None:
    """Validate infinite-horizon model parameters.

    Parameters
    ----------
    gamma : float
        CARA risk-aversion. Must be > 0.
    sigma : float
        Mid-price volatility. Must be >= 0.
    omega : float
        Discount parameter. Must be > 0.
    q_max : int
        Maximum inventory level. Must be >= 1.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if omega <= 0:
        raise ValueError(f"omega must be > 0, got {omega}")
    if q_max < 1:
        raise ValueError(f"q_max must be >= 1, got {q_max}")


# ---------------------------------------------------------------------------
# Admissibility condition
# ---------------------------------------------------------------------------

def admissibility_condition(gamma: float, sigma: float, q: float) -> float:
    """Compute the minimum omega required for admissibility at inventory level q.

    The paper's condition: omega > 0.5 * gamma^2 * sigma^2 * q^2

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    q : float
        Inventory level.

    Returns
    -------
    float
        Minimum omega required: 0.5 * gamma^2 * sigma^2 * q^2.
    """
    # Custom — paper Eq. admissibility condition
    return 0.5 * gamma**2 * sigma**2 * q**2


def suggested_omega(gamma: float, sigma: float, q_max: int) -> float:
    """Compute the paper's suggested omega for a bounded inventory grid.

    Paper's choice: omega = 0.5 * gamma^2 * sigma^2 * (q_max + 1)^2

    This ensures omega > 0.5 * gamma^2 * sigma^2 * q^2 for all q in [-q_max, q_max].

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    q_max : int
        Maximum inventory level.

    Returns
    -------
    float
        Suggested omega value.
    """
    # Custom — paper Eq. suggested omega
    return 0.5 * gamma**2 * sigma**2 * (q_max + 1)**2


def check_admissibility(
    gamma: float, sigma: float, omega: float, q_max: int
) -> Dict[int, Dict]:
    """Check admissibility condition for all inventory levels in [-q_max, q_max].

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount parameter.
    q_max : int
        Maximum inventory level.

    Returns
    -------
    dict
        For each inventory level: minimum omega required, actual omega, and whether satisfied.
    """
    results = {}
    for q in range(-q_max, q_max + 1):
        min_omega = admissibility_condition(gamma, sigma, q)
        results[q] = {
            "min_omega_required": min_omega,
            "actual_omega": omega,
            "admissible": omega > min_omega,
        }
    return results


# ---------------------------------------------------------------------------
# Stationary HJB and value function structure
# ---------------------------------------------------------------------------

def stationary_theta_approximation(
    q: int,
    gamma: float,
    sigma: float,
    omega: float,
    k: float,
    A: float,
) -> float:
    """Approximate stationary theta(q) for the infinite-horizon problem.

    In the infinite-horizon setting, the stationary value function takes the form:
        u(x, q) = -exp(-gamma*x) * exp(-gamma*theta(q))

    The stationary theta satisfies a system of ODEs/equations. For the
    exponential intensity case lambda(delta) = A*exp(-k*delta), the stationary
    reservation price-like quantity can be approximated.

    The key insight: in the infinite-horizon case, the inventory effect is
    controlled by omega rather than (T-t). The effective "time-to-maturity"
    analog is 1/omega (the discount horizon).

    Approximate stationary theta:
        theta(q) ≈ -q^2 * gamma * sigma^2 / (2*omega)

    This is derived by analogy with the finite-horizon formula where (T-t)
    is replaced by 1/omega.

    Parameters
    ----------
    q : int
        Inventory level.
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount parameter.
    k : float
        Exponential intensity decay.
    A : float
        Intensity scale.

    Returns
    -------
    float
        Approximate stationary theta(q).
    """
    # Custom — paper infinite-horizon stationary approximation
    # Analogy: replace (T-t) with 1/omega in finite-horizon formula
    return -q**2 * gamma * sigma**2 / (2.0 * omega)


def stationary_reservation_price(
    s: float,
    q: int,
    gamma: float,
    sigma: float,
    omega: float,
) -> float:
    """Compute the stationary reservation price for the infinite-horizon problem.

    By analogy with the finite-horizon formula, replacing (T-t) with 1/omega:
        r_inf(s, q) = s - q * gamma * sigma^2 / omega

    Parameters
    ----------
    s : float
        Current mid-price.
    q : int
        Current inventory.
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount parameter.

    Returns
    -------
    float
        Stationary reservation price.
    """
    # Custom — paper infinite-horizon reservation price (analogy)
    return s - q * gamma * sigma**2 / omega


def stationary_spread_offset(gamma: float, k: float) -> float:
    """Compute the stationary spread offset (1/gamma)*ln(1 + gamma/k).

    This is the same as in the finite-horizon case; the spread offset from
    the exponential intensity first-order conditions does not depend on time.

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    k : float
        Exponential intensity decay.

    Returns
    -------
    float
        Stationary spread offset.
    """
    # Custom — paper Eq. spread offset (same as finite-horizon)
    return (1.0 / gamma) * math.log(1.0 + gamma / k)


def stationary_total_spread(gamma: float, sigma: float, omega: float, k: float) -> float:
    """Compute the stationary total spread for the infinite-horizon problem.

    By analogy with finite-horizon (replacing gamma*sigma^2*(T-t) with gamma*sigma^2/omega):
        Spread_inf = gamma*sigma^2/omega + (2/gamma)*ln(1 + gamma/k)

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount parameter.
    k : float
        Exponential intensity decay.

    Returns
    -------
    float
        Stationary total spread.
    """
    # Custom — paper infinite-horizon total spread (analogy)
    return gamma * sigma**2 / omega + (2.0 / gamma) * math.log(1.0 + gamma / k)


# ---------------------------------------------------------------------------
# Numerical stationary solution on bounded inventory grid
# ---------------------------------------------------------------------------

def solve_stationary_bellman(
    gamma: float,
    sigma: float,
    omega: float,
    k: float,
    A: float,
    q_max: int,
    max_iter: int = 10000,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Solve the stationary Bellman system on a bounded inventory grid.

    The stationary HJB for the transformed variable w(q) = exp(-gamma*theta(q))
    (where u = -exp(-gamma*x)*w(q)) leads to a system of equations.

    For the exponential intensity case, the stationary equations for w(q) are:
        omega * w(q) = 0.5 * gamma^2 * sigma^2 * (w(q+1) - 2*w(q) + w(q-1))
                     + max_{delta^a} [lambda^a(delta^a) * (w(q-1)*exp(gamma*delta^a) - w(q))]
                     + max_{delta^b} [lambda^b(delta^b) * (w(q+1)*exp(gamma*delta^b) - w(q))]

    With optimal distances:
        delta^a* = (1/k)*ln(A*w(q-1)/w(q)) + (1/gamma)*ln(1 + gamma/k)  [if w(q-1) > 0]
        delta^b* = (1/k)*ln(A*w(q+1)/w(q)) + (1/gamma)*ln(1 + gamma/k)  [if w(q+1) > 0]

    Boundary conditions: w(-q_max-1) = w(q_max+1) = 0 (reflecting/absorbing).

    Uses value iteration (fixed-point iteration) on the w vector.

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount parameter.
    k : float
        Exponential intensity decay.
    A : float
        Intensity scale.
    q_max : int
        Maximum inventory level.
    max_iter : int
        Maximum iterations for fixed-point solver.
    tol : float
        Convergence tolerance.

    Returns
    -------
    tuple
        (q_grid, w_values, converged)
    """
    validate_infinite_horizon_params(gamma, sigma, omega, q_max)

    # Inventory grid: q in {-q_max, ..., q_max}
    q_grid = np.arange(-q_max, q_max + 1)
    n = len(q_grid)  # 2*q_max + 1

    # Initialize w with approximate solution
    # w(q) = exp(-gamma*theta(q)) where theta(q) = -q^2*gamma*sigma^2/(2*omega)
    w = np.array([
        math.exp(-gamma * stationary_theta_approximation(q, gamma, sigma, omega, k, A))
        for q in q_grid
    ])
    w = w / w[q_max]  # normalize so w(0) = 1

    # Boundary: w at q = ±(q_max+1) = 0 (absorbing boundary)
    def get_w(idx: int) -> float:
        """Get w value with boundary conditions."""
        if idx < 0 or idx >= n:
            return 0.0
        return w[idx]

    converged = False
    for iteration in range(max_iter):
        w_new = np.zeros(n)

        for i, q in enumerate(q_grid):
            w_q = w[i]
            w_qp1 = get_w(i + 1)  # w(q+1)
            w_qm1 = get_w(i - 1)  # w(q-1)

            # Diffusion term: 0.5*gamma^2*sigma^2*(w(q+1) - 2*w(q) + w(q-1))
            diffusion = 0.5 * gamma**2 * sigma**2 * (w_qp1 - 2 * w_q + w_qm1)

            # Optimal ask side: maximize over delta^a
            # Optimal: delta^a* = (1/k)*ln(A*w(q-1)/w(q)) + (1/gamma)*ln(1+gamma/k)
            # Contribution: lambda^a*(delta^a*) * (w(q-1)*exp(gamma*delta^a*) - w(q))
            if w_qm1 > 1e-15 and w_q > 1e-15:
                ratio_a = A * w_qm1 / w_q
                if ratio_a > 0:
                    delta_a_star = (1.0 / k) * math.log(ratio_a) + (1.0 / gamma) * math.log(1.0 + gamma / k)
                    lam_a = A * math.exp(-k * delta_a_star)
                    ask_contribution = lam_a * (w_qm1 * math.exp(gamma * delta_a_star) - w_q)
                else:
                    ask_contribution = 0.0
            else:
                ask_contribution = 0.0

            # Optimal bid side: maximize over delta^b
            if w_qp1 > 1e-15 and w_q > 1e-15:
                ratio_b = A * w_qp1 / w_q
                if ratio_b > 0:
                    delta_b_star = (1.0 / k) * math.log(ratio_b) + (1.0 / gamma) * math.log(1.0 + gamma / k)
                    lam_b = A * math.exp(-k * delta_b_star)
                    bid_contribution = lam_b * (w_qp1 * math.exp(gamma * delta_b_star) - w_q)
                else:
                    bid_contribution = 0.0
            else:
                bid_contribution = 0.0

            # Stationary equation: omega*w(q) = diffusion + ask + bid
            # => w_new(q) = (diffusion + ask + bid) / omega
            w_new[i] = (diffusion + ask_contribution + bid_contribution) / omega

        # Normalize to prevent drift
        if w_new[q_max] > 1e-15:
            w_new = w_new / w_new[q_max]

        # Check convergence
        delta_w = np.max(np.abs(w_new - w))
        w = w_new.copy()

        if delta_w < tol:
            converged = True
            break

    return q_grid, w, converged


def compute_stationary_quotes(
    q_grid: np.ndarray,
    w: np.ndarray,
    s: float,
    gamma: float,
    k: float,
    A: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute stationary optimal quote distances from the solved w vector.

    Parameters
    ----------
    q_grid : np.ndarray
        Inventory grid.
    w : np.ndarray
        Solved stationary w values.
    s : float
        Current mid-price (for reference).
    gamma : float
        CARA risk-aversion.
    k : float
        Exponential intensity decay.
    A : float
        Intensity scale.

    Returns
    -------
    tuple
        (delta_ask, delta_bid, reservation_prices)
    """
    n = len(q_grid)
    delta_ask = np.full(n, np.nan)
    delta_bid = np.full(n, np.nan)
    res_prices = np.zeros(n)

    def get_w(idx: int) -> float:
        if idx < 0 or idx >= n:
            return 0.0
        return w[idx]

    for i, q in enumerate(q_grid):
        w_q = w[i]
        w_qp1 = get_w(i + 1)
        w_qm1 = get_w(i - 1)

        # Reservation price from theta differences
        # theta(q) = -(1/gamma)*ln(w(q))
        # r^a = theta(q) - theta(q-1) = (1/gamma)*ln(w(q-1)/w(q))
        # r^b = theta(q+1) - theta(q) = (1/gamma)*ln(w(q+1)/w(q))
        if w_qm1 > 1e-15 and w_q > 1e-15:
            r_a = (1.0 / gamma) * math.log(w_qm1 / w_q)
            ratio_a = A * w_qm1 / w_q
            if ratio_a > 0:
                delta_ask[i] = (1.0 / k) * math.log(ratio_a) + (1.0 / gamma) * math.log(1.0 + gamma / k)

        if w_qp1 > 1e-15 and w_q > 1e-15:
            r_b = (1.0 / gamma) * math.log(w_qp1 / w_q)
            ratio_b = A * w_qp1 / w_q
            if ratio_b > 0:
                delta_bid[i] = (1.0 / k) * math.log(ratio_b) + (1.0 / gamma) * math.log(1.0 + gamma / k)

        # Approximate reservation price
        theta_q = -(1.0 / gamma) * math.log(w_q) if w_q > 1e-15 else 0.0
        res_prices[i] = s + theta_q  # simplified; full form requires s-dependence

    return delta_ask, delta_bid, res_prices


# ---------------------------------------------------------------------------
# Conceptual comparison: finite vs infinite horizon
# ---------------------------------------------------------------------------

def compare_finite_infinite_horizon(
    gamma: float,
    sigma: float,
    k: float,
    q_max: int = 5,
    T: float = 1.0,
    t_values: List[float] = None,
) -> Dict:
    """Compare finite-horizon and infinite-horizon inventory effects.

    In finite horizon: inventory effect = q*gamma*sigma^2*(T-t) -> 0 as t->T
    In infinite horizon: inventory effect = q*gamma*sigma^2/omega (constant)

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    k : float
        Exponential intensity decay.
    q_max : int
        Maximum inventory level.
    T : float
        Terminal horizon for finite-horizon comparison.
    t_values : list of float
        Time values for finite-horizon comparison.

    Returns
    -------
    dict
        Comparison results.
    """
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]

    omega = suggested_omega(gamma, sigma, q_max)
    s = 100.0

    # Finite-horizon inventory effects at q=1
    finite_effects = {}
    for t in t_values:
        tau = T - t
        effect = gamma * sigma**2 * tau  # inventory effect per unit q
        finite_effects[t] = effect

    # Infinite-horizon inventory effect (constant)
    inf_effect = gamma * sigma**2 / omega

    return {
        "gamma": gamma,
        "sigma": sigma,
        "omega": omega,
        "q_max": q_max,
        "finite_horizon_effects": finite_effects,
        "infinite_horizon_effect": inf_effect,
        "key_difference": (
            "Finite-horizon: inventory effect decays to 0 as t->T. "
            "Infinite-horizon: inventory effect is constant = gamma*sigma^2/omega."
        ),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

@dataclass
class InfiniteHorizonResults:
    """Container for infinite-horizon experiment results."""
    gamma: float
    sigma: float
    omega: float
    q_max: int
    admissibility_checks: Dict
    omega_condition_satisfied: bool
    stationary_q_grid: np.ndarray
    stationary_w: np.ndarray
    stationary_delta_ask: np.ndarray
    stationary_delta_bid: np.ndarray
    stationary_res_prices: np.ndarray
    solver_converged: bool
    finite_infinite_comparison: Dict


def run_experiment_4(
    gamma_values: List[float] = None,
    sigma: float = 2.0,
    k: float = 1.5,
    A: float = 140.0,
    q_max: int = 5,
    output_dir: str = "results",
) -> Dict[float, InfiniteHorizonResults]:
    """Run Experiment 4: Infinite-horizon analytical extension.

    Parameters
    ----------
    gamma_values : list of float
        Risk-aversion values (default [0.1, 0.01, 0.5]).
    sigma : float
        Mid-price volatility.
    k : float
        Exponential intensity decay.
    A : float
        Intensity scale.
    q_max : int
        Maximum inventory level.
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict
        Results indexed by gamma.
    """
    if gamma_values is None:
        gamma_values = [0.1, 0.01, 0.5]

    import os
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    print("=" * 70)
    print("EXPERIMENT 4: Infinite-Horizon Analytical Extension")
    print("=" * 70)
    print(f"Parameters: sigma={sigma}, k={k}, A={A}, q_max={q_max}")
    print()

    for gamma in gamma_values:
        omega = suggested_omega(gamma, sigma, q_max)

        print(f"gamma = {gamma}")
        print(f"  Suggested omega = 0.5*gamma^2*sigma^2*(q_max+1)^2 = {omega:.6f}")

        # Admissibility checks
        adm_checks = check_admissibility(gamma, sigma, omega, q_max)
        all_satisfied = all(v["admissible"] for v in adm_checks.values())
        print(f"  Admissibility condition omega > 0.5*gamma^2*sigma^2*q^2:")
        for q, info in adm_checks.items():
            print(f"    q={q:+d}: min_omega={info['min_omega_required']:.6f}, "
                  f"omega={info['actual_omega']:.6f}, OK={info['admissible']}")

        # Numerical stationary solution
        q_grid, w, converged = solve_stationary_bellman(
            gamma, sigma, omega, k, A, q_max
        )
        print(f"  Stationary Bellman solver converged: {converged}")

        # Compute stationary quotes
        delta_ask, delta_bid, res_prices = compute_stationary_quotes(
            q_grid, w, 100.0, gamma, k, A
        )

        print(f"  Stationary quote distances (delta^a, delta^b) by inventory:")
        for i, q in enumerate(q_grid):
            da = f"{delta_ask[i]:.4f}" if not np.isnan(delta_ask[i]) else "N/A"
            db = f"{delta_bid[i]:.4f}" if not np.isnan(delta_bid[i]) else "N/A"
            print(f"    q={q:+d}: delta^a={da}, delta^b={db}")

        # Finite vs infinite comparison
        comparison = compare_finite_infinite_horizon(gamma, sigma, k, q_max)
        print(f"  Infinite-horizon inventory effect (per unit q): "
              f"{comparison['infinite_horizon_effect']:.6f}")
        print(f"  Finite-horizon inventory effect at t=0: "
              f"{comparison['finite_horizon_effects'][0.0]:.6f}")
        print(f"  Key difference: {comparison['key_difference']}")
        print()

        result = InfiniteHorizonResults(
            gamma=gamma,
            sigma=sigma,
            omega=omega,
            q_max=q_max,
            admissibility_checks=adm_checks,
            omega_condition_satisfied=all_satisfied,
            stationary_q_grid=q_grid,
            stationary_w=w,
            stationary_delta_ask=delta_ask,
            stationary_delta_bid=delta_bid,
            stationary_res_prices=res_prices,
            solver_converged=converged,
            finite_infinite_comparison=comparison,
        )
        all_results[gamma] = result

    # Generate plots
    _plot_stationary_quotes(all_results, output_dir)
    _plot_omega_admissibility(gamma_values, sigma, q_max, output_dir)

    return all_results


def _plot_stationary_quotes(
    results: Dict[float, InfiniteHorizonResults],
    output_dir: str,
) -> None:
    """Plot stationary quote distances vs inventory for each gamma.

    Parameters
    ----------
    results : dict
        Results from run_experiment_4.
    output_dir : str
        Directory to save the plot.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (gamma, res) in zip(axes, results.items()):
        q_grid = res.stationary_q_grid
        valid_ask = ~np.isnan(res.stationary_delta_ask)
        valid_bid = ~np.isnan(res.stationary_delta_bid)

        if valid_ask.any():
            ax.plot(q_grid[valid_ask], res.stationary_delta_ask[valid_ask],
                    "g-o", label="$\\delta^a$ (ask)", markersize=6)
        if valid_bid.any():
            ax.plot(q_grid[valid_bid], res.stationary_delta_bid[valid_bid],
                    "r-s", label="$\\delta^b$ (bid)", markersize=6)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Inventory $q$")
        ax.set_ylabel("Quote distance $\\delta$")
        ax.set_title(f"Stationary Quotes — $\\gamma={gamma}$\n$\\omega={res.omega:.4f}$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Exp 4: Infinite-Horizon Stationary Quote Distances", fontsize=12)
    plt.tight_layout()
    fname = os.path.join(output_dir, "exp4_stationary_quotes.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_omega_admissibility(
    gamma_values: List[float],
    sigma: float,
    q_max: int,
    output_dir: str,
) -> None:
    """Plot omega admissibility condition vs inventory for each gamma.

    Parameters
    ----------
    gamma_values : list of float
        Risk-aversion values.
    sigma : float
        Mid-price volatility.
    q_max : int
        Maximum inventory level.
    output_dir : str
        Directory to save the plot.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    q_range = np.arange(-q_max - 1, q_max + 2)
    fig, ax = plt.subplots(figsize=(10, 6))

    for gamma in gamma_values:
        omega = suggested_omega(gamma, sigma, q_max)
        min_omegas = [admissibility_condition(gamma, sigma, q) for q in q_range]
        ax.plot(q_range, min_omegas, "-o", markersize=5,
                label=f"$\\gamma={gamma}$, min $\\omega$ required")
        ax.axhline(omega, linestyle="--", alpha=0.7,
                   label=f"$\\omega={omega:.4f}$ (suggested, $\\gamma={gamma}$)")

    ax.set_xlabel("Inventory $q$")
    ax.set_ylabel("$\\omega$ value")
    ax.set_title(
        f"Admissibility: $\\omega > 0.5\\gamma^2\\sigma^2 q^2$\n"
        f"$\\sigma={sigma}$, $q_{{max}}={q_max}$"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "exp4_omega_admissibility.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiment_4(output_dir="results")
