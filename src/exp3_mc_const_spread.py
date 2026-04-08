"""
Experiment 3: Monte Carlo replication of finite-horizon inventory-based versus
symmetric quoting with table-faithful constant spread.

Uses the paper's reservation price formula and the constant spread convention
implied by the published simulation tables:
    Spread = (2/gamma)*ln(1 + gamma/k)

This branch maximizes fidelity to the paper's reported numerical tables.

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Simulation parameters (same as exp2 but with constant spread)
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """Simulation parameters for the Avellaneda-Stoikov model (constant spread)."""
    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    N: int = 200
    q0: int = 0
    X0: float = 0.0
    A: float = 140.0
    k: float = 1.5
    n_paths: int = 1000
    gamma: float = 0.1
    seed: int = 42
    use_common_random: bool = True


def validate_sim_params(p: SimParams) -> None:
    """Validate simulation parameters.

    Parameters
    ----------
    p : SimParams
        Simulation parameters to validate.
    """
    if p.gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {p.gamma}")
    if p.sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {p.sigma}")
    if p.T <= 0:
        raise ValueError(f"T must be > 0, got {p.T}")
    if p.dt <= 0:
        raise ValueError(f"dt must be > 0, got {p.dt}")
    if p.k <= 0:
        raise ValueError(f"k must be > 0, got {p.k}")
    if p.A <= 0:
        raise ValueError(f"A must be > 0, got {p.A}")
    assert abs(p.N * p.dt - p.T) < 1e-10, f"N*dt must equal T"


# ---------------------------------------------------------------------------
# Core formula functions
# ---------------------------------------------------------------------------

def reservation_price(s: float, q: int, t: float, gamma: float, sigma: float, T: float) -> float:
    """Compute reservation price r_t = s - q*gamma*sigma^2*(T-t).

    Parameters
    ----------
    s : float
        Current mid-price.
    q : int
        Current inventory.
    t : float
        Current time.
    gamma : float
        CARA risk-aversion.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.

    Returns
    -------
    float
        Reservation price.
    """
    # Custom — paper Eq. reservation price
    return s - q * gamma * sigma**2 * (T - t)


def constant_spread(gamma: float, k: float) -> float:
    """Compute the table-faithful constant total spread.

        Spread = (2/gamma)*ln(1 + gamma/k)

    This matches the published simulation table values.
    Expected values for k=1.5:
        gamma=0.1  -> ~1.29
        gamma=0.01 -> ~1.33
        gamma=0.5  -> ~1.15

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    k : float
        Exponential intensity decay.

    Returns
    -------
    float
        Constant total spread.
    """
    # Custom — paper table-implied constant spread formula
    return (2.0 / gamma) * math.log(1.0 + gamma / k)


def execution_intensity(delta: float, A: float, k: float) -> float:
    """Compute execution intensity lambda(delta) = A*exp(-k*delta).

    Parameters
    ----------
    delta : float
        Quote distance from mid-price.
    A : float
        Intensity scale.
    k : float
        Intensity decay.

    Returns
    -------
    float
        Execution intensity.
    """
    return A * math.exp(-k * delta)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsLog:
    """Log of edge cases and anomalies during simulation."""
    negative_delta_ask_count: int = 0
    negative_delta_bid_count: int = 0
    lambda_dt_exceed_1_ask: int = 0
    lambda_dt_exceed_1_bid: int = 0
    simultaneous_fills: int = 0
    total_steps: int = 0

    def report(self) -> str:
        """Generate a text report of diagnostics."""
        lines = [
            "Diagnostics Log:",
            f"  Total steps simulated: {self.total_steps}",
            f"  Negative delta^a events: {self.negative_delta_ask_count}",
            f"  Negative delta^b events: {self.negative_delta_bid_count}",
            f"  lambda^a*dt > 1 events (capped): {self.lambda_dt_exceed_1_ask}",
            f"  lambda^b*dt > 1 events (capped): {self.lambda_dt_exceed_1_bid}",
            f"  Simultaneous bid+ask fills: {self.simultaneous_fills}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-path simulation
# ---------------------------------------------------------------------------

def simulate_path_inventory(
    p: SimParams,
    spread: float,
    price_innovations: np.ndarray,
    ask_uniforms: np.ndarray,
    bid_uniforms: np.ndarray,
    diag: DiagnosticsLog,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Simulate one path of the inventory-adjusted quoting strategy (constant spread).

    Inventory strategy: center constant spread around reservation price r_t.
        p^a_t = r_t + Spread/2
        p^b_t = r_t - Spread/2

    Parameters
    ----------
    p : SimParams
        Simulation parameters.
    spread : float
        Pre-computed constant spread.
    price_innovations : np.ndarray
        Pre-drawn ±1 innovations for mid-price (shape: N).
    ask_uniforms : np.ndarray
        Pre-drawn uniform [0,1] for ask fill decisions (shape: N).
    bid_uniforms : np.ndarray
        Pre-drawn uniform [0,1] for bid fill decisions (shape: N).
    diag : DiagnosticsLog
        Diagnostics accumulator (modified in-place).

    Returns
    -------
    tuple
        (S_path, r_path, pa_path, pb_path, q_path, X_T, q_T)
    """
    S = p.s0
    q = p.q0
    X = p.X0
    half_spread = spread / 2.0

    S_path = np.zeros(p.N + 1)
    r_path = np.zeros(p.N + 1)
    pa_path = np.zeros(p.N + 1)
    pb_path = np.zeros(p.N + 1)
    q_path = np.zeros(p.N + 1, dtype=int)

    S_path[0] = S
    r_path[0] = reservation_price(S, q, 0.0, p.gamma, p.sigma, p.T)
    q_path[0] = q

    for n in range(p.N):
        t = n * p.dt

        # Compute reservation price
        r_t = reservation_price(S, q, t, p.gamma, p.sigma, p.T)

        # Inventory strategy: center constant spread around reservation price
        pa = r_t + half_spread
        pb = r_t - half_spread

        delta_a = pa - S  # ask distance from mid
        delta_b = S - pb  # bid distance from mid

        # Track negative deltas
        if delta_a < 0:
            diag.negative_delta_ask_count += 1
        if delta_b < 0:
            diag.negative_delta_bid_count += 1

        # Compute intensities
        lam_a = execution_intensity(delta_a, p.A, p.k)
        lam_b = execution_intensity(delta_b, p.A, p.k)

        prob_a = lam_a * p.dt
        prob_b = lam_b * p.dt

        if prob_a > 1.0:
            diag.lambda_dt_exceed_1_ask += 1
            prob_a = 1.0
        if prob_b > 1.0:
            diag.lambda_dt_exceed_1_bid += 1
            prob_b = 1.0

        diag.total_steps += 1

        ask_fill = ask_uniforms[n] < prob_a
        bid_fill = bid_uniforms[n] < prob_b

        if ask_fill and bid_fill:
            diag.simultaneous_fills += 1

        if ask_fill:
            X += pa
            q -= 1
        if bid_fill:
            X -= pb
            q += 1

        # Mid-price update: ±sigma*sqrt(dt) with prob 0.5
        S += price_innovations[n] * p.sigma * math.sqrt(p.dt)

        S_path[n + 1] = S
        r_path[n + 1] = reservation_price(S, q, (n + 1) * p.dt, p.gamma, p.sigma, p.T)
        pa_path[n] = pa
        pb_path[n] = pb
        q_path[n + 1] = q

    pa_path[p.N] = S
    pb_path[p.N] = S

    return S_path, r_path, pa_path, pb_path, q_path, X, q


def simulate_path_symmetric(
    p: SimParams,
    spread: float,
    price_innovations: np.ndarray,
    ask_uniforms: np.ndarray,
    bid_uniforms: np.ndarray,
    diag: DiagnosticsLog,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Simulate one path of the symmetric quoting strategy (constant spread).

    Symmetric strategy: center constant spread around mid-price S_t.
        p^a_t = S_t + Spread/2
        p^b_t = S_t - Spread/2
    So delta^a = delta^b = Spread/2 always.

    Parameters
    ----------
    p : SimParams
        Simulation parameters.
    spread : float
        Pre-computed constant spread.
    price_innovations : np.ndarray
        Pre-drawn ±1 innovations for mid-price (shape: N).
    ask_uniforms : np.ndarray
        Pre-drawn uniform [0,1] for ask fill decisions (shape: N).
    bid_uniforms : np.ndarray
        Pre-drawn uniform [0,1] for bid fill decisions (shape: N).
    diag : DiagnosticsLog
        Diagnostics accumulator (modified in-place).

    Returns
    -------
    tuple
        (S_path, r_path, pa_path, pb_path, q_path, X_T, q_T)
    """
    S = p.s0
    q = p.q0
    X = p.X0
    half_spread = spread / 2.0

    # Symmetric: delta is constant, so intensity is constant
    delta_sym = half_spread
    lam_sym = execution_intensity(delta_sym, p.A, p.k)
    prob_sym = min(lam_sym * p.dt, 1.0)

    S_path = np.zeros(p.N + 1)
    r_path = np.zeros(p.N + 1)
    pa_path = np.zeros(p.N + 1)
    pb_path = np.zeros(p.N + 1)
    q_path = np.zeros(p.N + 1, dtype=int)

    S_path[0] = S
    r_path[0] = reservation_price(S, q, 0.0, p.gamma, p.sigma, p.T)
    q_path[0] = q

    for n in range(p.N):
        t = n * p.dt

        # Reservation price (for tracking only)
        r_t = reservation_price(S, q, t, p.gamma, p.sigma, p.T)

        # Symmetric strategy: center at mid-price
        pa = S + half_spread
        pb = S - half_spread

        ask_fill = ask_uniforms[n] < prob_sym
        bid_fill = bid_uniforms[n] < prob_sym

        if ask_fill and bid_fill:
            diag.simultaneous_fills += 1

        if ask_fill:
            X += pa
            q -= 1
        if bid_fill:
            X -= pb
            q += 1

        diag.total_steps += 1

        S += price_innovations[n] * p.sigma * math.sqrt(p.dt)

        S_path[n + 1] = S
        r_path[n + 1] = reservation_price(S, q, (n + 1) * p.dt, p.gamma, p.sigma, p.T)
        pa_path[n] = pa
        pb_path[n] = pb
        q_path[n + 1] = q

    pa_path[p.N] = S
    pb_path[p.N] = S

    return S_path, r_path, pa_path, pb_path, q_path, X, q


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

@dataclass
class MCResults:
    """Results from a Monte Carlo simulation run."""
    strategy: str
    gamma: float
    spread: float
    profits: np.ndarray
    inventories: np.ndarray
    mean_profit: float
    std_profit: float
    mean_inventory: float
    std_inventory: float
    sample_path: Optional[Dict] = None
    diagnostics: Optional[DiagnosticsLog] = None


def run_monte_carlo(p: SimParams) -> Dict[str, MCResults]:
    """Run Monte Carlo simulation for both strategies with constant spread.

    Parameters
    ----------
    p : SimParams
        Simulation parameters.

    Returns
    -------
    dict
        Results for 'inventory' and 'symmetric' strategies.
    """
    validate_sim_params(p)
    rng = np.random.default_rng(p.seed)

    # Pre-compute constant spread
    spread = constant_spread(p.gamma, p.k)

    # Pre-draw all random numbers
    price_innovations = rng.choice([-1.0, 1.0], size=(p.n_paths, p.N))

    if p.use_common_random:
        ask_uniforms = rng.uniform(0, 1, size=(p.n_paths, p.N))
        bid_uniforms = rng.uniform(0, 1, size=(p.n_paths, p.N))
        ask_uniforms_sym = ask_uniforms
        bid_uniforms_sym = bid_uniforms
    else:
        ask_uniforms = rng.uniform(0, 1, size=(p.n_paths, p.N))
        bid_uniforms = rng.uniform(0, 1, size=(p.n_paths, p.N))
        ask_uniforms_sym = rng.uniform(0, 1, size=(p.n_paths, p.N))
        bid_uniforms_sym = rng.uniform(0, 1, size=(p.n_paths, p.N))

    profits_inv = np.zeros(p.n_paths)
    inventories_inv = np.zeros(p.n_paths, dtype=int)
    profits_sym = np.zeros(p.n_paths)
    inventories_sym = np.zeros(p.n_paths, dtype=int)

    diag_inv = DiagnosticsLog()
    diag_sym = DiagnosticsLog()

    sample_path_inv = None

    for i in range(p.n_paths):
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_inventory(
            p, spread, price_innovations[i], ask_uniforms[i], bid_uniforms[i], diag_inv
        )
        profits_inv[i] = X_T + q_T * S_path[-1]
        inventories_inv[i] = q_T

        if i == 0:
            sample_path_inv = {
                "S": S_path, "r": r_path, "pa": pa_path, "pb": pb_path, "q": q_path
            }

        _, _, _, _, _, X_T_sym, q_T_sym = simulate_path_symmetric(
            p, spread, price_innovations[i], ask_uniforms_sym[i], bid_uniforms_sym[i], diag_sym
        )
        profits_sym[i] = X_T_sym + q_T_sym * S_path[-1]
        inventories_sym[i] = q_T_sym

    inv_results = MCResults(
        strategy="inventory",
        gamma=p.gamma,
        spread=spread,
        profits=profits_inv,
        inventories=inventories_inv,
        mean_profit=float(np.mean(profits_inv)),
        std_profit=float(np.std(profits_inv, ddof=1)),
        mean_inventory=float(np.mean(inventories_inv)),
        std_inventory=float(np.std(inventories_inv, ddof=1)),
        sample_path=sample_path_inv,
        diagnostics=diag_inv,
    )

    sym_results = MCResults(
        strategy="symmetric",
        gamma=p.gamma,
        spread=spread,
        profits=profits_sym,
        inventories=inventories_sym,
        mean_profit=float(np.mean(profits_sym)),
        std_profit=float(np.std(profits_sym, ddof=1)),
        mean_inventory=float(np.mean(inventories_sym)),
        std_inventory=float(np.std(inventories_sym, ddof=1)),
        diagnostics=diag_sym,
    )

    return {"inventory": inv_results, "symmetric": sym_results}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_sample_path(
    sample_path: Dict,
    gamma: float,
    spread: float,
    output_dir: str = "results",
) -> str:
    """Plot a sample path showing mid-price, reservation price, and quotes.

    Parameters
    ----------
    sample_path : dict
        Sample path data with keys S, r, pa, pb, q.
    gamma : float
        Risk-aversion parameter.
    spread : float
        Constant spread value.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    N = len(sample_path["S"]) - 1
    times = np.linspace(0, 1, N + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = axes[0]
    ax1.plot(times, sample_path["S"], "b-", linewidth=1.5, label="Mid-price $S_t$")
    ax1.plot(times, sample_path["r"], "r--", linewidth=1.5, label="Reservation price $r_t$")
    ax1.plot(times[:-1], sample_path["pa"][:-1], "g:", linewidth=1.0, label="Ask quote $p^a_t$")
    ax1.plot(times[:-1], sample_path["pb"][:-1], "m:", linewidth=1.0, label="Bid quote $p^b_t$")
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("Price")
    ax1.set_title(
        f"Exp 3 Sample Path — Inventory Strategy\n"
        f"$\\gamma={gamma}$, constant spread={spread:.4f}"
    )
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.step(times, sample_path["q"], "k-", linewidth=1.5, where="post")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("Inventory $q_t$")
    ax2.set_title(f"Inventory Path — $\\gamma={gamma}$")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"exp3_sample_path_gamma{gamma}.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    return fname


def plot_profit_histograms(
    inv_results: MCResults,
    sym_results: MCResults,
    output_dir: str = "results",
) -> str:
    """Plot histograms of terminal profit for both strategies.

    Parameters
    ----------
    inv_results : MCResults
        Inventory strategy results.
    sym_results : MCResults
        Symmetric strategy results.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    gamma = inv_results.gamma

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(inv_results.profits, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax1.axvline(inv_results.mean_profit, color="red", linestyle="--",
                label=f"Mean={inv_results.mean_profit:.2f}")
    ax1.set_xlabel("Terminal Profit $X_T + q_T S_T$")
    ax1.set_ylabel("Frequency")
    ax1.set_title(
        f"Inventory Strategy — $\\gamma={gamma}$\n"
        f"Std={inv_results.std_profit:.2f}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(sym_results.profits, bins=50, alpha=0.7, color="darkorange", edgecolor="white")
    ax2.axvline(sym_results.mean_profit, color="red", linestyle="--",
                label=f"Mean={sym_results.mean_profit:.2f}")
    ax2.set_xlabel("Terminal Profit $X_T + q_T S_T$")
    ax2.set_ylabel("Frequency")
    ax2.set_title(
        f"Symmetric Strategy — $\\gamma={gamma}$\n"
        f"Std={sym_results.std_profit:.2f}"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Exp 3: Terminal Profit Distribution — Constant Spread\n$\\gamma={gamma}$",
        fontsize=12
    )
    plt.tight_layout()
    fname = os.path.join(output_dir, f"exp3_profit_hist_gamma{gamma}.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    return fname


def plot_spread_comparison(
    gamma_values: List[float],
    k: float = 1.5,
    output_dir: str = "results",
) -> str:
    """Plot spread values vs gamma for both formula variants.

    Parameters
    ----------
    gamma_values : list of float
        Risk-aversion values.
    k : float
        Intensity decay parameter.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    gamma_range = np.linspace(0.001, 1.0, 200)
    const_spreads = [(2.0 / g) * math.log(1.0 + g / k) for g in gamma_range]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gamma_range, const_spreads, "b-", linewidth=2, label="$(2/\\gamma)\\ln(1+\\gamma/k)$")
    ax.axhline(2.0 / k, color="gray", linestyle="--", alpha=0.7,
               label=f"Symmetric limit $2/k = {2/k:.3f}$")

    for gamma in gamma_values:
        sp_val = constant_spread(gamma, k)
        ax.scatter([gamma], [sp_val], s=100, zorder=5)
        ax.annotate(f"$\\gamma={gamma}$\n{sp_val:.4f}", (gamma, sp_val),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("$\\gamma$ (risk aversion)")
    ax.set_ylabel("Spread")
    ax.set_title(f"Table-Faithful Constant Spread vs $\\gamma$ (k={k})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "exp3_spread_vs_gamma.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close()
    return fname


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

# Published target values from the paper (for comparison)
PAPER_TARGETS = {
    0.1: {
        "inventory": {"spread": 1.29, "mean_profit": 62.94, "std_profit": 5.89,
                      "mean_q": 0.10, "std_q": 2.80},
        "symmetric": {"spread": 1.29, "mean_profit": 67.21, "std_profit": 13.43,
                      "mean_q": -0.018, "std_q": 8.66},
    },
    # gamma=0.01 and 0.5 targets from paper narrative (approximate)
    0.01: {
        "inventory": {"spread": 1.33, "mean_profit": None, "std_profit": None,
                      "mean_q": None, "std_q": None},
        "symmetric": {"spread": 1.33, "mean_profit": None, "std_profit": None,
                      "mean_q": None, "std_q": None},
    },
    0.5: {
        "inventory": {"spread": 1.15, "mean_profit": None, "std_profit": None,
                      "mean_q": None, "std_q": None},
        "symmetric": {"spread": 1.15, "mean_profit": None, "std_profit": None,
                      "mean_q": None, "std_q": None},
    },
}


def run_experiment_3(
    gamma_values: List[float] = None,
    output_dir: str = "results",
    seed: int = 42,
) -> Dict[float, Dict[str, MCResults]]:
    """Run Experiment 3: Monte Carlo with table-faithful constant spread.

    Parameters
    ----------
    gamma_values : list of float
        Risk-aversion values (default [0.1, 0.01, 0.5]).
    output_dir : str
        Directory for output files.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Results indexed by gamma, then strategy.
    """
    if gamma_values is None:
        gamma_values = [0.1, 0.01, 0.5]

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    print("=" * 70)
    print("EXPERIMENT 3: Monte Carlo — Table-Faithful Constant Spread")
    print("=" * 70)
    print(f"Spread formula: (2/gamma)*ln(1 + gamma/k)  [constant, no time dependence]")
    print(f"Parameters: s0=100, T=1, sigma=2, dt=0.005, N=200, A=140, k=1.5")
    print(f"Paths: 1000 per gamma, Common random numbers: True")
    print(f"Random seed: {seed}")
    print()

    # Verify spread values match paper
    print("Spread verification (k=1.5):")
    for gamma in gamma_values:
        sp = constant_spread(gamma, 1.5)
        target = PAPER_TARGETS.get(gamma, {}).get("inventory", {}).get("spread", None)
        match = f"(paper: {target})" if target else ""
        print(f"  gamma={gamma}: spread={sp:.4f} {match}")
    print()

    for gamma in gamma_values:
        p = SimParams(gamma=gamma, seed=seed, use_common_random=True)
        results = run_monte_carlo(p)
        all_results[gamma] = results

        inv = results["inventory"]
        sym = results["symmetric"]

        print(f"gamma = {gamma}")
        print(f"  Spread: {inv.spread:.4f}")
        print(f"  Inventory  — mean profit: {inv.mean_profit:.2f}, "
              f"std profit: {inv.std_profit:.2f}, "
              f"mean q_T: {inv.mean_inventory:.3f}, "
              f"std q_T: {inv.std_inventory:.3f}")
        print(f"  Symmetric  — mean profit: {sym.mean_profit:.2f}, "
              f"std profit: {sym.std_profit:.2f}, "
              f"mean q_T: {sym.mean_inventory:.3f}, "
              f"std q_T: {sym.std_inventory:.3f}")

        # Compare with paper targets
        if gamma in PAPER_TARGETS:
            tgt_inv = PAPER_TARGETS[gamma]["inventory"]
            tgt_sym = PAPER_TARGETS[gamma]["symmetric"]
            if tgt_inv["mean_profit"] is not None:
                print(f"  Paper targets (inventory): "
                      f"profit={tgt_inv['mean_profit']}, std={tgt_inv['std_profit']}, "
                      f"q={tgt_inv['mean_q']}, std_q={tgt_inv['std_q']}")
                print(f"  Paper targets (symmetric): "
                      f"profit={tgt_sym['mean_profit']}, std={tgt_sym['std_profit']}, "
                      f"q={tgt_sym['mean_q']}, std_q={tgt_sym['std_q']}")

        print(f"  Variance reduction (profit): "
              f"{(1 - inv.std_profit/sym.std_profit)*100:.1f}%")
        print(f"  Variance reduction (inventory): "
              f"{(1 - inv.std_inventory/sym.std_inventory)*100:.1f}%")
        print()

        print(f"  {inv.diagnostics.report()}")
        print()

        # Plots
        if inv.sample_path is not None:
            plot_sample_path(inv.sample_path, gamma, inv.spread, output_dir)
        plot_profit_histograms(inv, sym, output_dir)

    # Spread comparison plot
    plot_spread_comparison(gamma_values, k=1.5, output_dir=output_dir)

    print("Reproducibility log:")
    print(f"  Seed: {seed}")
    print(f"  Common random numbers: True (same price/fill draws for both strategies)")
    print(f"  Mid-price update: S_{{t+dt}} = S_t ± sigma*sqrt(dt) with prob 0.5")
    print(f"  Simultaneous fills: allowed (independent Bernoulli draws)")
    print(f"  lambda*dt > 1: capped at 1, frequency logged")
    print(f"  Spread: constant (2/gamma)*ln(1+gamma/k), no time dependence")

    return all_results


if __name__ == "__main__":
    run_experiment_3(output_dir="results")
