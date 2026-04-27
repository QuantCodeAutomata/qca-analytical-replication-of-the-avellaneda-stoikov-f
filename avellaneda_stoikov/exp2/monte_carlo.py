"""
Experiment 2: Monte Carlo replication of finite-horizon inventory-aware versus
symmetric quoting strategies.

Replicates the paper's simulation comparing:
  - Inventory-based strategy (uses reservation price + approximate spread)
  - Symmetric benchmark (centers spread on mid-price)

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"

# Custom — Context7 found no library equivalent for AS market-making simulation (paper Section 4)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    """
    Parameters for the Avellaneda-Stoikov Monte Carlo simulation.

    Attributes
    ----------
    S0 : initial mid-price
    T : horizon
    sigma : mid-price volatility
    dt : time step
    q0 : initial inventory
    X0 : initial cash
    A : intensity scale parameter
    k : intensity decay parameter
    n_paths : number of Monte Carlo paths
    gamma : risk aversion coefficient
    seed : random seed for reproducibility
    allow_simultaneous_fills : whether bid and ask can fill in same step
    cap_lambda_dt : whether to cap lambda*dt at 1 if it exceeds 1
    use_common_random_numbers : use same random draws for both strategies
    """
    S0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    q0: int = 0
    X0: float = 0.0
    A: float = 140.0
    k: float = 1.5
    n_paths: int = 1000
    gamma: float = 0.1
    seed: int = 42
    allow_simultaneous_fills: bool = True
    cap_lambda_dt: bool = True
    use_common_random_numbers: bool = True


@dataclass
class PathDiagnostics:
    """Diagnostic counters accumulated over all paths."""
    negative_delta_a_count: int = 0
    negative_delta_b_count: int = 0
    lambda_dt_exceeded_1_count: int = 0
    simultaneous_fill_count: int = 0
    total_steps: int = 0


@dataclass
class SimResults:
    """
    Results from a Monte Carlo simulation run.

    Attributes
    ----------
    strategy : strategy name ('inventory' or 'symmetric')
    gamma : risk aversion used
    profits : array of terminal profits X_T + q_T * S_T
    final_inventories : array of terminal inventories q_T
    mean_profit : mean terminal profit
    std_profit : std of terminal profit
    mean_final_q : mean final inventory
    std_final_q : std of final inventory
    reported_spread : spread statistic (time-averaged)
    diagnostics : PathDiagnostics object
    sample_path : dict with a single representative path for plotting
    """
    strategy: str
    gamma: float
    profits: np.ndarray
    final_inventories: np.ndarray
    mean_profit: float
    std_profit: float
    mean_final_q: float
    std_final_q: float
    reported_spread: float
    diagnostics: PathDiagnostics
    sample_path: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Quoting rules
# ---------------------------------------------------------------------------

def compute_inventory_quotes(
    S: float,
    q: int,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> Tuple[float, float, float, float]:
    """
    Compute inventory-based quotes using the paper's approximate formulas.

    Reservation price: r = S - q * gamma * sigma^2 * (T-t)
    Spread: psi = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)
    Ask: p^a = r + psi/2  =>  delta^a = p^a - S
    Bid: p^b = r - psi/2  =>  delta^b = S - p^b

    Equivalently:
      delta^a = ((1 - 2q) * gamma * sigma^2 * (T-t)) / 2 + (1/gamma)*ln(1+gamma/k)
      delta^b = ((1 + 2q) * gamma * sigma^2 * (T-t)) / 2 + (1/gamma)*ln(1+gamma/k)

    Parameters
    ----------
    S : current mid-price
    q : current inventory
    t : current time
    T : horizon
    gamma : risk aversion
    sigma : volatility
    k : intensity decay

    Returns
    -------
    (delta_a, delta_b, p_a, p_b)
    """
    tau = T - t
    adj = np.log(1.0 + gamma / k) / gamma
    delta_a = ((1.0 - 2.0 * q) * gamma * sigma**2 * tau) / 2.0 + adj
    delta_b = ((1.0 + 2.0 * q) * gamma * sigma**2 * tau) / 2.0 + adj
    p_a = S + delta_a
    p_b = S - delta_b
    return delta_a, delta_b, p_a, p_b


def compute_symmetric_quotes(
    S: float,
    t: float,
    T: float,
    gamma: float,
    sigma: float,
    k: float,
) -> Tuple[float, float, float, float]:
    """
    Compute symmetric benchmark quotes centered on mid-price.

    Uses the same spread formula but ignores inventory:
      psi = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)
      delta^a = delta^b = psi / 2
      p^a = S + psi/2,  p^b = S - psi/2

    Parameters
    ----------
    S : current mid-price
    t : current time
    T : horizon
    gamma : risk aversion
    sigma : volatility
    k : intensity decay

    Returns
    -------
    (delta_a, delta_b, p_a, p_b)
    """
    tau = T - t
    psi = gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
    half_psi = psi / 2.0
    delta_a = half_psi
    delta_b = half_psi
    p_a = S + half_psi
    p_b = S - half_psi
    return delta_a, delta_b, p_a, p_b


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_paths(
    params: SimParams,
    strategy: str,
    random_state: Optional[np.random.RandomState] = None,
    pre_generated_randoms: Optional[Dict[str, np.ndarray]] = None,
) -> SimResults:
    """
    Simulate Monte Carlo paths for a given strategy.

    Parameters
    ----------
    params : SimParams object with all simulation parameters
    strategy : 'inventory' or 'symmetric'
    random_state : optional numpy RandomState for reproducibility
    pre_generated_randoms : optional dict with pre-generated random arrays
        (for common random numbers across strategies). Keys:
        'price_signs' : (n_paths, n_steps) array of +/-1 for mid-price steps
        'ask_fills'   : (n_paths, n_steps) array of U[0,1] for ask fill draws
        'bid_fills'   : (n_paths, n_steps) array of U[0,1] for bid fill draws

    Returns
    -------
    SimResults object
    """
    assert strategy in ("inventory", "symmetric"), (
        f"strategy must be 'inventory' or 'symmetric', got '{strategy}'"
    )

    if random_state is None:
        random_state = np.random.RandomState(params.seed)

    n_steps = int(round(params.T / params.dt))
    time_grid = np.linspace(0.0, params.T, n_steps + 1)

    # Generate or use pre-generated random numbers
    if pre_generated_randoms is not None:
        price_signs = pre_generated_randoms["price_signs"]
        ask_uniforms = pre_generated_randoms["ask_fills"]
        bid_uniforms = pre_generated_randoms["bid_fills"]
    else:
        # Mid-price: S_{n+1} = S_n +/- sigma*sqrt(dt) with equal probability
        price_signs = random_state.choice(
            [-1, 1], size=(params.n_paths, n_steps)
        ).astype(float)
        ask_uniforms = random_state.uniform(0, 1, size=(params.n_paths, n_steps))
        bid_uniforms = random_state.uniform(0, 1, size=(params.n_paths, n_steps))

    # Storage
    profits = np.zeros(params.n_paths)
    final_inventories = np.zeros(params.n_paths, dtype=int)
    spread_sum = 0.0
    spread_count = 0

    diag = PathDiagnostics()
    sample_path: Optional[Dict] = None

    sqrt_dt = np.sqrt(params.dt)

    for path_idx in range(params.n_paths):
        S = params.S0
        q = params.q0
        X = params.X0

        # For sample path recording (first path only)
        if path_idx == 0:
            sp_S = [S]
            sp_r = []
            sp_pa = []
            sp_pb = []
            sp_q = [q]
            sp_t = [0.0]

        for step_idx in range(n_steps):
            t_n = time_grid[step_idx]

            # Compute quotes
            if strategy == "inventory":
                delta_a, delta_b, p_a, p_b = compute_inventory_quotes(
                    S, q, t_n, params.T, params.gamma, params.sigma, params.k
                )
            else:
                delta_a, delta_b, p_a, p_b = compute_symmetric_quotes(
                    S, t_n, params.T, params.gamma, params.sigma, params.k
                )

            # Diagnostics: negative quote distances
            if delta_a < 0:
                diag.negative_delta_a_count += 1
            if delta_b < 0:
                diag.negative_delta_b_count += 1

            # Accumulate spread
            spread_sum += delta_a + delta_b
            spread_count += 1

            # Compute fill probabilities
            lambda_a = params.A * np.exp(-params.k * delta_a)
            lambda_b = params.A * np.exp(-params.k * delta_b)
            p_fill_a = lambda_a * params.dt
            p_fill_b = lambda_b * params.dt

            # Diagnostics: lambda*dt > 1
            if p_fill_a > 1.0 or p_fill_b > 1.0:
                diag.lambda_dt_exceeded_1_count += 1

            # Cap probabilities if requested
            if params.cap_lambda_dt:
                p_fill_a = min(p_fill_a, 1.0)
                p_fill_b = min(p_fill_b, 1.0)

            # Independent Bernoulli draws for ask and bid fills
            ask_fill = ask_uniforms[path_idx, step_idx] < p_fill_a
            bid_fill = bid_uniforms[path_idx, step_idx] < p_fill_b

            # Diagnostics: simultaneous fills
            if ask_fill and bid_fill:
                diag.simultaneous_fill_count += 1

            # Apply fills
            if ask_fill:
                q -= 1
                X += p_a
            if bid_fill:
                q += 1
                X -= p_b

            # Update mid-price: S_{n+1} = S_n +/- sigma*sqrt(dt)
            S += params.sigma * sqrt_dt * price_signs[path_idx, step_idx]

            # Record sample path
            if path_idx == 0:
                r_n = S - q * params.gamma * params.sigma**2 * (params.T - time_grid[step_idx + 1])
                sp_S.append(S)
                sp_r.append(r_n)
                sp_pa.append(p_a)
                sp_pb.append(p_b)
                sp_q.append(q)
                sp_t.append(time_grid[step_idx + 1])

        diag.total_steps += n_steps

        # Terminal marked-to-market profit
        profits[path_idx] = X + q * S
        final_inventories[path_idx] = q

    # Build sample path dict
    if params.n_paths > 0:
        sample_path = {
            "t": np.array(sp_t),
            "S": np.array(sp_S),
            "r": np.array(sp_r + [sp_S[-1]]),  # pad last
            "p_a": np.array(sp_pa + [sp_pa[-1]]),
            "p_b": np.array(sp_pb + [sp_pb[-1]]),
            "q": np.array(sp_q),
        }

    reported_spread = spread_sum / spread_count if spread_count > 0 else 0.0

    return SimResults(
        strategy=strategy,
        gamma=params.gamma,
        profits=profits,
        final_inventories=final_inventories,
        mean_profit=float(np.mean(profits)),
        std_profit=float(np.std(profits)),
        mean_final_q=float(np.mean(final_inventories)),
        std_final_q=float(np.std(final_inventories)),
        reported_spread=reported_spread,
        diagnostics=diag,
        sample_path=sample_path,
    )


def run_comparison(
    gamma: float,
    params_override: Optional[Dict] = None,
    seed: int = 42,
) -> Tuple[SimResults, SimResults]:
    """
    Run both inventory and symmetric strategies for a given gamma.

    Uses common random numbers for variance reduction in comparison.

    Parameters
    ----------
    gamma : risk aversion coefficient
    params_override : optional dict to override default SimParams fields
    seed : random seed

    Returns
    -------
    (inventory_results, symmetric_results)
    """
    base_params = {
        "S0": 100.0, "T": 1.0, "sigma": 2.0, "dt": 0.005,
        "q0": 0, "X0": 0.0, "A": 140.0, "k": 1.5,
        "n_paths": 1000, "gamma": gamma, "seed": seed,
        "allow_simultaneous_fills": True, "cap_lambda_dt": True,
        "use_common_random_numbers": True,
    }
    if params_override:
        base_params.update(params_override)

    params = SimParams(**base_params)
    n_steps = int(round(params.T / params.dt))

    # Generate common random numbers
    rng = np.random.RandomState(seed)
    price_signs = rng.choice([-1, 1], size=(params.n_paths, n_steps)).astype(float)
    ask_uniforms = rng.uniform(0, 1, size=(params.n_paths, n_steps))
    bid_uniforms = rng.uniform(0, 1, size=(params.n_paths, n_steps))

    pre_gen = {
        "price_signs": price_signs,
        "ask_fills": ask_uniforms,
        "bid_fills": bid_uniforms,
    }

    inv_results = simulate_paths(params, "inventory", pre_generated_randoms=pre_gen)
    sym_results = simulate_paths(params, "symmetric", pre_generated_randoms=pre_gen)

    return inv_results, sym_results


def run_all_gammas(
    gammas: Tuple[float, ...] = (0.01, 0.1, 0.5),
    n_paths: int = 1000,
    seed: int = 42,
) -> Dict[float, Tuple[SimResults, SimResults]]:
    """
    Run the full comparison experiment for all gamma values.

    Parameters
    ----------
    gammas : tuple of risk aversion values
    n_paths : number of Monte Carlo paths
    seed : random seed

    Returns
    -------
    dict mapping gamma -> (inventory_results, symmetric_results)
    """
    all_results = {}
    for g in gammas:
        print(f"  Running gamma = {g} ...")
        inv_res, sym_res = run_comparison(
            gamma=g,
            params_override={"n_paths": n_paths},
            seed=seed,
        )
        all_results[g] = (inv_res, sym_res)
    return all_results


def build_summary_table(
    all_results: Dict[float, Tuple[SimResults, SimResults]],
) -> pd.DataFrame:
    """
    Build a summary table matching the paper's format.

    Columns: gamma, Strategy, Spread, Profit, std(Profit), Final q, std(Final q)

    Parameters
    ----------
    all_results : dict from run_all_gammas

    Returns
    -------
    pandas DataFrame
    """
    rows = []
    for g, (inv_res, sym_res) in all_results.items():
        for res in (inv_res, sym_res):
            rows.append({
                "gamma": g,
                "Strategy": res.strategy.capitalize(),
                "Spread": round(res.reported_spread, 4),
                "Profit": round(res.mean_profit, 2),
                "std(Profit)": round(res.std_profit, 2),
                "Final q": round(res.mean_final_q, 3),
                "std(Final q)": round(res.std_final_q, 2),
            })
    return pd.DataFrame(rows)


def build_diagnostics_table(
    all_results: Dict[float, Tuple[SimResults, SimResults]],
) -> pd.DataFrame:
    """
    Build a diagnostics table summarizing simulation ambiguity flags.

    Parameters
    ----------
    all_results : dict from run_all_gammas

    Returns
    -------
    pandas DataFrame
    """
    rows = []
    for g, (inv_res, sym_res) in all_results.items():
        for res in (inv_res, sym_res):
            d = res.diagnostics
            rows.append({
                "gamma": g,
                "Strategy": res.strategy.capitalize(),
                "neg_delta_a": d.negative_delta_a_count,
                "neg_delta_b": d.negative_delta_b_count,
                "lambda_dt>1": d.lambda_dt_exceeded_1_count,
                "simultaneous_fills": d.simultaneous_fill_count,
                "total_steps": d.total_steps,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ambiguity documentation
# ---------------------------------------------------------------------------

AMBIGUITY_LOG = """
SIMULATION AMBIGUITIES FROM THE PAPER (Avellaneda & Stoikov 2008)
=================================================================

(a) SPREAD REPORTING AMBIGUITY:
    The paper's table 'Spread' column appears to match only the constant
    component (2/gamma)*ln(1+gamma/k) rather than the full time-varying
    formula psi(t) = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k).
    PRIMARY MODE: uses full formula (time-averaged over the path).
    DIAGNOSTIC MODE: uses only constant component.
    This is flagged as a reporting inconsistency; the analytical derivation
    is not altered.

(b) SIMULTANEOUS FILLS:
    The paper does not specify whether bid and ask can fill in the same dt.
    PRIMARY MODE: allows simultaneous fills (independent Bernoulli draws).

(c) LAMBDA*DT > 1:
    For large A and small delta, lambda*dt may exceed 1.
    PRIMARY MODE: caps fill probability at 1.0.
    Occurrences are counted in diagnostics.

(d) NEGATIVE QUOTE DISTANCES:
    For large |q| near maturity, delta^a or delta^b may become negative.
    PRIMARY MODE: leaves them unchanged (no floor).
    Occurrences are counted in diagnostics.

(e) COMMON RANDOM NUMBERS:
    PRIMARY MODE: uses common random numbers across strategies for
    variance reduction in the comparison.

(f) MID-PRICE DISCRETIZATION:
    The paper states S_{n+1} = S_n +/- sigma*sqrt(dt) with equal probability.
    PRIMARY MODE: uses this exact binary step (not Gaussian increment).
"""
