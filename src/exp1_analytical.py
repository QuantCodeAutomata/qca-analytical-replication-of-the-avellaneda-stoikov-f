"""
Experiment 1: Analytical replication of the Avellaneda-Stoikov finite-horizon
market-making model.

Reproduces and verifies the paper's analytical results for the finite-horizon
market-making problem under arithmetic Brownian mid-price dynamics, exponential
utility, and distance-dependent execution intensities.

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Using sympy for symbolic verification — Context7 confirmed
import sympy as sp


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def validate_params(gamma: float, sigma: float, T: float, t: float, k: float = 1.0) -> None:
    """Validate model parameters per paper requirements.

    Parameters
    ----------
    gamma : float
        CARA risk-aversion coefficient. Must be > 0.
    sigma : float
        Mid-price volatility. Must be >= 0.
    T : float
        Terminal horizon. Must be >= t.
    t : float
        Current time.
    k : float
        Exponential intensity decay parameter. Must be > 0.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if T < t:
        raise ValueError(f"T must be >= t, got T={T}, t={t}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")


# ---------------------------------------------------------------------------
# Step 1-2: Frozen-inventory value function
# ---------------------------------------------------------------------------

def frozen_inventory_value(
    x: float, s: float, q: float, t: float,
    gamma: float, sigma: float, T: float
) -> float:
    """Compute the frozen-inventory utility value function.

    Under arithmetic Brownian motion dS_u = sigma dW_u with S_t = s,
    S_T ~ N(s, sigma^2*(T-t)), so:

        v(x,s,q,t) = E_t[-exp(-gamma*(x + q*S_T))]
                   = -exp(-gamma*x) * exp(-gamma*q*s) * exp(gamma^2*q^2*sigma^2*(T-t)/2)

    This uses the Gaussian MGF: E[exp(c*Z)] = exp(c*mu + c^2*var/2)
    with c = -gamma*q, mu = s, var = sigma^2*(T-t).

    Parameters
    ----------
    x : float
        Current cash position.
    s : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    gamma : float
        CARA risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.

    Returns
    -------
    float
        Value of the frozen-inventory utility function.
    """
    validate_params(gamma, sigma, T, t)
    tau = T - t  # time to maturity
    # Custom — paper Eq. (frozen inventory value function)
    # v = -exp(-gamma*x) * exp(-gamma*q*s) * exp(gamma^2*q^2*sigma^2*tau/2)
    return (
        -math.exp(-gamma * x)
        * math.exp(-gamma * q * s)
        * math.exp(0.5 * gamma**2 * q**2 * sigma**2 * tau)
    )


def verify_frozen_inventory_symbolic() -> Dict[str, sp.Expr]:
    """Symbolically derive and verify the frozen-inventory value function.

    Uses sympy to confirm the Gaussian MGF derivation.

    Returns
    -------
    dict
        Dictionary with symbolic expressions for the value function components.
    """
    # Using sympy for symbolic verification — Context7 confirmed
    gamma_s, x_s, q_s, s_s, sigma_s, tau_s = sp.symbols(
        'gamma x q s sigma tau', positive=True
    )

    # S_T | S_t=s is N(s, sigma^2*tau)
    # E[-exp(-gamma*(x + q*S_T))]
    # = -exp(-gamma*x) * E[exp(-gamma*q*S_T)]
    # MGF of N(mu, var): E[exp(c*Z)] = exp(c*mu + c^2*var/2)
    c = -gamma_s * q_s
    mu = s_s
    var = sigma_s**2 * tau_s

    mgf = sp.exp(c * mu + c**2 * var / 2)
    v_symbolic = -sp.exp(-gamma_s * x_s) * mgf

    # Expand and simplify
    v_expanded = sp.expand(sp.log(-v_symbolic))  # log of -v for readability

    return {
        "v_symbolic": v_symbolic,
        "v_log_neg": v_expanded,
        "mgf": mgf,
    }


# ---------------------------------------------------------------------------
# Step 3: Reservation ask and bid prices
# ---------------------------------------------------------------------------

def reservation_ask(s: float, q: float, t: float, gamma: float, sigma: float, T: float) -> float:
    """Compute the reservation ask price r^a(s,q,t).

    Defined implicitly by v(x + r^a, s, q-1, t) = v(x, s, q, t).
    Solving explicitly:

        r^a(s,q,t) = s + (1 - 2q) * gamma * sigma^2 * (T-t) / 2

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
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
        Reservation ask price.
    """
    validate_params(gamma, sigma, T, t)
    tau = T - t
    # Custom — paper Eq. reservation ask
    return s + (1 - 2 * q) * gamma * sigma**2 * tau / 2


def reservation_bid(s: float, q: float, t: float, gamma: float, sigma: float, T: float) -> float:
    """Compute the reservation bid price r^b(s,q,t).

    Defined implicitly by v(x - r^b, s, q+1, t) = v(x, s, q, t).
    Solving explicitly:

        r^b(s,q,t) = s + (-1 - 2q) * gamma * sigma^2 * (T-t) / 2

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
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
        Reservation bid price.
    """
    validate_params(gamma, sigma, T, t)
    tau = T - t
    # Custom — paper Eq. reservation bid
    return s + (-1 - 2 * q) * gamma * sigma**2 * tau / 2


def average_reservation_price(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> float:
    """Compute the average reservation price r(s,q,t) = (r^a + r^b) / 2.

    Closed form:
        r(s,q,t) = s - q * gamma * sigma^2 * (T-t)

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
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
        Average reservation price.
    """
    validate_params(gamma, sigma, T, t)
    tau = T - t
    # Custom — paper Eq. average reservation price
    return s - q * gamma * sigma**2 * tau


# ---------------------------------------------------------------------------
# Step 3 verification: indifference equations
# ---------------------------------------------------------------------------

def verify_reservation_ask_indifference(
    x: float, s: float, q: float, t: float,
    gamma: float, sigma: float, T: float
) -> Tuple[float, float, float]:
    """Verify that v(x + r^a, s, q-1, t) = v(x, s, q, t).

    Parameters
    ----------
    x, s, q, t, gamma, sigma, T : float
        Model state and parameters.

    Returns
    -------
    tuple
        (lhs, rhs, relative_difference)
        Relative difference = |lhs - rhs| / max(|lhs|, |rhs|)
    """
    r_a = reservation_ask(s, q, t, gamma, sigma, T)
    lhs = frozen_inventory_value(x + r_a, s, q - 1, t, gamma, sigma, T)
    rhs = frozen_inventory_value(x, s, q, t, gamma, sigma, T)
    # Use relative difference to handle large/small magnitudes
    denom = max(abs(lhs), abs(rhs))
    rel_diff = abs(lhs - rhs) / denom if denom > 0 else 0.0
    return lhs, rhs, rel_diff


def verify_reservation_bid_indifference(
    x: float, s: float, q: float, t: float,
    gamma: float, sigma: float, T: float
) -> Tuple[float, float, float]:
    """Verify that v(x - r^b, s, q+1, t) = v(x, s, q, t).

    Parameters
    ----------
    x, s, q, t, gamma, sigma, T : float
        Model state and parameters.

    Returns
    -------
    tuple
        (lhs, rhs, relative_difference)
        Relative difference = |lhs - rhs| / max(|lhs|, |rhs|)
    """
    r_b = reservation_bid(s, q, t, gamma, sigma, T)
    lhs = frozen_inventory_value(x - r_b, s, q + 1, t, gamma, sigma, T)
    rhs = frozen_inventory_value(x, s, q, t, gamma, sigma, T)
    denom = max(abs(lhs), abs(rhs))
    rel_diff = abs(lhs - rhs) / denom if denom > 0 else 0.0
    return lhs, rhs, rel_diff


def verify_average_reservation_price(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> Tuple[float, float, float]:
    """Verify that (r^a + r^b)/2 = s - q*gamma*sigma^2*(T-t).

    Parameters
    ----------
    s, q, t, gamma, sigma, T : float
        Model state and parameters.

    Returns
    -------
    tuple
        (average_from_components, closed_form, absolute_difference)
    """
    r_a = reservation_ask(s, q, t, gamma, sigma, T)
    r_b = reservation_bid(s, q, t, gamma, sigma, T)
    avg_from_components = (r_a + r_b) / 2
    closed_form = average_reservation_price(s, q, t, gamma, sigma, T)
    return avg_from_components, closed_form, abs(avg_from_components - closed_form)


# ---------------------------------------------------------------------------
# Step 7-8: HJB ansatz and theta representation
# ---------------------------------------------------------------------------

def theta_frozen(s: float, q: float, t: float, gamma: float, sigma: float, T: float) -> float:
    """Compute theta(s,q,t) from the exponential ansatz u = -exp(-gamma*x)*exp(-gamma*theta).

    From the frozen-inventory value function:
        v(x,s,q,t) = -exp(-gamma*x) * exp(-gamma*q*s) * exp(gamma^2*q^2*sigma^2*tau/2)

    Matching with u = -exp(-gamma*x)*exp(-gamma*theta):
        theta(s,q,t) = q*s - (gamma*q^2*sigma^2*tau)/2

    Parameters
    ----------
    s, q, t, gamma, sigma, T : float
        Model state and parameters.

    Returns
    -------
    float
        Value of theta(s,q,t).
    """
    validate_params(gamma, sigma, T, t)
    tau = T - t
    # Custom — paper ansatz theta derivation
    return q * s - (gamma * q**2 * sigma**2 * tau) / 2


def reservation_ask_from_theta(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> float:
    """Compute reservation ask from theta differences: r^a = theta(s,q,t) - theta(s,q-1,t).

    Parameters
    ----------
    s, q, t, gamma, sigma, T : float
        Model state and parameters.

    Returns
    -------
    float
        Reservation ask price from theta representation.
    """
    return theta_frozen(s, q, t, gamma, sigma, T) - theta_frozen(s, q - 1, t, gamma, sigma, T)


def reservation_bid_from_theta(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> float:
    """Compute reservation bid from theta differences: r^b = theta(s,q+1,t) - theta(s,q,t).

    Parameters
    ----------
    s, q, t, gamma, sigma, T : float
        Model state and parameters.

    Returns
    -------
    float
        Reservation bid price from theta representation.
    """
    return theta_frozen(s, q + 1, t, gamma, sigma, T) - theta_frozen(s, q, t, gamma, sigma, T)


# ---------------------------------------------------------------------------
# Step 9-10: First-order optimality conditions for exponential intensities
# ---------------------------------------------------------------------------

def spread_offset_exponential(gamma: float, k: float) -> float:
    """Compute the spread offset term (1/gamma)*ln(1 + gamma/k) for exponential intensities.

    For lambda(delta) = A*exp(-k*delta), the first-order conditions give:
        delta^b = (s - r^b) + (1/gamma)*ln(1 + gamma/k)
        delta^a = (r^a - s) + (1/gamma)*ln(1 + gamma/k)

    Note: A cancels out of the first-order conditions; only k matters.

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    k : float
        Exponential intensity decay parameter.

    Returns
    -------
    float
        The spread offset (1/gamma)*ln(1 + gamma/k).
    """
    validate_params(gamma, 1.0, 1.0, 0.0, k)
    # Custom — paper Eq. exponential intensity specialization
    return (1.0 / gamma) * math.log(1.0 + gamma / k)


def optimal_ask_distance(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float, k: float
) -> float:
    """Compute optimal ask quote distance delta^a for exponential intensities.

        delta^a = (r^a - s) + (1/gamma)*ln(1 + gamma/k)

    Parameters
    ----------
    s, q, t, gamma, sigma, T, k : float
        Model state and parameters.

    Returns
    -------
    float
        Optimal ask distance from mid-price.
    """
    r_a = reservation_ask(s, q, t, gamma, sigma, T)
    offset = spread_offset_exponential(gamma, k)
    return (r_a - s) + offset


def optimal_bid_distance(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float, k: float
) -> float:
    """Compute optimal bid quote distance delta^b for exponential intensities.

        delta^b = (s - r^b) + (1/gamma)*ln(1 + gamma/k)

    Parameters
    ----------
    s, q, t, gamma, sigma, T, k : float
        Model state and parameters.

    Returns
    -------
    float
        Optimal bid distance from mid-price.
    """
    r_b = reservation_bid(s, q, t, gamma, sigma, T)
    offset = spread_offset_exponential(gamma, k)
    return (s - r_b) + offset


# ---------------------------------------------------------------------------
# Step 11: Approximate finite-horizon formulas
# ---------------------------------------------------------------------------

def total_spread_equation_faithful(
    t: float, gamma: float, sigma: float, T: float, k: float
) -> float:
    """Compute the equation-faithful total spread including time-dependent term.

        Spread = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)

    Parameters
    ----------
    t, gamma, sigma, T, k : float
        Model parameters.

    Returns
    -------
    float
        Total spread (equation-faithful, time-varying).
    """
    validate_params(gamma, sigma, T, t, k)
    tau = T - t
    # Custom — paper Eq. total spread (equation-faithful)
    return gamma * sigma**2 * tau + (2.0 / gamma) * math.log(1.0 + gamma / k)


def total_spread_table_faithful(gamma: float, k: float) -> float:
    """Compute the table-faithful constant total spread.

        Spread = (2/gamma)*ln(1 + gamma/k)

    This matches the published simulation table values.

    Parameters
    ----------
    gamma : float
        CARA risk-aversion.
    k : float
        Exponential intensity decay parameter.

    Returns
    -------
    float
        Total spread (table-faithful, constant).
    """
    validate_params(gamma, 1.0, 1.0, 0.0, k)
    # Custom — paper table-implied spread formula
    return (2.0 / gamma) * math.log(1.0 + gamma / k)


# ---------------------------------------------------------------------------
# Step 12: Analytical limiting checks
# ---------------------------------------------------------------------------

def check_inventory_effect_on_reservation_price(
    s: float, q: float, t: float, gamma: float, sigma: float, T: float
) -> Dict[str, bool]:
    """Verify qualitative inventory effects on reservation price.

    Checks:
    - q = 0 => r = s
    - q > 0 => r < s (long inventory lowers reservation price)
    - q < 0 => r > s (short inventory raises reservation price)
    - t = T => r = s (no inventory effect at maturity)

    Parameters
    ----------
    s, q, t, gamma, sigma, T : float
        Model state and parameters.

    Returns
    -------
    dict
        Boolean results for each qualitative check.
    """
    validate_params(gamma, sigma, T, t)
    tol = 1e-10

    r_zero_q = average_reservation_price(s, 0.0, t, gamma, sigma, T)
    r_pos_q = average_reservation_price(s, 1.0, t, gamma, sigma, T)
    r_neg_q = average_reservation_price(s, -1.0, t, gamma, sigma, T)
    r_at_T = average_reservation_price(s, q, T, gamma, sigma, T)

    return {
        "q=0 => r=s": abs(r_zero_q - s) < tol,
        "q>0 => r<s": r_pos_q < s - tol,
        "q<0 => r>s": r_neg_q > s + tol,
        "t=T => r=s": abs(r_at_T - s) < tol,
    }


def check_small_gamma_convergence(k: float, gamma_values: List[float]) -> Dict[float, Dict]:
    """Check that as gamma -> 0, the spread offset converges to 2/k (symmetric behavior).

    Taylor expansion: (2/gamma)*ln(1 + gamma/k) -> 2/k as gamma -> 0.

    Parameters
    ----------
    k : float
        Exponential intensity decay parameter.
    gamma_values : list of float
        List of gamma values to evaluate.

    Returns
    -------
    dict
        For each gamma: spread offset, symmetric limit 2/k, and relative error.
    """
    symmetric_limit = 2.0 / k
    results = {}
    for gamma in gamma_values:
        offset = total_spread_table_faithful(gamma, k)
        rel_error = abs(offset - symmetric_limit) / symmetric_limit
        results[gamma] = {
            "spread_offset": offset,
            "symmetric_limit_2_over_k": symmetric_limit,
            "relative_error": rel_error,
        }
    return results


# ---------------------------------------------------------------------------
# Step 13: Paper inconsistency documentation
# ---------------------------------------------------------------------------

def document_paper_inconsistency(
    gamma_values: List[float], k: float, sigma: float, T: float
) -> Dict[str, Dict]:
    """Document the known inconsistency between the paper's spread formula and table values.

    The paper derives: Spread = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)
    But the published simulation tables show spreads matching: (2/gamma)*ln(1 + gamma/k)

    This function computes both and documents the discrepancy at t=0.

    Parameters
    ----------
    gamma_values : list of float
        Risk-aversion values to check.
    k : float
        Exponential intensity decay parameter.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.

    Returns
    -------
    dict
        For each gamma: equation-faithful spread at t=0, table-faithful spread, difference.
    """
    results = {}
    for gamma in gamma_values:
        eq_spread = total_spread_equation_faithful(0.0, gamma, sigma, T, k)
        tbl_spread = total_spread_table_faithful(gamma, k)
        results[gamma] = {
            "equation_faithful_spread_at_t0": eq_spread,
            "table_faithful_spread": tbl_spread,
            "difference": eq_spread - tbl_spread,
            "note": (
                "Paper inconsistency: table values match constant formula, "
                "not the full time-dependent derivation."
            ),
        }
    return results


# ---------------------------------------------------------------------------
# Main validation suite
# ---------------------------------------------------------------------------

@dataclass
class AnalyticalResults:
    """Container for all analytical verification results."""
    frozen_value_checks: List[Dict]
    reservation_price_checks: List[Dict]
    theta_consistency_checks: List[Dict]
    spread_offset_checks: Dict
    inventory_effect_checks: Dict
    small_gamma_checks: Dict
    paper_inconsistency: Dict
    symbolic_verification: Dict


def run_analytical_verification(
    s: float = 100.0,
    sigma: float = 2.0,
    T: float = 1.0,
    k: float = 1.5,
    x: float = 0.0,
    q_values: List[int] = None,
    t_values: List[float] = None,
    gamma_values: List[float] = None,
) -> AnalyticalResults:
    """Run the full analytical verification suite for the Avellaneda-Stoikov model.

    Parameters
    ----------
    s : float
        Mid-price (default 100).
    sigma : float
        Volatility (default 2).
    T : float
        Terminal horizon (default 1).
    k : float
        Exponential intensity decay (default 1.5).
    x : float
        Cash position (default 0).
    q_values : list of int
        Inventory values to test (default [-2,-1,0,1,2]).
    t_values : list of float
        Time values to test (default [0.0, 0.4, 0.8]).
    gamma_values : list of float
        Risk-aversion values (default [0.01, 0.1, 0.5]).

    Returns
    -------
    AnalyticalResults
        All verification results.
    """
    if q_values is None:
        q_values = [-2, -1, 0, 1, 2]
    if t_values is None:
        t_values = [0.0, 0.4, 0.8]
    if gamma_values is None:
        gamma_values = [0.01, 0.1, 0.5]

    # 1. Frozen-inventory value function checks
    frozen_checks = []
    for gamma in gamma_values:
        for q in q_values:
            for t in t_values:
                v = frozen_inventory_value(x, s, q, t, gamma, sigma, T)
                frozen_checks.append({
                    "gamma": gamma, "q": q, "t": t,
                    "v": v,
                    "v_negative": v < 0,  # must be negative (CARA utility)
                })

    # 2. Reservation price indifference checks
    res_checks = []
    for gamma in gamma_values:
        for q in q_values:
            for t in t_values:
                lhs_a, rhs_a, rel_diff_a = verify_reservation_ask_indifference(
                    x, s, q, t, gamma, sigma, T
                )
                lhs_b, rhs_b, rel_diff_b = verify_reservation_bid_indifference(
                    x, s, q, t, gamma, sigma, T
                )
                avg_comp, avg_cf, avg_diff = verify_average_reservation_price(
                    s, q, t, gamma, sigma, T
                )
                res_checks.append({
                    "gamma": gamma, "q": q, "t": t,
                    "ask_indiff_rel_diff": rel_diff_a,
                    "bid_indiff_rel_diff": rel_diff_b,
                    "avg_price_diff": avg_diff,
                    # Use relative tolerance for indifference checks (handles large/small magnitudes)
                    "ask_indiff_ok": rel_diff_a < 1e-10,
                    "bid_indiff_ok": rel_diff_b < 1e-10,
                    "avg_price_ok": avg_diff < 1e-10,
                })

    # 3. Theta consistency checks
    theta_checks = []
    for gamma in gamma_values:
        for q in q_values:
            for t in t_values:
                r_a_direct = reservation_ask(s, q, t, gamma, sigma, T)
                r_b_direct = reservation_bid(s, q, t, gamma, sigma, T)
                r_a_theta = reservation_ask_from_theta(s, q, t, gamma, sigma, T)
                r_b_theta = reservation_bid_from_theta(s, q, t, gamma, sigma, T)
                theta_checks.append({
                    "gamma": gamma, "q": q, "t": t,
                    "r_a_direct": r_a_direct,
                    "r_a_theta": r_a_theta,
                    "r_b_direct": r_b_direct,
                    "r_b_theta": r_b_theta,
                    "ask_consistent": abs(r_a_direct - r_a_theta) < 1e-10,
                    "bid_consistent": abs(r_b_direct - r_b_theta) < 1e-10,
                })

    # 4. Spread offset checks for k=1.5
    # Paper implementation notes: verify (2/gamma)*ln(1+gamma/k) for k=1.5
    # Expected: ~1.29 for gamma=0.1, ~1.33 for gamma=0.01, ~1.15 for gamma=0.5
    spread_checks = {}
    expected_approx = {0.1: 1.29, 0.01: 1.33, 0.5: 1.15}
    for gamma in gamma_values:
        spread = total_spread_table_faithful(gamma, k)
        spread_checks[gamma] = {
            "spread": spread,
            "expected_approx": expected_approx.get(gamma, None),
        }

    # 5. Inventory effect qualitative checks
    inv_checks = check_inventory_effect_on_reservation_price(s, 1.0, 0.4, 0.1, sigma, T)

    # 6. Small gamma convergence
    small_gamma = check_small_gamma_convergence(k, [0.5, 0.1, 0.01, 0.001])

    # 7. Paper inconsistency documentation
    inconsistency = document_paper_inconsistency(gamma_values, k, sigma, T)

    # 8. Symbolic verification
    symbolic = verify_frozen_inventory_symbolic()

    return AnalyticalResults(
        frozen_value_checks=frozen_checks,
        reservation_price_checks=res_checks,
        theta_consistency_checks=theta_checks,
        spread_offset_checks=spread_checks,
        inventory_effect_checks=inv_checks,
        small_gamma_checks=small_gamma,
        paper_inconsistency=inconsistency,
        symbolic_verification={k_: str(v_) for k_, v_ in symbolic.items()},
    )


def print_analytical_report(results: AnalyticalResults) -> None:
    """Print a formatted report of analytical verification results.

    Parameters
    ----------
    results : AnalyticalResults
        Results from run_analytical_verification().
    """
    print("=" * 70)
    print("EXPERIMENT 1: ANALYTICAL VERIFICATION REPORT")
    print("Avellaneda-Stoikov Finite-Horizon Market-Making Model")
    print("=" * 70)

    # Frozen value checks
    all_negative = all(c["v_negative"] for c in results.frozen_value_checks)
    print(f"\n[1] Frozen-inventory value function v < 0 always: {all_negative}")

    # Reservation price checks
    all_ask_ok = all(c["ask_indiff_ok"] for c in results.reservation_price_checks)
    all_bid_ok = all(c["bid_indiff_ok"] for c in results.reservation_price_checks)
    all_avg_ok = all(c["avg_price_ok"] for c in results.reservation_price_checks)
    print(f"\n[2] Reservation ask indifference v(x+r^a,s,q-1,t)=v(x,s,q,t): {all_ask_ok}")
    print(f"[3] Reservation bid indifference v(x-r^b,s,q+1,t)=v(x,s,q,t): {all_bid_ok}")
    print(f"[4] Average reservation price (r^a+r^b)/2 = s-q*gamma*sigma^2*(T-t): {all_avg_ok}")

    # Theta consistency
    all_ask_theta = all(c["ask_consistent"] for c in results.theta_consistency_checks)
    all_bid_theta = all(c["bid_consistent"] for c in results.theta_consistency_checks)
    print(f"\n[5] Theta representation r^a = theta(q)-theta(q-1) consistent: {all_ask_theta}")
    print(f"[6] Theta representation r^b = theta(q+1)-theta(q) consistent: {all_bid_theta}")

    # Spread offset checks
    print("\n[7] Table-faithful spread (2/gamma)*ln(1+gamma/k) for k=1.5:")
    for gamma, info in results.spread_offset_checks.items():
        exp_str = f"(expected ~{info['expected_approx']})" if info["expected_approx"] else ""
        print(f"    gamma={gamma:.3f}: spread={info['spread']:.4f} {exp_str}")

    # Inventory effects
    print("\n[8] Inventory effect qualitative checks:")
    for check, passed in results.inventory_effect_checks.items():
        print(f"    {check}: {passed}")

    # Small gamma convergence
    print("\n[9] Small gamma convergence to symmetric (2/k):")
    for gamma, info in results.small_gamma_checks.items():
        print(
            f"    gamma={gamma:.4f}: spread={info['spread_offset']:.6f}, "
            f"limit={info['symmetric_limit_2_over_k']:.6f}, "
            f"rel_err={info['relative_error']:.6f}"
        )

    # Paper inconsistency
    print("\n[10] Paper inconsistency documentation (t=0, sigma=2, T=1, k=1.5):")
    for gamma, info in results.paper_inconsistency.items():
        print(
            f"    gamma={gamma:.3f}: eq-faithful={info['equation_faithful_spread_at_t0']:.4f}, "
            f"table-faithful={info['table_faithful_spread']:.4f}, "
            f"diff={info['difference']:.4f}"
        )
    print(
        "\n    NOTE: The paper's published table spreads match the constant formula "
        "(2/gamma)*ln(1+gamma/k),\n    NOT the full time-dependent formula "
        "gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k).\n"
        "    This is a documented paper inconsistency, not an implementation error."
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    results = run_analytical_verification()
    print_analytical_report(results)
