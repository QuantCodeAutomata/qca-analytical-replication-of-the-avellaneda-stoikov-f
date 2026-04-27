"""
Experiment 1: Analytical replication of the Avellaneda-Stoikov finite-horizon
market-making model.

Replicates:
  - Frozen-inventory utility-indifference reservation prices
  - HJB transformation under exponential utility ansatz
  - First-order optimal quote conditions (generic and exponential intensities)
  - Approximate closed-form finite-horizon quoting formulas

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"

# Custom — Context7 found no library equivalent for AS market-making analytics (paper Eq. 2-20)
# Using sympy for symbolic algebra — Context7 confirmed
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Symbolic variables
# ---------------------------------------------------------------------------
s, x, q, t, T = sp.symbols("s x q t T", real=True)
gamma, sigma = sp.symbols("gamma sigma", positive=True)
A, k = sp.symbols("A k", positive=True)
delta, delta_a, delta_b = sp.symbols("delta delta_a delta_b", real=True)
omega = sp.symbols("omega", positive=True)


# ---------------------------------------------------------------------------
# Step 1-3: Frozen-inventory value function
# ---------------------------------------------------------------------------

def frozen_value_function(
    x_val: sp.Expr,
    s_val: sp.Expr,
    q_val: sp.Expr,
    t_val: sp.Expr,
) -> sp.Expr:
    """
    Compute the frozen-inventory value function under exponential utility.

    Under dS_u = sigma dW_u (arithmetic Brownian, zero drift), with
    S_T = s + sigma*(W_T - W_t), the terminal wealth is x + q*S_T.

    v(x,s,q,t) = E_t[-exp(-gamma*(x + q*S_T))]
               = -exp(-gamma*x) * exp(-gamma*q*s) * exp(gamma^2*q^2*sigma^2*(T-t)/2)

    Parameters
    ----------
    x_val : sympy expression for cash
    s_val : sympy expression for mid-price
    q_val : sympy expression for inventory
    t_val : sympy expression for current time

    Returns
    -------
    sympy expression for v(x,s,q,t)
    """
    # E[-exp(-gamma*(x + q*(s + sigma*Z*sqrt(T-t))))] where Z ~ N(0,1)
    # = -exp(-gamma*x) * exp(-gamma*q*s) * E[exp(-gamma*q*sigma*sqrt(T-t)*Z)]
    # MGF of N(0,1): E[exp(c*Z)] = exp(c^2/2)
    # c = -gamma*q*sigma*sqrt(T-t)  =>  c^2/2 = gamma^2*q^2*sigma^2*(T-t)/2
    tau = T - t_val
    v = (
        -sp.exp(-gamma * x_val)
        * sp.exp(-gamma * q_val * s_val)
        * sp.exp(gamma**2 * q_val**2 * sigma**2 * tau / 2)
    )
    return v


# ---------------------------------------------------------------------------
# Steps 4-5: Reservation bid and ask via utility indifference
# ---------------------------------------------------------------------------

def reservation_bid(
    s_val: sp.Expr,
    q_val: sp.Expr,
    t_val: sp.Expr,
) -> sp.Expr:
    """
    Compute the reservation bid price via utility indifference.

    Solves v(x - r^b, s, q+1, t) = v(x, s, q, t) for r^b.

    Result: r^b(s,q,t) = s + ((-1 - 2q) * gamma * sigma^2 * (T-t)) / 2

    Parameters
    ----------
    s_val : mid-price (sympy expression)
    q_val : inventory (sympy expression)
    t_val : current time (sympy expression)

    Returns
    -------
    sympy expression for r^b
    """
    tau = T - t_val
    # Paper formula (derived from indifference equation)
    r_b = s_val + ((-1 - 2 * q_val) * gamma * sigma**2 * tau) / 2
    return r_b


def reservation_ask(
    s_val: sp.Expr,
    q_val: sp.Expr,
    t_val: sp.Expr,
) -> sp.Expr:
    """
    Compute the reservation ask price via utility indifference.

    Solves v(x + r^a, s, q-1, t) = v(x, s, q, t) for r^a.

    Result: r^a(s,q,t) = s + ((1 - 2q) * gamma * sigma^2 * (T-t)) / 2

    Parameters
    ----------
    s_val : mid-price (sympy expression)
    q_val : inventory (sympy expression)
    t_val : current time (sympy expression)

    Returns
    -------
    sympy expression for r^a
    """
    tau = T - t_val
    # Paper formula (derived from indifference equation)
    r_a = s_val + ((1 - 2 * q_val) * gamma * sigma**2 * tau) / 2
    return r_a


def average_reservation_price(
    s_val: sp.Expr,
    q_val: sp.Expr,
    t_val: sp.Expr,
) -> sp.Expr:
    """
    Compute the average reservation price r = (r^a + r^b) / 2.

    Result: r(s,q,t) = s - q * gamma * sigma^2 * (T-t)

    Parameters
    ----------
    s_val : mid-price
    q_val : inventory
    t_val : current time

    Returns
    -------
    sympy expression for r
    """
    r_b = reservation_bid(s_val, q_val, t_val)
    r_a = reservation_ask(s_val, q_val, t_val)
    return sp.simplify((r_a + r_b) / 2)


# ---------------------------------------------------------------------------
# Step 10: HJB ansatz verification
# ---------------------------------------------------------------------------

def hjb_ansatz_theta_pde_description() -> str:
    """
    Describe the HJB PDE reduction under the exponential utility ansatz.

    The ansatz u(s,x,q,t) = -exp(-gamma*x) * exp(-gamma*theta(s,q,t))
    transforms the HJB equation into a nonlinear PDE for theta(s,q,t).

    Returns
    -------
    str describing the reduced PDE structure
    """
    description = (
        "HJB Ansatz: u(s,x,q,t) = -exp(-gamma*x) * exp(-gamma*theta(s,q,t))\n\n"
        "Under this substitution, the HJB equation reduces to a PDE for theta:\n\n"
        "  d_t(theta) + (sigma^2/2) * d_ss(theta)\n"
        "  + max_{delta^a} lambda^a(delta^a) * [exp(-gamma*(theta(s,q,t) - theta(s,q-1,t) - delta^a)) - 1]\n"
        "  + max_{delta^b} lambda^b(delta^b) * [exp(-gamma*(theta(s,q+1,t) - theta(s,q,t) + delta^b)) - 1]\n"
        "  = 0\n\n"
        "Terminal condition: theta(s,q,T) = q*s\n\n"
        "The key insight is that the cash variable x decouples completely,\n"
        "reducing the problem from 4D to 3D (s, q, t)."
    )
    return description


# ---------------------------------------------------------------------------
# Step 11: Reservation prices in terms of theta
# ---------------------------------------------------------------------------

def reservation_prices_from_theta() -> Dict[str, str]:
    """
    Express reservation prices in terms of the theta function.

    From the ansatz, the utility-indifference conditions yield:
      r^b(s,q,t) = theta(s,q+1,t) - theta(s,q,t)
      r^a(s,q,t) = theta(s,q,t) - theta(s,q-1,t)

    Returns
    -------
    dict with keys 'r_b' and 'r_a' containing string descriptions
    """
    return {
        "r_b": "r^b(s,q,t) = theta(s,q+1,t) - theta(s,q,t)",
        "r_a": "r^a(s,q,t) = theta(s,q,t) - theta(s,q-1,t)",
        "interpretation": (
            "These are finite differences of theta in the inventory dimension. "
            "The reservation price is the marginal value of inventory in units of cash."
        ),
    }


# ---------------------------------------------------------------------------
# Step 12: Generic first-order conditions
# ---------------------------------------------------------------------------

def generic_foc_residual_ask(
    delta_a_val: sp.Expr,
    r_a_val: sp.Expr,
    lambda_a: sp.Expr,
    lambda_a_prime: sp.Expr,
) -> sp.Expr:
    """
    Compute the residual of the generic first-order condition for the ask quote.

    FOC: r^a - s = delta^a - (1/gamma) * ln(1 - gamma * lambda^a / lambda^{a'})

    Equivalently, the residual is:
      FOC_residual = (r^a - s) - delta^a + (1/gamma) * ln(1 - gamma * lambda^a / lambda^{a'})

    Parameters
    ----------
    delta_a_val : ask quote distance (sympy expression)
    r_a_val : reservation ask price (sympy expression)
    lambda_a : arrival intensity at delta^a
    lambda_a_prime : derivative of lambda^a w.r.t. delta^a

    Returns
    -------
    sympy expression for the FOC residual (should be 0 at optimum)
    """
    # Note: lambda_a_prime < 0 for decreasing intensity, so ratio is positive
    foc = (r_a_val - s) - delta_a_val + (1 / gamma) * sp.log(
        1 - gamma * lambda_a / lambda_a_prime
    )
    return foc


def generic_foc_residual_bid(
    delta_b_val: sp.Expr,
    r_b_val: sp.Expr,
    lambda_b: sp.Expr,
    lambda_b_prime: sp.Expr,
) -> sp.Expr:
    """
    Compute the residual of the generic first-order condition for the bid quote.

    FOC: s - r^b = delta^b - (1/gamma) * ln(1 - gamma * lambda^b / lambda^{b'})

    Parameters
    ----------
    delta_b_val : bid quote distance (sympy expression)
    r_b_val : reservation bid price (sympy expression)
    lambda_b : arrival intensity at delta^b
    lambda_b_prime : derivative of lambda^b w.r.t. delta^b

    Returns
    -------
    sympy expression for the FOC residual (should be 0 at optimum)
    """
    foc = (s - r_b_val) - delta_b_val + (1 / gamma) * sp.log(
        1 - gamma * lambda_b / lambda_b_prime
    )
    return foc


# ---------------------------------------------------------------------------
# Steps 13: Exponential intensity specialization
# ---------------------------------------------------------------------------

def exponential_intensity(delta_val: sp.Expr) -> sp.Expr:
    """
    Exponential arrival intensity: lambda(delta) = A * exp(-k * delta).

    Parameters
    ----------
    delta_val : quote distance

    Returns
    -------
    sympy expression for lambda(delta)
    """
    return A * sp.exp(-k * delta_val)


def exponential_intensity_derivative(delta_val: sp.Expr) -> sp.Expr:
    """
    Derivative of exponential intensity: lambda'(delta) = -k * A * exp(-k * delta).

    Parameters
    ----------
    delta_val : quote distance

    Returns
    -------
    sympy expression for lambda'(delta)
    """
    return -k * A * sp.exp(-k * delta_val)


def log_adjustment() -> sp.Expr:
    """
    Compute the liquidity adjustment term for exponential intensities.

    For lambda(delta) = A*exp(-k*delta), lambda'(delta) = -k*lambda(delta).
    Substituting into the generic FOC:
      (1/gamma) * ln(1 - gamma * lambda / lambda') = (1/gamma) * ln(1 + gamma/k)

    Returns
    -------
    sympy expression: (1/gamma) * ln(1 + gamma/k)
    """
    return (1 / gamma) * sp.log(1 + gamma / k)


def optimal_ask_distance(
    s_val: sp.Expr,
    q_val: sp.Expr,
    t_val: sp.Expr,
) -> sp.Expr:
    """
    Compute the optimal ask quote distance under exponential intensities.

    delta^a = r^a - s + (1/gamma) * ln(1 + gamma/k)
            = ((1 - 2q) * gamma * sigma^2 * (T-t)) / 2 + (1/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    s_val : mid-price
    q_val : inventory
    t_val : current time

    Returns
    -------
    sympy expression for delta^a
    """
    r_a = reservation_ask(s_val, q_val, t_val)
    adj = log_adjustment()
    return sp.simplify(r_a - s_val + adj)


def optimal_bid_distance(
    s_val: sp.Expr,
    q_val: sp.Expr,
    t_val: sp.Expr,
) -> sp.Expr:
    """
    Compute the optimal bid quote distance under exponential intensities.

    delta^b = s - r^b + (1/gamma) * ln(1 + gamma/k)
            = ((1 + 2q) * gamma * sigma^2 * (T-t)) / 2 + (1/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    s_val : mid-price
    q_val : inventory
    t_val : current time

    Returns
    -------
    sympy expression for delta^b
    """
    r_b = reservation_bid(s_val, q_val, t_val)
    adj = log_adjustment()
    return sp.simplify(s_val - r_b + adj)


def total_spread(
    s_val: sp.Expr,
    q_val: sp.Expr,
    t_val: sp.Expr,
) -> sp.Expr:
    """
    Compute the total quoted spread delta^a + delta^b.

    Result: gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    s_val : mid-price
    q_val : inventory
    t_val : current time

    Returns
    -------
    sympy expression for total spread
    """
    da = optimal_ask_distance(s_val, q_val, t_val)
    db = optimal_bid_distance(s_val, q_val, t_val)
    return sp.simplify(da + db)


# ---------------------------------------------------------------------------
# Steps 14-17: Approximate finite-horizon expansion
# ---------------------------------------------------------------------------

def approximate_spread_formula() -> sp.Expr:
    """
    Return the paper's approximate total spread formula.

    psi(t) = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

    Returns
    -------
    sympy expression for the approximate spread
    """
    tau = T - t
    return gamma * sigma**2 * tau + (2 / gamma) * sp.log(1 + gamma / k)


def approximate_reservation_price(
    s_val: sp.Expr,
    q_val: sp.Expr,
    t_val: sp.Expr,
) -> sp.Expr:
    """
    Return the paper's approximate reservation price formula.

    r(s,q,t) = s - q * gamma * sigma^2 * (T-t)

    Parameters
    ----------
    s_val : mid-price
    q_val : inventory
    t_val : current time

    Returns
    -------
    sympy expression for approximate reservation price
    """
    tau = T - t_val
    return s_val - q_val * gamma * sigma**2 * tau


# ---------------------------------------------------------------------------
# Numerical evaluation helpers
# ---------------------------------------------------------------------------

def numerical_reservation_prices(
    s0: float,
    q0: float,
    t0: float,
    T0: float,
    gamma0: float,
    sigma0: float,
) -> Dict[str, float]:
    """
    Evaluate reservation prices numerically for given parameter values.

    Parameters
    ----------
    s0 : current mid-price
    q0 : current inventory
    t0 : current time
    T0 : horizon
    gamma0 : risk aversion
    sigma0 : volatility

    Returns
    -------
    dict with keys 'r_b', 'r_a', 'r_avg'
    """
    tau = T0 - t0
    r_b_val = s0 + ((-1 - 2 * q0) * gamma0 * sigma0**2 * tau) / 2
    r_a_val = s0 + ((1 - 2 * q0) * gamma0 * sigma0**2 * tau) / 2
    r_avg = (r_a_val + r_b_val) / 2
    return {"r_b": r_b_val, "r_a": r_a_val, "r_avg": r_avg}


def numerical_quote_distances(
    s0: float,
    q0: float,
    t0: float,
    T0: float,
    gamma0: float,
    sigma0: float,
    k0: float,
) -> Dict[str, float]:
    """
    Evaluate optimal quote distances numerically.

    Parameters
    ----------
    s0 : current mid-price
    q0 : current inventory
    t0 : current time
    T0 : horizon
    gamma0 : risk aversion
    sigma0 : volatility
    k0 : intensity decay parameter

    Returns
    -------
    dict with keys 'delta_a', 'delta_b', 'spread'
    """
    tau = T0 - t0
    adj = np.log(1 + gamma0 / k0) / gamma0
    delta_a = ((1 - 2 * q0) * gamma0 * sigma0**2 * tau) / 2 + adj
    delta_b = ((1 + 2 * q0) * gamma0 * sigma0**2 * tau) / 2 + adj
    spread = delta_a + delta_b
    return {"delta_a": delta_a, "delta_b": delta_b, "spread": spread}


def numerical_spread_formula(
    t0: float,
    T0: float,
    gamma0: float,
    sigma0: float,
    k0: float,
) -> float:
    """
    Evaluate the approximate spread formula numerically.

    psi(t) = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    t0 : current time
    T0 : horizon
    gamma0 : risk aversion
    sigma0 : volatility
    k0 : intensity decay parameter

    Returns
    -------
    float spread value
    """
    tau = T0 - t0
    return gamma0 * sigma0**2 * tau + (2 / gamma0) * np.log(1 + gamma0 / k0)


# ---------------------------------------------------------------------------
# Symbolic verification functions
# ---------------------------------------------------------------------------

def verify_utility_indifference_bid() -> bool:
    """
    Verify that the reservation bid satisfies the utility-indifference equation.

    Checks: v(x - r^b, s, q+1, t) == v(x, s, q, t)

    Returns
    -------
    bool: True if the identity holds symbolically
    """
    r_b = reservation_bid(s, q, t)
    lhs = frozen_value_function(x - r_b, s, q + 1, t)
    rhs = frozen_value_function(x, s, q, t)
    diff = sp.simplify(lhs - rhs)
    return diff == 0


def verify_utility_indifference_ask() -> bool:
    """
    Verify that the reservation ask satisfies the utility-indifference equation.

    Checks: v(x + r^a, s, q-1, t) == v(x, s, q, t)

    Returns
    -------
    bool: True if the identity holds symbolically
    """
    r_a = reservation_ask(s, q, t)
    lhs = frozen_value_function(x + r_a, s, q - 1, t)
    rhs = frozen_value_function(x, s, q, t)
    diff = sp.simplify(lhs - rhs)
    return diff == 0


def verify_average_reservation_price() -> bool:
    """
    Verify that the average reservation price equals s - q*gamma*sigma^2*(T-t).

    Returns
    -------
    bool: True if the identity holds symbolically
    """
    r_avg = average_reservation_price(s, q, t)
    expected = s - q * gamma * sigma**2 * (T - t)
    diff = sp.simplify(r_avg - expected)
    return diff == 0


def verify_exponential_foc_log_adjustment() -> bool:
    """
    Verify that substituting exponential intensities into the generic FOC
    yields the log adjustment (1/gamma)*ln(1 + gamma/k).

    For lambda(delta) = A*exp(-k*delta), lambda'(delta) = -k*lambda(delta).
    The adjustment term: (1/gamma)*ln(1 - gamma*lambda/lambda')
                       = (1/gamma)*ln(1 - gamma*(-k*lambda)/(-k*lambda))
    Wait — correct derivation:
      lambda'(delta) = -k*lambda(delta)
      1 - gamma*lambda/lambda' = 1 - gamma*lambda/(-k*lambda) = 1 + gamma/k

    Returns
    -------
    bool: True if the identity holds symbolically
    """
    lam = exponential_intensity(delta)
    lam_prime = exponential_intensity_derivative(delta)
    # The adjustment term
    adj_computed = (1 / gamma) * sp.log(1 - gamma * lam / lam_prime)
    adj_expected = (1 / gamma) * sp.log(1 + gamma / k)
    diff = sp.simplify(adj_computed - adj_expected)
    return diff == 0


def verify_spread_sum() -> bool:
    """
    Verify that delta^a + delta^b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k).

    Returns
    -------
    bool: True if the identity holds symbolically
    """
    spread = total_spread(s, q, t)
    expected = gamma * sigma**2 * (T - t) + (2 / gamma) * sp.log(1 + gamma / k)
    diff = sp.simplify(spread - expected)
    return diff == 0


# ---------------------------------------------------------------------------
# Qualitative implications
# ---------------------------------------------------------------------------

def qualitative_implications(
    s0: float = 100.0,
    T0: float = 1.0,
    sigma0: float = 2.0,
    k0: float = 1.5,
    gammas: Tuple[float, ...] = (0.01, 0.1, 0.5),
    inventories: Tuple[int, ...] = (-2, -1, 0, 1, 2),
    t0: float = 0.0,
) -> Dict:
    """
    Compute numerical spot-checks for qualitative implications of the model.

    Verifies:
    - q > 0 => r < s (reservation price below mid)
    - q < 0 => r > s (reservation price above mid)
    - t -> T => r -> s
    - gamma -> 0 => inventory skew disappears

    Parameters
    ----------
    s0 : initial mid-price
    T0 : horizon
    sigma0 : volatility
    k0 : intensity decay
    gammas : tuple of risk aversion values
    inventories : tuple of inventory levels
    t0 : current time

    Returns
    -------
    dict with numerical results
    """
    results = {}
    for g in gammas:
        results[f"gamma={g}"] = {}
        for q0 in inventories:
            prices = numerical_reservation_prices(s0, q0, t0, T0, g, sigma0)
            dists = numerical_quote_distances(s0, q0, t0, T0, g, sigma0, k0)
            results[f"gamma={g}"][f"q={q0}"] = {
                "r_b": prices["r_b"],
                "r_a": prices["r_a"],
                "r_avg": prices["r_avg"],
                "r_avg_below_s": prices["r_avg"] < s0 if q0 > 0 else None,
                "r_avg_above_s": prices["r_avg"] > s0 if q0 < 0 else None,
                "delta_a": dists["delta_a"],
                "delta_b": dists["delta_b"],
                "spread": dists["spread"],
            }
        # t -> T limit
        prices_at_T = numerical_reservation_prices(s0, 0, T0 - 1e-10, T0, g, sigma0)
        results[f"gamma={g}"]["t_near_T"] = {
            "r_avg_near_T": prices_at_T["r_avg"],
            "converges_to_s": abs(prices_at_T["r_avg"] - s0) < 1e-6,
        }

    # gamma -> 0 limit: inventory skew should vanish
    g_small = 1e-8
    prices_small_g = numerical_reservation_prices(s0, 5, t0, T0, g_small, sigma0)
    results["gamma_near_0"] = {
        "r_avg_q5": prices_small_g["r_avg"],
        "skew": prices_small_g["r_avg"] - s0,
        "skew_near_zero": abs(prices_small_g["r_avg"] - s0) < 1e-3,
    }
    return results


def run_symbolic_verification() -> Dict[str, bool]:
    """
    Run all symbolic verification checks.

    Returns
    -------
    dict mapping check name to bool (True = passed)
    """
    results = {}
    results["utility_indifference_bid"] = verify_utility_indifference_bid()
    results["utility_indifference_ask"] = verify_utility_indifference_ask()
    results["average_reservation_price"] = verify_average_reservation_price()
    results["exponential_foc_log_adjustment"] = verify_exponential_foc_log_adjustment()
    results["spread_sum"] = verify_spread_sum()
    return results


def print_formulas() -> None:
    """Print all key formulas in symbolic form."""
    print("=" * 70)
    print("AVELLANEDA-STOIKOV FINITE-HORIZON ANALYTICAL FORMULAS")
    print("=" * 70)

    print("\n--- Frozen-inventory value function ---")
    v = frozen_value_function(x, s, q, t)
    print(f"v(x,s,q,t) = {v}")

    print("\n--- Reservation bid ---")
    r_b = reservation_bid(s, q, t)
    print(f"r^b(s,q,t) = {r_b}")

    print("\n--- Reservation ask ---")
    r_a = reservation_ask(s, q, t)
    print(f"r^a(s,q,t) = {r_a}")

    print("\n--- Average reservation price ---")
    r_avg = average_reservation_price(s, q, t)
    print(f"r(s,q,t) = {r_avg}")

    print("\n--- Log adjustment (exponential intensities) ---")
    adj = log_adjustment()
    print(f"(1/gamma)*ln(1 + gamma/k) = {adj}")

    print("\n--- Optimal ask distance ---")
    da = optimal_ask_distance(s, q, t)
    print(f"delta^a = {da}")

    print("\n--- Optimal bid distance ---")
    db = optimal_bid_distance(s, q, t)
    print(f"delta^b = {db}")

    print("\n--- Total spread ---")
    spread = total_spread(s, q, t)
    print(f"delta^a + delta^b = {spread}")

    print("\n--- Approximate spread formula ---")
    psi = approximate_spread_formula()
    print(f"psi(t) = {psi}")

    print("\n--- HJB ansatz description ---")
    print(hjb_ansatz_theta_pde_description())

    print("\n--- Reservation prices from theta ---")
    theta_ids = reservation_prices_from_theta()
    for k_name, v_name in theta_ids.items():
        print(f"  {k_name}: {v_name}")
