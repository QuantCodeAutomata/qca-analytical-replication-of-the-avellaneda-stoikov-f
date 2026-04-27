"""
Experiment 3: Analytical replication of order-arrival intensity derivations
and the infinite-horizon extension of the Avellaneda-Stoikov model.

Covers:
  - Derivation of exponential intensity from power-law order sizes + log impact
  - Derivation of power-law intensity from power-law order sizes + power-law impact
  - Infinite-horizon stationary reservation price formulas
  - Admissibility conditions and boundedness

Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"

# Custom — Context7 found no library equivalent for AS microstructure intensity derivations
# Using sympy for symbolic algebra — Context7 confirmed
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Symbolic variables
# ---------------------------------------------------------------------------
delta_sym = sp.Symbol("delta", positive=True)
Q_sym = sp.Symbol("Q", positive=True)
x_sym = sp.Symbol("x", positive=True)
alpha_sym = sp.Symbol("alpha", positive=True)
beta_sym = sp.Symbol("beta", positive=True)
K_sym = sp.Symbol("K", positive=True)
B_sym = sp.Symbol("B", positive=True)
Lambda_sym = sp.Symbol("Lambda", positive=True)
A_sym = sp.Symbol("A", positive=True)
k_sym = sp.Symbol("k", positive=True)
gamma_sym = sp.Symbol("gamma", positive=True)
sigma_sym = sp.Symbol("sigma", positive=True)
omega_sym = sp.Symbol("omega", positive=True)
q_sym = sp.Symbol("q", real=True)
s_sym = sp.Symbol("s", real=True)
q_max_sym = sp.Symbol("q_max", positive=True)


# ---------------------------------------------------------------------------
# Steps 1-6: Exponential intensity from log-impact + power-law order sizes
# ---------------------------------------------------------------------------

def derive_exponential_intensity_symbolic() -> Dict[str, sp.Expr]:
    """
    Derive exponential execution intensity from stylized microstructure.

    Assumptions:
      - Market order size Q has power-law density: f^Q(x) ~ x^(-1-alpha), alpha > 0
      - Temporary price impact: Delta_p = (1/K) * ln(Q)  (logarithmic impact)
      - Fill intensity at quote distance delta:
          lambda(delta) = Lambda * P(Delta_p > delta)
                        = Lambda * P(ln(Q) > K*delta)
                        = Lambda * P(Q > exp(K*delta))

    Derivation:
      P(Q > exp(K*delta)) = integral_{exp(K*delta)}^{inf} C * x^(-1-alpha) dx
                          = C * [x^(-alpha) / (-alpha)]_{exp(K*delta)}^{inf}
                          = (C/alpha) * exp(-alpha*K*delta)

    Result: lambda(delta) = A * exp(-k*delta)
      where A = Lambda * C / alpha  and  k = alpha * K

    Returns
    -------
    dict with symbolic expressions for each derivation step
    """
    # Normalization constant for power-law density on [x_min, inf)
    # f^Q(x) = alpha * x_min^alpha * x^(-1-alpha)  (Pareto distribution)
    # For simplicity, use unnormalized tail: P(Q > z) = (z/x_min)^(-alpha)
    # With x_min = 1 (unit normalization): P(Q > z) = z^(-alpha)

    # Threshold from log-impact: Q > exp(K*delta)
    threshold = sp.exp(K_sym * delta_sym)

    # Tail probability: P(Q > threshold) = threshold^(-alpha)
    tail_prob = threshold ** (-alpha_sym)
    tail_prob_simplified = sp.simplify(tail_prob)

    # Full intensity
    lambda_expr = Lambda_sym * tail_prob_simplified
    lambda_simplified = sp.simplify(lambda_expr)

    # Identify A and k
    # lambda = Lambda * exp(-alpha*K*delta) = A * exp(-k*delta)
    A_identified = Lambda_sym  # (absorbing normalization constant)
    k_identified = alpha_sym * K_sym

    # Verify: lambda = A * exp(-k*delta)
    lambda_exponential_form = A_sym * sp.exp(-k_sym * delta_sym)

    return {
        "threshold_Q": threshold,
        "tail_probability": tail_prob_simplified,
        "lambda_full": lambda_simplified,
        "A_identified": A_identified,
        "k_identified": k_identified,
        "lambda_exponential_form": lambda_exponential_form,
        "derivation_note": (
            "lambda(delta) = Lambda * exp(-alpha*K*delta) = A*exp(-k*delta) "
            "with A=Lambda (up to normalization) and k=alpha*K"
        ),
    }


def exponential_intensity_numerical(
    delta: float,
    A: float,
    k: float,
) -> float:
    """
    Evaluate exponential intensity lambda(delta) = A * exp(-k * delta).

    Parameters
    ----------
    delta : quote distance
    A : intensity scale
    k : intensity decay

    Returns
    -------
    float intensity value
    """
    return A * np.exp(-k * delta)


# ---------------------------------------------------------------------------
# Steps 7-8: Power-law intensity from power-law impact
# ---------------------------------------------------------------------------

def derive_power_law_intensity_symbolic() -> Dict[str, sp.Expr]:
    """
    Derive power-law execution intensity from power-law market impact.

    Assumptions:
      - Market order size Q has power-law density: f^Q(x) ~ x^(-1-alpha), alpha > 0
      - Temporary price impact: Delta_p = Q^beta  (power-law impact), beta > 0
      - Fill intensity at quote distance delta:
          lambda(delta) = Lambda * P(Delta_p > delta)
                        = Lambda * P(Q^beta > delta)
                        = Lambda * P(Q > delta^(1/beta))

    Derivation:
      P(Q > delta^(1/beta)) = (delta^(1/beta))^(-alpha) = delta^(-alpha/beta)

    Result: lambda(delta) = B * delta^(-alpha/beta)
      where B absorbs Lambda and normalization constants

    Returns
    -------
    dict with symbolic expressions for each derivation step
    """
    # Threshold from power-law impact: Q > delta^(1/beta)
    threshold = delta_sym ** (1 / beta_sym)

    # Tail probability: P(Q > threshold) = threshold^(-alpha)
    tail_prob = threshold ** (-alpha_sym)
    tail_prob_simplified = sp.simplify(tail_prob)

    # Full intensity
    lambda_expr = Lambda_sym * tail_prob_simplified
    lambda_simplified = sp.simplify(lambda_expr)

    # Power-law form: lambda = B * delta^(-alpha/beta)
    lambda_power_form = B_sym * delta_sym ** (-alpha_sym / beta_sym)

    return {
        "threshold_Q": threshold,
        "tail_probability": tail_prob_simplified,
        "lambda_full": lambda_simplified,
        "exponent": -alpha_sym / beta_sym,
        "lambda_power_form": lambda_power_form,
        "derivation_note": (
            "lambda(delta) = Lambda * delta^(-alpha/beta) = B * delta^(-alpha/beta) "
            "with B absorbing Lambda and normalization constants"
        ),
    }


def power_law_intensity_numerical(
    delta: float,
    B: float,
    alpha: float,
    beta: float,
) -> float:
    """
    Evaluate power-law intensity lambda(delta) = B * delta^(-alpha/beta).

    Parameters
    ----------
    delta : quote distance (must be > 0)
    B : intensity scale
    alpha : order size tail exponent
    beta : impact exponent

    Returns
    -------
    float intensity value
    """
    assert delta > 0, "delta must be positive for power-law intensity"
    return B * delta ** (-alpha / beta)


# ---------------------------------------------------------------------------
# Steps 9-12: Infinite-horizon reservation prices
# ---------------------------------------------------------------------------

def infinite_horizon_reservation_ask_symbolic() -> sp.Expr:
    """
    Compute the stationary infinite-horizon reservation ask price.

    Under the infinite-horizon objective with discount rate omega:
      v_bar(x,s,q) = E[integral_0^inf -exp(-omega*t) * exp(-gamma*(x + q*S_t)) dt]

    The stationary reservation ask is:
      r_bar^a(s,q) = s + (1/gamma) * ln(1 + ((1-2q)*gamma^2*sigma^2) /
                                              (2*omega - gamma^2*q^2*sigma^2))

    Returns
    -------
    sympy expression for r_bar^a
    """
    numerator = (1 - 2 * q_sym) * gamma_sym**2 * sigma_sym**2
    denominator = 2 * omega_sym - gamma_sym**2 * q_sym**2 * sigma_sym**2
    r_bar_a = s_sym + (1 / gamma_sym) * sp.log(1 + numerator / denominator)
    return r_bar_a


def infinite_horizon_reservation_bid_symbolic() -> sp.Expr:
    """
    Compute the stationary infinite-horizon reservation bid price.

    The stationary reservation bid is:
      r_bar^b(s,q) = s + (1/gamma) * ln(1 + ((-1-2q)*gamma^2*sigma^2) /
                                              (2*omega - gamma^2*q^2*sigma^2))

    Returns
    -------
    sympy expression for r_bar^b
    """
    numerator = (-1 - 2 * q_sym) * gamma_sym**2 * sigma_sym**2
    denominator = 2 * omega_sym - gamma_sym**2 * q_sym**2 * sigma_sym**2
    r_bar_b = s_sym + (1 / gamma_sym) * sp.log(1 + numerator / denominator)
    return r_bar_b


def admissibility_condition_symbolic() -> sp.Expr:
    """
    Return the admissibility condition for the infinite-horizon model.

    Condition: omega > (1/2) * gamma^2 * sigma^2 * q^2

    This ensures the denominator 2*omega - gamma^2*q^2*sigma^2 > 0
    and that the utility integral converges.

    Returns
    -------
    sympy expression for the condition (should be > 0)
    """
    return 2 * omega_sym - gamma_sym**2 * q_sym**2 * sigma_sym**2


def boundedness_omega_choice() -> sp.Expr:
    """
    Return the suggested omega choice for bounded inventory |q| <= q_max.

    omega = (1/2) * gamma^2 * sigma^2 * (q_max + 1)^2

    This ensures admissibility for all |q| <= q_max.

    Returns
    -------
    sympy expression for the suggested omega
    """
    return sp.Rational(1, 2) * gamma_sym**2 * sigma_sym**2 * (q_max_sym + 1)**2


def infinite_horizon_reservation_prices_numerical(
    s: float,
    q: int,
    gamma: float,
    sigma: float,
    omega: float,
) -> Dict[str, float]:
    """
    Evaluate infinite-horizon reservation prices numerically.

    Parameters
    ----------
    s : current mid-price
    q : current inventory
    gamma : risk aversion
    sigma : volatility
    omega : discount rate

    Returns
    -------
    dict with keys 'r_bar_a', 'r_bar_b', 'r_bar_avg', 'admissible'
    """
    denom = 2 * omega - gamma**2 * q**2 * sigma**2
    admissible = denom > 0

    if not admissible:
        return {
            "r_bar_a": np.nan,
            "r_bar_b": np.nan,
            "r_bar_avg": np.nan,
            "admissible": False,
            "denominator": denom,
        }

    num_a = (1 - 2 * q) * gamma**2 * sigma**2
    num_b = (-1 - 2 * q) * gamma**2 * sigma**2

    r_bar_a = s + np.log(1 + num_a / denom) / gamma
    r_bar_b = s + np.log(1 + num_b / denom) / gamma
    r_bar_avg = (r_bar_a + r_bar_b) / 2

    return {
        "r_bar_a": r_bar_a,
        "r_bar_b": r_bar_b,
        "r_bar_avg": r_bar_avg,
        "admissible": True,
        "denominator": denom,
    }


def compute_omega_for_q_max(
    q_max: int,
    gamma: float,
    sigma: float,
) -> float:
    """
    Compute the suggested omega for bounded inventory |q| <= q_max.

    omega = (1/2) * gamma^2 * sigma^2 * (q_max + 1)^2

    Parameters
    ----------
    q_max : maximum absolute inventory
    gamma : risk aversion
    sigma : volatility

    Returns
    -------
    float omega value
    """
    return 0.5 * gamma**2 * sigma**2 * (q_max + 1)**2


# ---------------------------------------------------------------------------
# Numerical illustrations
# ---------------------------------------------------------------------------

def intensity_comparison_table(
    deltas: np.ndarray,
    A: float = 140.0,
    k: float = 1.5,
    B: float = 1.0,
    alpha: float = 1.4,
    beta: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Compute both exponential and power-law intensities for a range of deltas.

    Parameters
    ----------
    deltas : array of quote distances
    A : exponential intensity scale
    k : exponential intensity decay
    B : power-law intensity scale
    alpha : order size tail exponent
    beta : impact exponent

    Returns
    -------
    dict with 'deltas', 'lambda_exp', 'lambda_power'
    """
    lambda_exp = np.array([exponential_intensity_numerical(d, A, k) for d in deltas])
    lambda_power = np.array([power_law_intensity_numerical(d, B, alpha, beta) for d in deltas])
    return {
        "deltas": deltas,
        "lambda_exp": lambda_exp,
        "lambda_power": lambda_power,
    }


def infinite_horizon_inventory_scan(
    s: float = 100.0,
    gamma: float = 0.1,
    sigma: float = 2.0,
    q_max: int = 5,
    inventories: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Scan infinite-horizon reservation prices across inventory levels.

    Uses the suggested omega = (1/2)*gamma^2*sigma^2*(q_max+1)^2.

    Parameters
    ----------
    s : mid-price
    gamma : risk aversion
    sigma : volatility
    q_max : maximum inventory for omega choice
    inventories : array of inventory values to scan

    Returns
    -------
    dict with arrays for q, r_bar_a, r_bar_b, r_bar_avg, admissible
    """
    if inventories is None:
        inventories = np.arange(-q_max, q_max + 1)

    omega = compute_omega_for_q_max(q_max, gamma, sigma)

    r_bar_a_arr = []
    r_bar_b_arr = []
    r_bar_avg_arr = []
    admissible_arr = []

    for q in inventories:
        res = infinite_horizon_reservation_prices_numerical(s, int(q), gamma, sigma, omega)
        r_bar_a_arr.append(res["r_bar_a"])
        r_bar_b_arr.append(res["r_bar_b"])
        r_bar_avg_arr.append(res["r_bar_avg"])
        admissible_arr.append(res["admissible"])

    return {
        "q": inventories,
        "r_bar_a": np.array(r_bar_a_arr),
        "r_bar_b": np.array(r_bar_b_arr),
        "r_bar_avg": np.array(r_bar_avg_arr),
        "admissible": np.array(admissible_arr),
        "omega": omega,
        "s": s,
    }


def compare_finite_vs_infinite_horizon(
    s: float = 100.0,
    gamma: float = 0.1,
    sigma: float = 2.0,
    T: float = 1.0,
    t: float = 0.0,
    q_max: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Compare finite-horizon and infinite-horizon average reservation prices.

    Finite-horizon: r(s,q,t) = s - q*gamma*sigma^2*(T-t)
    Infinite-horizon: r_bar_avg(s,q) = (r_bar_a + r_bar_b) / 2

    Parameters
    ----------
    s : mid-price
    gamma : risk aversion
    sigma : volatility
    T : finite horizon
    t : current time
    q_max : max inventory for omega choice

    Returns
    -------
    dict with comparison arrays
    """
    inventories = np.arange(-q_max, q_max + 1)
    tau = T - t

    # Finite-horizon
    r_finite = s - inventories * gamma * sigma**2 * tau

    # Infinite-horizon
    inf_scan = infinite_horizon_inventory_scan(s, gamma, sigma, q_max, inventories)

    return {
        "q": inventories,
        "r_finite": r_finite,
        "r_inf_avg": inf_scan["r_bar_avg"],
        "r_inf_a": inf_scan["r_bar_a"],
        "r_inf_b": inf_scan["r_bar_b"],
        "omega": inf_scan["omega"],
        "tau": tau,
    }


def run_symbolic_verification() -> Dict[str, bool]:
    """
    Run symbolic verification checks for Experiment 3.

    Returns
    -------
    dict mapping check name to bool (True = passed)
    """
    results = {}

    # Check 1: Exponential intensity derivation
    exp_deriv = derive_exponential_intensity_symbolic()
    # lambda = Lambda * exp(-alpha*K*delta) should match A*exp(-k*delta) form
    # Verify by substituting A=Lambda, k=alpha*K
    lam_derived = exp_deriv["lambda_full"]
    lam_expected = Lambda_sym * sp.exp(-alpha_sym * K_sym * delta_sym)
    diff1 = sp.simplify(lam_derived - lam_expected)
    results["exponential_intensity_derivation"] = (diff1 == 0)

    # Check 2: Power-law intensity derivation
    pow_deriv = derive_power_law_intensity_symbolic()
    lam_pow_derived = pow_deriv["lambda_full"]
    lam_pow_expected = Lambda_sym * delta_sym ** (-alpha_sym / beta_sym)
    diff2 = sp.simplify(lam_pow_derived - lam_pow_expected)
    results["power_law_intensity_derivation"] = (diff2 == 0)

    # Check 3: Admissibility condition is positive for valid omega
    # For q=0, omega>0 is sufficient
    adm = admissibility_condition_symbolic()
    adm_at_q0 = adm.subs(q_sym, 0)
    # Should be 2*omega > 0, which is True for omega > 0
    results["admissibility_q0"] = sp.simplify(adm_at_q0 - 2 * omega_sym) == 0

    # Check 4: Boundedness omega choice satisfies admissibility for |q| <= q_max
    omega_choice = boundedness_omega_choice()
    # For q = q_max: 2*omega - gamma^2*q_max^2*sigma^2
    #   = gamma^2*sigma^2*(q_max+1)^2 - gamma^2*q_max^2*sigma^2
    #   = gamma^2*sigma^2*(2*q_max + 1) > 0
    adm_at_qmax = adm.subs(omega_sym, omega_choice).subs(q_sym, q_max_sym)
    adm_simplified = sp.simplify(adm_at_qmax)
    # Should be gamma^2*sigma^2*(2*q_max+1) > 0
    results["boundedness_omega_admissible"] = sp.simplify(
        adm_simplified - gamma_sym**2 * sigma_sym**2 * (2 * q_max_sym + 1)
    ) == 0

    return results


def print_derivation_summary() -> None:
    """Print a summary of all derivations in Experiment 3."""
    print("=" * 70)
    print("EXPERIMENT 3: INTENSITY DERIVATIONS & INFINITE-HORIZON MODEL")
    print("=" * 70)

    print("\n--- Exponential Intensity Derivation ---")
    exp_deriv = derive_exponential_intensity_symbolic()
    print(f"  Threshold Q: {exp_deriv['threshold_Q']}")
    print(f"  Tail P(Q > threshold): {exp_deriv['tail_probability']}")
    print(f"  lambda(delta): {exp_deriv['lambda_full']}")
    print(f"  Note: {exp_deriv['derivation_note']}")

    print("\n--- Power-Law Intensity Derivation ---")
    pow_deriv = derive_power_law_intensity_symbolic()
    print(f"  Threshold Q: {pow_deriv['threshold_Q']}")
    print(f"  Tail P(Q > threshold): {pow_deriv['tail_probability']}")
    print(f"  lambda(delta): {pow_deriv['lambda_full']}")
    print(f"  Note: {pow_deriv['derivation_note']}")

    print("\n--- Infinite-Horizon Reservation Prices ---")
    r_bar_a = infinite_horizon_reservation_ask_symbolic()
    r_bar_b = infinite_horizon_reservation_bid_symbolic()
    print(f"  r_bar^a(s,q) = {r_bar_a}")
    print(f"  r_bar^b(s,q) = {r_bar_b}")

    print("\n--- Admissibility Condition ---")
    adm = admissibility_condition_symbolic()
    print(f"  Condition (must be > 0): {adm} > 0")
    print(f"  i.e., omega > (1/2)*gamma^2*sigma^2*q^2")

    print("\n--- Suggested Omega for |q| <= q_max ---")
    omega_choice = boundedness_omega_choice()
    print(f"  omega = {omega_choice}")
