"""
Tests for Experiment 3: Intensity derivations and infinite-horizon extension.

Verifies:
  - Exponential intensity derivation from log-impact + power-law order sizes
  - Power-law intensity derivation from power-law impact
  - Infinite-horizon reservation price formulas
  - Admissibility conditions
  - Numerical consistency
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import sympy as sp

from avellaneda_stoikov.exp3.intensity_and_infinite_horizon import (
    derive_exponential_intensity_symbolic,
    derive_power_law_intensity_symbolic,
    exponential_intensity_numerical,
    power_law_intensity_numerical,
    infinite_horizon_reservation_ask_symbolic,
    infinite_horizon_reservation_bid_symbolic,
    admissibility_condition_symbolic,
    boundedness_omega_choice,
    infinite_horizon_reservation_prices_numerical,
    compute_omega_for_q_max,
    infinite_horizon_inventory_scan,
    compare_finite_vs_infinite_horizon,
    run_symbolic_verification,
    intensity_comparison_table,
    alpha_sym, beta_sym, K_sym, Lambda_sym, delta_sym,
    gamma_sym, sigma_sym, omega_sym, q_sym, s_sym, q_max_sym,
)


# ---------------------------------------------------------------------------
# Symbolic verification tests
# ---------------------------------------------------------------------------

class TestSymbolicVerification:
    """Tests for symbolic derivation verification."""

    def test_all_symbolic_checks_pass(self):
        """All symbolic verification checks must pass."""
        results = run_symbolic_verification()
        for check_name, passed in results.items():
            assert passed, f"Symbolic check failed: {check_name}"

    def test_exponential_intensity_derivation(self):
        """Exponential intensity derivation must yield A*exp(-k*delta)."""
        results = run_symbolic_verification()
        assert results["exponential_intensity_derivation"]

    def test_power_law_intensity_derivation(self):
        """Power-law intensity derivation must yield B*delta^(-alpha/beta)."""
        results = run_symbolic_verification()
        assert results["power_law_intensity_derivation"]

    def test_admissibility_q0(self):
        """Admissibility condition at q=0 must be 2*omega."""
        results = run_symbolic_verification()
        assert results["admissibility_q0"]

    def test_boundedness_omega_admissible(self):
        """Boundedness omega choice must satisfy admissibility for |q| <= q_max."""
        results = run_symbolic_verification()
        assert results["boundedness_omega_admissible"]


# ---------------------------------------------------------------------------
# Exponential intensity tests
# ---------------------------------------------------------------------------

class TestExponentialIntensityDerivation:
    """Tests for exponential intensity derivation."""

    def test_derivation_returns_dict(self):
        """Derivation must return a dict with required keys."""
        result = derive_exponential_intensity_symbolic()
        required_keys = {"threshold_Q", "tail_probability", "lambda_full",
                         "A_identified", "k_identified", "lambda_exponential_form"}
        assert required_keys.issubset(set(result.keys()))

    def test_threshold_is_exponential(self):
        """Threshold Q must be exp(K*delta)."""
        result = derive_exponential_intensity_symbolic()
        expected = sp.exp(K_sym * delta_sym)
        diff = sp.simplify(result["threshold_Q"] - expected)
        assert diff == 0

    def test_tail_probability_is_exponential(self):
        """Tail probability must be exp(-alpha*K*delta)."""
        result = derive_exponential_intensity_symbolic()
        expected = sp.exp(-alpha_sym * K_sym * delta_sym)
        diff = sp.simplify(result["tail_probability"] - expected)
        assert diff == 0

    def test_k_identified_as_alpha_times_K(self):
        """k must be identified as alpha*K."""
        result = derive_exponential_intensity_symbolic()
        diff = sp.simplify(result["k_identified"] - alpha_sym * K_sym)
        assert diff == 0

    def test_numerical_exponential_intensity_positive(self):
        """Numerical exponential intensity must be positive."""
        for delta in [0.1, 0.5, 1.0, 2.0]:
            lam = exponential_intensity_numerical(delta, A=140.0, k=1.5)
            assert lam > 0, f"Intensity must be positive for delta={delta}"

    def test_numerical_exponential_intensity_decreasing(self):
        """Numerical exponential intensity must decrease with delta."""
        deltas = [0.1, 0.5, 1.0, 1.5, 2.0]
        lams = [exponential_intensity_numerical(d, A=140.0, k=1.5) for d in deltas]
        for i in range(len(lams) - 1):
            assert lams[i] > lams[i + 1], (
                f"Intensity should decrease: lambda[{deltas[i]}]={lams[i]:.4f} > lambda[{deltas[i+1]}]={lams[i+1]:.4f}"
            )

    def test_numerical_exponential_intensity_known_value(self):
        """Regression: lambda(1.0, A=140, k=1.5) = 140*exp(-1.5) ≈ 31.37."""
        lam = exponential_intensity_numerical(1.0, A=140.0, k=1.5)
        expected = 140.0 * math.exp(-1.5)
        assert abs(lam - expected) < 1e-10


# ---------------------------------------------------------------------------
# Power-law intensity tests
# ---------------------------------------------------------------------------

class TestPowerLawIntensityDerivation:
    """Tests for power-law intensity derivation."""

    def test_derivation_returns_dict(self):
        """Derivation must return a dict with required keys."""
        result = derive_power_law_intensity_symbolic()
        required_keys = {"threshold_Q", "tail_probability", "lambda_full",
                         "exponent", "lambda_power_form"}
        assert required_keys.issubset(set(result.keys()))

    def test_exponent_is_minus_alpha_over_beta(self):
        """Exponent must be -alpha/beta."""
        result = derive_power_law_intensity_symbolic()
        diff = sp.simplify(result["exponent"] - (-alpha_sym / beta_sym))
        assert diff == 0

    def test_tail_probability_is_power_law(self):
        """Tail probability must be delta^(-alpha/beta)."""
        result = derive_power_law_intensity_symbolic()
        expected = delta_sym ** (-alpha_sym / beta_sym)
        diff = sp.simplify(result["tail_probability"] - expected)
        assert diff == 0

    def test_numerical_power_law_intensity_positive(self):
        """Numerical power-law intensity must be positive."""
        for delta in [0.1, 0.5, 1.0, 2.0]:
            lam = power_law_intensity_numerical(delta, B=1.0, alpha=1.4, beta=0.5)
            assert lam > 0, f"Power-law intensity must be positive for delta={delta}"

    def test_numerical_power_law_intensity_decreasing(self):
        """Numerical power-law intensity must decrease with delta."""
        deltas = [0.1, 0.5, 1.0, 1.5, 2.0]
        lams = [power_law_intensity_numerical(d, B=1.0, alpha=1.4, beta=0.5) for d in deltas]
        for i in range(len(lams) - 1):
            assert lams[i] > lams[i + 1], (
                f"Power-law intensity should decrease: lambda[{deltas[i]}]={lams[i]:.4f} > lambda[{deltas[i+1]}]={lams[i+1]:.4f}"
            )

    def test_numerical_power_law_intensity_known_value(self):
        """Regression: lambda(1.0, B=1, alpha=1.4, beta=0.5) = 1.0^(-2.8) = 1.0."""
        lam = power_law_intensity_numerical(1.0, B=1.0, alpha=1.4, beta=0.5)
        expected = 1.0 ** (-1.4 / 0.5)
        assert abs(lam - expected) < 1e-10

    def test_power_law_intensity_zero_delta_raises(self):
        """Power-law intensity with delta=0 must raise AssertionError."""
        with pytest.raises(AssertionError):
            power_law_intensity_numerical(0.0, B=1.0, alpha=1.4, beta=0.5)


# ---------------------------------------------------------------------------
# Infinite-horizon reservation price tests
# ---------------------------------------------------------------------------

class TestInfiniteHorizonReservationPrices:
    """Tests for infinite-horizon stationary reservation prices."""

    def test_reservation_ask_symbolic_form(self):
        """Reservation ask must have correct symbolic form."""
        r_bar_a = infinite_horizon_reservation_ask_symbolic()
        # Should contain s_sym, gamma_sym, sigma_sym, omega_sym, q_sym
        assert s_sym in r_bar_a.free_symbols
        assert gamma_sym in r_bar_a.free_symbols
        assert sigma_sym in r_bar_a.free_symbols
        assert omega_sym in r_bar_a.free_symbols
        assert q_sym in r_bar_a.free_symbols

    def test_reservation_bid_symbolic_form(self):
        """Reservation bid must have correct symbolic form."""
        r_bar_b = infinite_horizon_reservation_bid_symbolic()
        assert s_sym in r_bar_b.free_symbols

    def test_admissibility_condition_positive_for_valid_params(self):
        """Admissibility condition must be positive for valid parameters."""
        # omega = 0.5*gamma^2*sigma^2*(q_max+1)^2 > 0.5*gamma^2*sigma^2*q^2 for |q| <= q_max
        g, sig, q_max = 0.1, 2.0, 5
        omega = compute_omega_for_q_max(q_max, g, sig)
        for q_val in range(-q_max, q_max + 1):
            denom = 2 * omega - g**2 * q_val**2 * sig**2
            assert denom > 0, (
                f"Admissibility condition must be positive for q={q_val}, got {denom}"
            )

    def test_admissibility_condition_fails_for_large_q(self):
        """Admissibility condition must fail for q > q_max."""
        g, sig, q_max = 0.1, 2.0, 5
        omega = compute_omega_for_q_max(q_max, g, sig)
        q_too_large = q_max + 1
        denom = 2 * omega - g**2 * q_too_large**2 * sig**2
        assert denom <= 0, (
            f"Admissibility should fail for q={q_too_large} > q_max={q_max}"
        )

    def test_numerical_prices_admissible_params(self):
        """Numerical prices must be finite for admissible parameters."""
        g, sig, q_max = 0.1, 2.0, 5
        omega = compute_omega_for_q_max(q_max, g, sig)
        for q_val in range(-q_max, q_max + 1):
            res = infinite_horizon_reservation_prices_numerical(100.0, q_val, g, sig, omega)
            assert res["admissible"], f"Should be admissible for q={q_val}"
            assert math.isfinite(res["r_bar_a"]), f"r_bar_a must be finite for q={q_val}"
            assert math.isfinite(res["r_bar_b"]), f"r_bar_b must be finite for q={q_val}"

    def test_numerical_prices_inadmissible_params(self):
        """Numerical prices must return NaN for inadmissible parameters."""
        g, sig = 0.1, 2.0
        omega_too_small = 0.001  # Too small for q=5
        res = infinite_horizon_reservation_prices_numerical(100.0, 5, g, sig, omega_too_small)
        assert not res["admissible"]
        assert math.isnan(res["r_bar_a"])
        assert math.isnan(res["r_bar_b"])

    def test_reservation_ask_above_mid_for_negative_inventory(self):
        """For negative inventory, reservation ask must be above mid-price."""
        g, sig, q_max = 0.1, 2.0, 5
        omega = compute_omega_for_q_max(q_max, g, sig)
        for q_val in [-1, -2, -3]:
            res = infinite_horizon_reservation_prices_numerical(100.0, q_val, g, sig, omega)
            assert res["r_bar_a"] > 100.0, (
                f"r_bar_a should be > s for q={q_val}, got {res['r_bar_a']}"
            )

    def test_reservation_bid_below_mid_for_positive_inventory(self):
        """For positive inventory, reservation bid must be below mid-price."""
        g, sig, q_max = 0.1, 2.0, 5
        omega = compute_omega_for_q_max(q_max, g, sig)
        for q_val in [1, 2, 3]:
            res = infinite_horizon_reservation_prices_numerical(100.0, q_val, g, sig, omega)
            assert res["r_bar_b"] < 100.0, (
                f"r_bar_b should be < s for q={q_val}, got {res['r_bar_b']}"
            )

    def test_reservation_prices_at_zero_inventory(self):
        """At q=0, reservation prices should be symmetric around mid-price."""
        g, sig, q_max = 0.1, 2.0, 5
        omega = compute_omega_for_q_max(q_max, g, sig)
        res = infinite_horizon_reservation_prices_numerical(100.0, 0, g, sig, omega)
        # r_bar_a(q=0) = s + (1/gamma)*ln(1 + gamma^2*sigma^2/(2*omega))
        # r_bar_b(q=0) = s + (1/gamma)*ln(1 - gamma^2*sigma^2/(2*omega))
        # r_bar_a > s and r_bar_b < s
        assert res["r_bar_a"] > 100.0, "r_bar_a should be > s at q=0"
        assert res["r_bar_b"] < 100.0, "r_bar_b should be < s at q=0"


# ---------------------------------------------------------------------------
# Omega computation tests
# ---------------------------------------------------------------------------

class TestOmegaComputation:
    """Tests for omega computation."""

    def test_omega_positive(self):
        """Omega must be positive."""
        for q_max in [1, 5, 10]:
            omega = compute_omega_for_q_max(q_max, 0.1, 2.0)
            assert omega > 0, f"Omega must be positive for q_max={q_max}"

    def test_omega_increases_with_q_max(self):
        """Omega must increase with q_max."""
        omegas = [compute_omega_for_q_max(q_max, 0.1, 2.0) for q_max in [1, 3, 5, 10]]
        for i in range(len(omegas) - 1):
            assert omegas[i] < omegas[i + 1], "Omega should increase with q_max"

    def test_omega_formula(self):
        """Omega must equal (1/2)*gamma^2*sigma^2*(q_max+1)^2."""
        g, sig, q_max = 0.1, 2.0, 5
        omega = compute_omega_for_q_max(q_max, g, sig)
        expected = 0.5 * g**2 * sig**2 * (q_max + 1)**2
        assert abs(omega - expected) < 1e-10


# ---------------------------------------------------------------------------
# Inventory scan tests
# ---------------------------------------------------------------------------

class TestInventoryScan:
    """Tests for infinite-horizon inventory scan."""

    def test_scan_returns_correct_keys(self):
        """Scan must return dict with required keys."""
        scan = infinite_horizon_inventory_scan(100.0, 0.1, 2.0, 5)
        required_keys = {"q", "r_bar_a", "r_bar_b", "r_bar_avg", "admissible", "omega", "s"}
        assert required_keys.issubset(set(scan.keys()))

    def test_scan_all_admissible_within_q_max(self):
        """All inventory levels within q_max must be admissible."""
        scan = infinite_horizon_inventory_scan(100.0, 0.1, 2.0, 5)
        assert np.all(scan["admissible"]), "All q within q_max should be admissible"

    def test_scan_r_bar_avg_decreases_with_inventory(self):
        """Average reservation price must decrease with inventory."""
        scan = infinite_horizon_inventory_scan(100.0, 0.1, 2.0, 5)
        r_avg = scan["r_bar_avg"]
        for i in range(len(r_avg) - 1):
            assert r_avg[i] > r_avg[i + 1], (
                f"r_bar_avg should decrease with q: r[{i}]={r_avg[i]:.4f} > r[{i+1}]={r_avg[i+1]:.4f}"
            )

    def test_scan_r_bar_a_above_r_bar_b(self):
        """Reservation ask must be above reservation bid for all inventories."""
        scan = infinite_horizon_inventory_scan(100.0, 0.1, 2.0, 5)
        assert np.all(scan["r_bar_a"] > scan["r_bar_b"]), (
            "r_bar_a must be above r_bar_b for all inventories"
        )


# ---------------------------------------------------------------------------
# Finite vs infinite horizon comparison tests
# ---------------------------------------------------------------------------

class TestFiniteVsInfiniteHorizon:
    """Tests for comparison between finite and infinite horizon models."""

    def test_comparison_returns_correct_keys(self):
        """Comparison must return dict with required keys."""
        comp = compare_finite_vs_infinite_horizon(100.0, 0.1, 2.0, 1.0, 0.0, 5)
        required_keys = {"q", "r_finite", "r_inf_avg", "r_inf_a", "r_inf_b", "omega", "tau"}
        assert required_keys.issubset(set(comp.keys()))

    def test_finite_horizon_linear_in_inventory(self):
        """Finite-horizon reservation price must be linear in inventory."""
        comp = compare_finite_vs_infinite_horizon(100.0, 0.1, 2.0, 1.0, 0.0, 5)
        q_vals = comp["q"]
        r_finite = comp["r_finite"]
        # Fit linear regression
        slope = np.polyfit(q_vals, r_finite, 1)[0]
        expected_slope = -0.1 * 4.0 * 1.0  # -gamma*sigma^2*tau
        assert abs(slope - expected_slope) < 1e-10

    def test_both_models_agree_at_zero_inventory(self):
        """Both models should give reservation price near s at q=0."""
        comp = compare_finite_vs_infinite_horizon(100.0, 0.1, 2.0, 1.0, 0.0, 5)
        q_idx = np.where(comp["q"] == 0)[0][0]
        # Finite horizon: r(s,0,t) = s
        assert abs(comp["r_finite"][q_idx] - 100.0) < 1e-10
        # Infinite horizon: r_bar_avg(s,0) != s in general (asymmetric log terms)
        # But should be close to s for small gamma
        assert abs(comp["r_inf_avg"][q_idx] - 100.0) < 5.0  # Within 5 of mid-price

    def test_both_models_qualitative_agreement(self):
        """Both models must show same qualitative inventory dependence."""
        comp = compare_finite_vs_infinite_horizon(100.0, 0.1, 2.0, 1.0, 0.0, 5)
        # Both should decrease with inventory
        r_finite = comp["r_finite"]
        r_inf = comp["r_inf_avg"]
        for i in range(len(r_finite) - 1):
            assert r_finite[i] > r_finite[i + 1], "Finite-horizon r should decrease with q"
        for i in range(len(r_inf) - 1):
            assert r_inf[i] > r_inf[i + 1], "Infinite-horizon r should decrease with q"


# ---------------------------------------------------------------------------
# Intensity comparison table tests
# ---------------------------------------------------------------------------

class TestIntensityComparisonTable:
    """Tests for intensity comparison table."""

    def test_table_returns_correct_keys(self):
        """Table must return dict with required keys."""
        deltas = np.array([0.1, 0.5, 1.0])
        result = intensity_comparison_table(deltas)
        assert "deltas" in result
        assert "lambda_exp" in result
        assert "lambda_power" in result

    def test_table_shapes_match(self):
        """All arrays in table must have same shape."""
        deltas = np.linspace(0.1, 3.0, 50)
        result = intensity_comparison_table(deltas)
        assert result["deltas"].shape == result["lambda_exp"].shape
        assert result["deltas"].shape == result["lambda_power"].shape

    def test_all_intensities_positive(self):
        """All intensities in table must be positive."""
        deltas = np.linspace(0.1, 3.0, 50)
        result = intensity_comparison_table(deltas)
        assert np.all(result["lambda_exp"] > 0)
        assert np.all(result["lambda_power"] > 0)
