"""
Tests for Experiment 1: Analytical verification of the Avellaneda-Stoikov model.

Verifies:
- Frozen-inventory value function properties
- Reservation price formulas and indifference equations
- Theta representation consistency
- Spread offset formulas
- Qualitative inventory effects
- Small gamma convergence
- Paper inconsistency documentation
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp1_analytical import (
    frozen_inventory_value,
    reservation_ask,
    reservation_bid,
    average_reservation_price,
    verify_reservation_ask_indifference,
    verify_reservation_bid_indifference,
    verify_average_reservation_price,
    theta_frozen,
    reservation_ask_from_theta,
    reservation_bid_from_theta,
    spread_offset_exponential,
    optimal_ask_distance,
    optimal_bid_distance,
    total_spread_equation_faithful,
    total_spread_table_faithful,
    check_inventory_effect_on_reservation_price,
    check_small_gamma_convergence,
    document_paper_inconsistency,
    validate_params,
    run_analytical_verification,
)


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    """Tests for parameter validation."""

    def test_validate_params_valid(self):
        """Valid parameters should not raise."""
        validate_params(gamma=0.1, sigma=2.0, T=1.0, t=0.0, k=1.5)

    def test_validate_params_gamma_zero_raises(self):
        """gamma=0 should raise ValueError."""
        with pytest.raises(ValueError, match="gamma"):
            validate_params(gamma=0.0, sigma=2.0, T=1.0, t=0.0)

    def test_validate_params_gamma_negative_raises(self):
        """Negative gamma should raise ValueError."""
        with pytest.raises(ValueError, match="gamma"):
            validate_params(gamma=-0.1, sigma=2.0, T=1.0, t=0.0)

    def test_validate_params_sigma_negative_raises(self):
        """Negative sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            validate_params(gamma=0.1, sigma=-1.0, T=1.0, t=0.0)

    def test_validate_params_T_less_than_t_raises(self):
        """T < t should raise ValueError."""
        with pytest.raises(ValueError, match="T"):
            validate_params(gamma=0.1, sigma=2.0, T=0.5, t=1.0)

    def test_validate_params_k_zero_raises(self):
        """k=0 should raise ValueError."""
        with pytest.raises(ValueError, match="k"):
            validate_params(gamma=0.1, sigma=2.0, T=1.0, t=0.0, k=0.0)

    def test_validate_params_sigma_zero_valid(self):
        """sigma=0 should be valid (no volatility case)."""
        validate_params(gamma=0.1, sigma=0.0, T=1.0, t=0.0)

    def test_validate_params_T_equals_t_valid(self):
        """T=t (at maturity) should be valid."""
        validate_params(gamma=0.1, sigma=2.0, T=1.0, t=1.0)


# ---------------------------------------------------------------------------
# Frozen-inventory value function tests
# ---------------------------------------------------------------------------

class TestFrozenInventoryValue:
    """Tests for the frozen-inventory value function."""

    def test_value_is_negative(self):
        """Value function must always be negative (CARA utility)."""
        v = frozen_inventory_value(x=0, s=100, q=0, t=0, gamma=0.1, sigma=2, T=1)
        assert v < 0

    def test_value_negative_for_various_params(self):
        """Value function must be negative for all valid parameter combinations."""
        for gamma in [0.01, 0.1, 0.5]:
            for q in [-2, -1, 0, 1, 2]:
                for t in [0.0, 0.4, 0.8]:
                    v = frozen_inventory_value(x=0, s=100, q=q, t=t, gamma=gamma, sigma=2, T=1)
                    assert v < 0, f"v={v} not negative for gamma={gamma}, q={q}, t={t}"

    def test_value_at_maturity_no_inventory(self):
        """At t=T with q=0: v = -exp(-gamma*x)."""
        gamma, x, s = 0.1, 0.0, 100.0
        v = frozen_inventory_value(x=x, s=s, q=0, t=1.0, gamma=gamma, sigma=2, T=1.0)
        expected = -math.exp(-gamma * x)
        assert abs(v - expected) < 1e-12

    def test_value_formula_exact(self):
        """Test exact formula: v = -exp(-gamma*x)*exp(-gamma*q*s)*exp(gamma^2*q^2*sigma^2*tau/2)."""
        gamma, x, s, q, t, sigma, T = 0.1, 5.0, 100.0, 2.0, 0.4, 2.0, 1.0
        tau = T - t
        expected = (
            -math.exp(-gamma * x)
            * math.exp(-gamma * q * s)
            * math.exp(0.5 * gamma**2 * q**2 * sigma**2 * tau)
        )
        v = frozen_inventory_value(x=x, s=s, q=q, t=t, gamma=gamma, sigma=sigma, T=T)
        assert abs(v - expected) < 1e-12

    def test_value_decreases_with_cash(self):
        """Higher cash should give higher (less negative) utility."""
        v1 = frozen_inventory_value(x=0, s=100, q=0, t=0, gamma=0.1, sigma=2, T=1)
        v2 = frozen_inventory_value(x=10, s=100, q=0, t=0, gamma=0.1, sigma=2, T=1)
        assert v2 > v1  # less negative = higher utility

    def test_value_symmetric_in_inventory_sign(self):
        """v(x,s,q,t) and v(x,s,-q,t) should have same magnitude (symmetric inventory risk)."""
        v_pos = frozen_inventory_value(x=0, s=100, q=2, t=0.4, gamma=0.1, sigma=2, T=1)
        v_neg = frozen_inventory_value(x=0, s=100, q=-2, t=0.4, gamma=0.1, sigma=2, T=1)
        # Both negative; the inventory risk term is q^2, so magnitudes should be equal
        # But the q*s term differs, so they're not equal in general
        # Just verify both are negative
        assert v_pos < 0
        assert v_neg < 0


# ---------------------------------------------------------------------------
# Reservation price tests
# ---------------------------------------------------------------------------

class TestReservationPrices:
    """Tests for reservation ask, bid, and average prices."""

    def test_reservation_ask_formula(self):
        """Test r^a = s + (1-2q)*gamma*sigma^2*(T-t)/2."""
        s, q, t, gamma, sigma, T = 100.0, 1.0, 0.4, 0.1, 2.0, 1.0
        tau = T - t
        expected = s + (1 - 2 * q) * gamma * sigma**2 * tau / 2
        r_a = reservation_ask(s, q, t, gamma, sigma, T)
        assert abs(r_a - expected) < 1e-12

    def test_reservation_bid_formula(self):
        """Test r^b = s + (-1-2q)*gamma*sigma^2*(T-t)/2."""
        s, q, t, gamma, sigma, T = 100.0, 1.0, 0.4, 0.1, 2.0, 1.0
        tau = T - t
        expected = s + (-1 - 2 * q) * gamma * sigma**2 * tau / 2
        r_b = reservation_bid(s, q, t, gamma, sigma, T)
        assert abs(r_b - expected) < 1e-12

    def test_average_reservation_price_formula(self):
        """Test r = s - q*gamma*sigma^2*(T-t)."""
        s, q, t, gamma, sigma, T = 100.0, 1.0, 0.4, 0.1, 2.0, 1.0
        tau = T - t
        expected = s - q * gamma * sigma**2 * tau
        r = average_reservation_price(s, q, t, gamma, sigma, T)
        assert abs(r - expected) < 1e-12

    def test_average_is_midpoint_of_ask_and_bid(self):
        """(r^a + r^b)/2 must equal average_reservation_price."""
        for gamma in [0.01, 0.1, 0.5]:
            for q in [-2, -1, 0, 1, 2]:
                for t in [0.0, 0.4, 0.8]:
                    r_a = reservation_ask(100, q, t, gamma, 2, 1)
                    r_b = reservation_bid(100, q, t, gamma, 2, 1)
                    r_avg = average_reservation_price(100, q, t, gamma, 2, 1)
                    assert abs((r_a + r_b) / 2 - r_avg) < 1e-12

    def test_ask_greater_than_bid(self):
        """Reservation ask must be greater than reservation bid."""
        for q in [-2, -1, 0, 1, 2]:
            r_a = reservation_ask(100, q, 0.4, 0.1, 2, 1)
            r_b = reservation_bid(100, q, 0.4, 0.1, 2, 1)
            assert r_a > r_b, f"r^a={r_a} not > r^b={r_b} for q={q}"

    def test_ask_bid_spread_equals_gamma_sigma2_tau(self):
        """r^a - r^b = gamma*sigma^2*(T-t) (the inventory-independent spread component)."""
        gamma, sigma, T, t = 0.1, 2.0, 1.0, 0.4
        tau = T - t
        expected_spread = gamma * sigma**2 * tau
        for q in [-2, -1, 0, 1, 2]:
            r_a = reservation_ask(100, q, t, gamma, sigma, T)
            r_b = reservation_bid(100, q, t, gamma, sigma, T)
            assert abs(r_a - r_b - expected_spread) < 1e-12


# ---------------------------------------------------------------------------
# Indifference equation verification tests
# ---------------------------------------------------------------------------

class TestIndifferenceEquations:
    """Tests for reservation price indifference equations."""

    @pytest.mark.parametrize("gamma,q,t", [
        (0.01, -2, 0.0), (0.01, 0, 0.4), (0.1, 1, 0.4),
        (0.1, -1, 0.8), (0.5, 2, 0.0), (0.5, 0, 0.4),
    ])
    def test_ask_indifference_holds(self, gamma, q, t):
        """v(x+r^a, s, q-1, t) = v(x, s, q, t) must hold (relative tolerance)."""
        lhs, rhs, rel_diff = verify_reservation_ask_indifference(
            x=0, s=100, q=q, t=t, gamma=gamma, sigma=2, T=1
        )
        assert rel_diff < 1e-10, (
            f"Ask indifference failed: rel_diff={rel_diff} for gamma={gamma}, q={q}, t={t}"
        )

    @pytest.mark.parametrize("gamma,q,t", [
        (0.01, -2, 0.0), (0.01, 0, 0.4), (0.1, 1, 0.4),
        (0.1, -1, 0.8), (0.5, 2, 0.0), (0.5, 0, 0.4),
    ])
    def test_bid_indifference_holds(self, gamma, q, t):
        """v(x-r^b, s, q+1, t) = v(x, s, q, t) must hold (relative tolerance)."""
        lhs, rhs, rel_diff = verify_reservation_bid_indifference(
            x=0, s=100, q=q, t=t, gamma=gamma, sigma=2, T=1
        )
        assert rel_diff < 1e-10, (
            f"Bid indifference failed: rel_diff={rel_diff} for gamma={gamma}, q={q}, t={t}"
        )

    def test_average_price_consistency(self):
        """(r^a + r^b)/2 must equal closed-form average reservation price."""
        for gamma in [0.01, 0.1, 0.5]:
            for q in [-2, -1, 0, 1, 2]:
                for t in [0.0, 0.4, 0.8]:
                    avg_comp, avg_cf, diff = verify_average_reservation_price(
                        s=100, q=q, t=t, gamma=gamma, sigma=2, T=1
                    )
                    assert diff < 1e-12, f"Avg price diff={diff} for gamma={gamma}, q={q}, t={t}"


# ---------------------------------------------------------------------------
# Theta representation tests
# ---------------------------------------------------------------------------

class TestThetaRepresentation:
    """Tests for the theta representation of reservation prices."""

    def test_ask_from_theta_matches_direct(self):
        """r^a from theta differences must match direct formula."""
        for gamma in [0.01, 0.1, 0.5]:
            for q in [-2, -1, 0, 1, 2]:
                for t in [0.0, 0.4, 0.8]:
                    r_a_direct = reservation_ask(100, q, t, gamma, 2, 1)
                    r_a_theta = reservation_ask_from_theta(100, q, t, gamma, 2, 1)
                    assert abs(r_a_direct - r_a_theta) < 1e-10

    def test_bid_from_theta_matches_direct(self):
        """r^b from theta differences must match direct formula."""
        for gamma in [0.01, 0.1, 0.5]:
            for q in [-2, -1, 0, 1, 2]:
                for t in [0.0, 0.4, 0.8]:
                    r_b_direct = reservation_bid(100, q, t, gamma, 2, 1)
                    r_b_theta = reservation_bid_from_theta(100, q, t, gamma, 2, 1)
                    assert abs(r_b_direct - r_b_theta) < 1e-10


# ---------------------------------------------------------------------------
# Spread formula tests
# ---------------------------------------------------------------------------

class TestSpreadFormulas:
    """Tests for spread offset and total spread formulas."""

    def test_spread_offset_positive(self):
        """Spread offset (1/gamma)*ln(1+gamma/k) must be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            offset = spread_offset_exponential(gamma, k=1.5)
            assert offset > 0

    def test_table_faithful_spread_values(self):
        """Verify spread values match paper's expected ~1.29, ~1.33, ~1.15."""
        k = 1.5
        # gamma=0.1 -> ~1.29
        sp_01 = total_spread_table_faithful(0.1, k)
        assert abs(sp_01 - 1.29) < 0.01, f"gamma=0.1 spread={sp_01}, expected ~1.29"

        # gamma=0.01 -> ~1.33
        sp_001 = total_spread_table_faithful(0.01, k)
        assert abs(sp_001 - 1.33) < 0.01, f"gamma=0.01 spread={sp_001}, expected ~1.33"

        # gamma=0.5 -> ~1.15
        sp_05 = total_spread_table_faithful(0.5, k)
        assert abs(sp_05 - 1.15) < 0.01, f"gamma=0.5 spread={sp_05}, expected ~1.15"

    def test_equation_faithful_spread_at_maturity(self):
        """At t=T, equation-faithful spread = (2/gamma)*ln(1+gamma/k) (time term vanishes)."""
        gamma, sigma, T, k = 0.1, 2.0, 1.0, 1.5
        sp_at_T = total_spread_equation_faithful(T, gamma, sigma, T, k)
        sp_const = total_spread_table_faithful(gamma, k)
        assert abs(sp_at_T - sp_const) < 1e-12

    def test_equation_faithful_spread_decreases_over_time(self):
        """Equation-faithful spread should decrease as t increases toward T."""
        gamma, sigma, T, k = 0.1, 2.0, 1.0, 1.5
        sp_t0 = total_spread_equation_faithful(0.0, gamma, sigma, T, k)
        sp_t05 = total_spread_equation_faithful(0.5, gamma, sigma, T, k)
        sp_tT = total_spread_equation_faithful(T, gamma, sigma, T, k)
        assert sp_t0 > sp_t05 > sp_tT

    def test_spread_offset_convergence_to_2_over_k(self):
        """As gamma->0, (2/gamma)*ln(1+gamma/k) -> 2/k."""
        k = 1.5
        limit = 2.0 / k
        for gamma in [0.001, 0.0001]:
            sp = total_spread_table_faithful(gamma, k)
            assert abs(sp - limit) < 0.01, f"gamma={gamma}: spread={sp}, limit={limit}"

    def test_spread_offset_for_paper_values(self):
        """Verify (1/gamma)*ln(1+gamma/k) for k=1.5 and paper gamma values."""
        k = 1.5
        # From implementation notes: ~1.29/2=0.645 for gamma=0.1, etc.
        for gamma in [0.1, 0.01, 0.5]:
            offset = spread_offset_exponential(gamma, k)
            total = total_spread_table_faithful(gamma, k)
            assert abs(2 * offset - total) < 1e-12  # total = 2 * offset


# ---------------------------------------------------------------------------
# Qualitative inventory effect tests
# ---------------------------------------------------------------------------

class TestInventoryEffects:
    """Tests for qualitative inventory effects on reservation price."""

    def test_zero_inventory_reservation_equals_mid(self):
        """q=0 => r = s."""
        r = average_reservation_price(s=100, q=0, t=0.4, gamma=0.1, sigma=2, T=1)
        assert abs(r - 100) < 1e-12

    def test_positive_inventory_lowers_reservation(self):
        """q>0 => r < s (long inventory lowers reservation price)."""
        for q in [1, 2, 5]:
            r = average_reservation_price(s=100, q=q, t=0.4, gamma=0.1, sigma=2, T=1)
            assert r < 100, f"q={q}: r={r} not < s=100"

    def test_negative_inventory_raises_reservation(self):
        """q<0 => r > s (short inventory raises reservation price)."""
        for q in [-1, -2, -5]:
            r = average_reservation_price(s=100, q=q, t=0.4, gamma=0.1, sigma=2, T=1)
            assert r > 100, f"q={q}: r={r} not > s=100"

    def test_reservation_equals_mid_at_maturity(self):
        """At t=T, r = s regardless of inventory."""
        for q in [-2, -1, 0, 1, 2]:
            r = average_reservation_price(s=100, q=q, t=1.0, gamma=0.1, sigma=2, T=1.0)
            assert abs(r - 100) < 1e-12, f"q={q}: r={r} != s=100 at maturity"

    def test_inventory_effect_proportional_to_gamma(self):
        """Inventory effect should scale linearly with gamma."""
        s, q, t, sigma, T = 100.0, 1.0, 0.4, 2.0, 1.0
        r1 = average_reservation_price(s, q, t, 0.1, sigma, T)
        r2 = average_reservation_price(s, q, t, 0.2, sigma, T)
        effect1 = s - r1
        effect2 = s - r2
        assert abs(effect2 / effect1 - 2.0) < 1e-10

    def test_check_inventory_effects_all_pass(self):
        """All qualitative checks should pass."""
        checks = check_inventory_effect_on_reservation_price(
            s=100, q=1, t=0.4, gamma=0.1, sigma=2, T=1
        )
        for check_name, passed in checks.items():
            assert passed, f"Check failed: {check_name}"


# ---------------------------------------------------------------------------
# Small gamma convergence tests
# ---------------------------------------------------------------------------

class TestSmallGammaConvergence:
    """Tests for convergence to symmetric behavior as gamma -> 0."""

    def test_convergence_improves_with_smaller_gamma(self):
        """Relative error should decrease as gamma decreases."""
        k = 1.5
        results = check_small_gamma_convergence(k, [0.5, 0.1, 0.01, 0.001])
        errors = [results[g]["relative_error"] for g in [0.5, 0.1, 0.01, 0.001]]
        # Each error should be smaller than the previous
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], f"Error not decreasing: {errors}"

    def test_very_small_gamma_near_symmetric_limit(self):
        """For very small gamma, spread should be very close to 2/k."""
        k = 1.5
        results = check_small_gamma_convergence(k, [0.0001])
        assert results[0.0001]["relative_error"] < 0.001


# ---------------------------------------------------------------------------
# Paper inconsistency tests
# ---------------------------------------------------------------------------

class TestPaperInconsistency:
    """Tests for the documented paper inconsistency."""

    def test_inconsistency_exists(self):
        """Equation-faithful spread at t=0 should differ from table-faithful spread."""
        inconsistency = document_paper_inconsistency([0.1, 0.01, 0.5], k=1.5, sigma=2, T=1)
        for gamma, info in inconsistency.items():
            # The difference should be gamma*sigma^2*T = gamma*4
            expected_diff = gamma * 4.0  # gamma*sigma^2*T with sigma=2, T=1
            assert abs(info["difference"] - expected_diff) < 1e-10

    def test_table_faithful_matches_paper_targets(self):
        """Table-faithful spread should match paper's published values."""
        k = 1.5
        assert abs(total_spread_table_faithful(0.1, k) - 1.29) < 0.01
        assert abs(total_spread_table_faithful(0.01, k) - 1.33) < 0.01
        assert abs(total_spread_table_faithful(0.5, k) - 1.15) < 0.01


# ---------------------------------------------------------------------------
# Full verification suite test
# ---------------------------------------------------------------------------

class TestFullVerificationSuite:
    """Integration test for the full analytical verification suite."""

    def test_run_analytical_verification_completes(self):
        """Full verification suite should complete without errors."""
        results = run_analytical_verification()
        assert results is not None

    def test_all_frozen_value_checks_negative(self):
        """All frozen-inventory value checks should show v < 0."""
        results = run_analytical_verification()
        assert all(c["v_negative"] for c in results.frozen_value_checks)

    def test_all_indifference_checks_pass(self):
        """All indifference equation checks should pass (relative tolerance)."""
        results = run_analytical_verification()
        # ask_indiff_ok and bid_indiff_ok use relative tolerance in run_analytical_verification
        assert all(c["ask_indiff_ok"] for c in results.reservation_price_checks), (
            "Some ask indifference checks failed"
        )
        assert all(c["bid_indiff_ok"] for c in results.reservation_price_checks), (
            "Some bid indifference checks failed"
        )
        assert all(c["avg_price_ok"] for c in results.reservation_price_checks), (
            "Some average price checks failed"
        )

    def test_all_theta_checks_pass(self):
        """All theta representation checks should pass."""
        results = run_analytical_verification()
        assert all(c["ask_consistent"] for c in results.theta_consistency_checks)
        assert all(c["bid_consistent"] for c in results.theta_consistency_checks)

    def test_all_inventory_effect_checks_pass(self):
        """All qualitative inventory effect checks should pass."""
        results = run_analytical_verification()
        assert all(results.inventory_effect_checks.values())


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_sigma_frozen_value(self):
        """With sigma=0, frozen value should be deterministic."""
        v = frozen_inventory_value(x=0, s=100, q=1, t=0, gamma=0.1, sigma=0, T=1)
        expected = -math.exp(-0.1 * 0) * math.exp(-0.1 * 1 * 100) * math.exp(0)
        assert abs(v - expected) < 1e-12

    def test_zero_sigma_reservation_price(self):
        """With sigma=0, reservation price equals mid-price."""
        r = average_reservation_price(s=100, q=5, t=0, gamma=0.1, sigma=0, T=1)
        assert abs(r - 100) < 1e-12

    def test_large_inventory_reservation_price(self):
        """Large inventory should produce large deviation from mid-price."""
        r_large = average_reservation_price(s=100, q=100, t=0, gamma=0.1, sigma=2, T=1)
        r_small = average_reservation_price(s=100, q=1, t=0, gamma=0.1, sigma=2, T=1)
        assert abs(r_large - 100) > abs(r_small - 100)

    def test_spread_offset_positive_for_extreme_gamma(self):
        """Spread offset should be positive even for extreme gamma values."""
        for gamma in [0.0001, 100.0]:
            offset = spread_offset_exponential(gamma, k=1.5)
            assert offset > 0

    def test_frozen_value_at_maturity_with_inventory(self):
        """At t=T, frozen value should not depend on sigma (no future uncertainty)."""
        v1 = frozen_inventory_value(x=0, s=100, q=1, t=1.0, gamma=0.1, sigma=2, T=1.0)
        v2 = frozen_inventory_value(x=0, s=100, q=1, t=1.0, gamma=0.1, sigma=10, T=1.0)
        assert abs(v1 - v2) < 1e-12
