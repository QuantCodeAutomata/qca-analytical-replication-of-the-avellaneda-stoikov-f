"""
Tests for Experiment 4: Infinite-horizon analytical extension.

Verifies:
- Admissibility condition formula
- Suggested omega formula
- Stationary reservation price
- Stationary spread
- Numerical solver convergence
- Conceptual comparison with finite horizon
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp4_infinite_horizon import (
    validate_infinite_horizon_params,
    admissibility_condition,
    suggested_omega,
    check_admissibility,
    stationary_theta_approximation,
    stationary_reservation_price,
    stationary_spread_offset,
    stationary_total_spread,
    solve_stationary_bellman,
    compute_stationary_quotes,
    compare_finite_infinite_horizon,
    run_experiment_4,
)


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    """Tests for infinite-horizon parameter validation."""

    def test_valid_params(self):
        """Valid parameters should not raise."""
        validate_infinite_horizon_params(gamma=0.1, sigma=2.0, omega=0.1, q_max=5)

    def test_gamma_zero_raises(self):
        """gamma=0 should raise ValueError."""
        with pytest.raises(ValueError, match="gamma"):
            validate_infinite_horizon_params(gamma=0.0, sigma=2.0, omega=0.1, q_max=5)

    def test_sigma_negative_raises(self):
        """Negative sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            validate_infinite_horizon_params(gamma=0.1, sigma=-1.0, omega=0.1, q_max=5)

    def test_omega_zero_raises(self):
        """omega=0 should raise ValueError."""
        with pytest.raises(ValueError, match="omega"):
            validate_infinite_horizon_params(gamma=0.1, sigma=2.0, omega=0.0, q_max=5)

    def test_q_max_zero_raises(self):
        """q_max=0 should raise ValueError."""
        with pytest.raises(ValueError, match="q_max"):
            validate_infinite_horizon_params(gamma=0.1, sigma=2.0, omega=0.1, q_max=0)


# ---------------------------------------------------------------------------
# Admissibility condition tests
# ---------------------------------------------------------------------------

class TestAdmissibilityCondition:
    """Tests for the omega admissibility condition."""

    def test_admissibility_formula(self):
        """Test exact formula: 0.5*gamma^2*sigma^2*q^2."""
        gamma, sigma, q = 0.1, 2.0, 3.0
        expected = 0.5 * gamma**2 * sigma**2 * q**2
        result = admissibility_condition(gamma, sigma, q)
        assert abs(result - expected) < 1e-12

    def test_admissibility_zero_at_zero_inventory(self):
        """At q=0, minimum omega required is 0."""
        result = admissibility_condition(0.1, 2.0, 0.0)
        assert abs(result) < 1e-12

    def test_admissibility_symmetric_in_inventory(self):
        """Condition should be symmetric: same for q and -q."""
        gamma, sigma = 0.1, 2.0
        for q in [1, 2, 3, 5]:
            assert abs(
                admissibility_condition(gamma, sigma, q) -
                admissibility_condition(gamma, sigma, -q)
            ) < 1e-12

    def test_admissibility_increases_with_q(self):
        """Minimum omega should increase with |q|."""
        gamma, sigma = 0.1, 2.0
        cond_1 = admissibility_condition(gamma, sigma, 1)
        cond_2 = admissibility_condition(gamma, sigma, 2)
        cond_3 = admissibility_condition(gamma, sigma, 3)
        assert cond_1 < cond_2 < cond_3

    def test_admissibility_increases_with_gamma(self):
        """Minimum omega should increase with gamma."""
        sigma, q = 2.0, 2.0
        cond_small = admissibility_condition(0.01, sigma, q)
        cond_large = admissibility_condition(0.5, sigma, q)
        assert cond_small < cond_large


# ---------------------------------------------------------------------------
# Suggested omega tests
# ---------------------------------------------------------------------------

class TestSuggestedOmega:
    """Tests for the suggested omega formula."""

    def test_suggested_omega_formula(self):
        """Test exact formula: 0.5*gamma^2*sigma^2*(q_max+1)^2."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        expected = 0.5 * gamma**2 * sigma**2 * (q_max + 1)**2
        result = suggested_omega(gamma, sigma, q_max)
        assert abs(result - expected) < 1e-12

    def test_suggested_omega_satisfies_admissibility(self):
        """Suggested omega should satisfy admissibility for all q in [-q_max, q_max]."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        omega = suggested_omega(gamma, sigma, q_max)
        for q in range(-q_max, q_max + 1):
            min_omega = admissibility_condition(gamma, sigma, q)
            assert omega > min_omega, f"omega={omega} not > min_omega={min_omega} for q={q}"

    def test_suggested_omega_tight_at_boundary(self):
        """Suggested omega should be just above the boundary condition at q=q_max."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        omega = suggested_omega(gamma, sigma, q_max)
        min_at_boundary = admissibility_condition(gamma, sigma, q_max)
        # omega = 0.5*gamma^2*sigma^2*(q_max+1)^2 > 0.5*gamma^2*sigma^2*q_max^2
        assert omega > min_at_boundary

    def test_check_admissibility_all_satisfied(self):
        """check_admissibility should show all satisfied with suggested omega."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        omega = suggested_omega(gamma, sigma, q_max)
        checks = check_admissibility(gamma, sigma, omega, q_max)
        for q, info in checks.items():
            assert info["admissible"], f"q={q}: not admissible"

    def test_check_admissibility_fails_with_small_omega(self):
        """With omega too small, some inventory levels should fail admissibility."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        omega_too_small = 0.001  # much too small
        checks = check_admissibility(gamma, sigma, omega_too_small, q_max)
        # At least some q values should fail
        any_failed = any(not info["admissible"] for info in checks.values())
        assert any_failed


# ---------------------------------------------------------------------------
# Stationary formula tests
# ---------------------------------------------------------------------------

class TestStationaryFormulas:
    """Tests for stationary infinite-horizon formulas."""

    def test_stationary_reservation_price_zero_inventory(self):
        """q=0 => stationary r = s."""
        r = stationary_reservation_price(s=100, q=0, gamma=0.1, sigma=2, omega=0.1)
        assert abs(r - 100) < 1e-12

    def test_stationary_reservation_price_positive_inventory(self):
        """q>0 => stationary r < s."""
        r = stationary_reservation_price(s=100, q=1, gamma=0.1, sigma=2, omega=0.1)
        assert r < 100

    def test_stationary_reservation_price_negative_inventory(self):
        """q<0 => stationary r > s."""
        r = stationary_reservation_price(s=100, q=-1, gamma=0.1, sigma=2, omega=0.1)
        assert r > 100

    def test_stationary_spread_offset_positive(self):
        """Stationary spread offset must be positive."""
        offset = stationary_spread_offset(gamma=0.1, k=1.5)
        assert offset > 0

    def test_stationary_total_spread_positive(self):
        """Stationary total spread must be positive."""
        sp = stationary_total_spread(gamma=0.1, sigma=2, omega=0.1, k=1.5)
        assert sp > 0

    def test_stationary_spread_larger_with_smaller_omega(self):
        """Smaller omega (less discounting) should give larger stationary spread."""
        sp_small_omega = stationary_total_spread(0.1, 2, 0.01, 1.5)
        sp_large_omega = stationary_total_spread(0.1, 2, 1.0, 1.5)
        assert sp_small_omega > sp_large_omega

    def test_stationary_theta_zero_at_zero_inventory(self):
        """theta(q=0) should be 0."""
        theta = stationary_theta_approximation(q=0, gamma=0.1, sigma=2, omega=0.1, k=1.5, A=140)
        assert abs(theta) < 1e-12

    def test_stationary_theta_symmetric(self):
        """theta(q) should equal theta(-q) (symmetric in inventory)."""
        for q in [1, 2, 3]:
            theta_pos = stationary_theta_approximation(q, 0.1, 2, 0.1, 1.5, 140)
            theta_neg = stationary_theta_approximation(-q, 0.1, 2, 0.1, 1.5, 140)
            assert abs(theta_pos - theta_neg) < 1e-12


# ---------------------------------------------------------------------------
# Numerical solver tests
# ---------------------------------------------------------------------------

class TestNumericalSolver:
    """Tests for the stationary Bellman solver."""

    def test_solver_runs(self):
        """Solver should run without errors."""
        gamma, sigma, q_max = 0.1, 2.0, 3
        omega = suggested_omega(gamma, sigma, q_max)
        q_grid, w, converged = solve_stationary_bellman(
            gamma, sigma, omega, k=1.5, A=140, q_max=q_max
        )
        assert len(q_grid) == 2 * q_max + 1
        assert len(w) == 2 * q_max + 1

    def test_solver_w_correct_length(self):
        """w values array should have correct length 2*q_max+1."""
        gamma, sigma, q_max = 0.1, 2.0, 3
        omega = suggested_omega(gamma, sigma, q_max)
        q_grid, w, converged = solve_stationary_bellman(
            gamma, sigma, omega, k=1.5, A=140, q_max=q_max
        )
        assert len(w) == 2 * q_max + 1

    def test_solver_w_finite(self):
        """w values should be finite (no NaN or Inf)."""
        gamma, sigma, q_max = 0.1, 2.0, 3
        omega = suggested_omega(gamma, sigma, q_max)
        q_grid, w, converged = solve_stationary_bellman(
            gamma, sigma, omega, k=1.5, A=140, q_max=q_max
        )
        assert np.all(np.isfinite(w))

    def test_solver_w_symmetric(self):
        """w should be approximately symmetric: w(q) ≈ w(-q) in absolute value."""
        gamma, sigma, q_max = 0.1, 2.0, 3
        omega = suggested_omega(gamma, sigma, q_max)
        q_grid, w, converged = solve_stationary_bellman(
            gamma, sigma, omega, k=1.5, A=140, q_max=q_max
        )
        for q in range(1, q_max + 1):
            idx_pos = np.where(q_grid == q)[0][0]
            idx_neg = np.where(q_grid == -q)[0][0]
            # Should be approximately symmetric in absolute value
            assert abs(abs(w[idx_pos]) - abs(w[idx_neg])) < 0.1

    def test_compute_stationary_quotes_runs(self):
        """compute_stationary_quotes should run without errors."""
        gamma, sigma, q_max = 0.1, 2.0, 3
        omega = suggested_omega(gamma, sigma, q_max)
        q_grid, w, converged = solve_stationary_bellman(
            gamma, sigma, omega, k=1.5, A=140, q_max=q_max
        )
        delta_ask, delta_bid, res_prices = compute_stationary_quotes(
            q_grid, w, 100.0, gamma, k=1.5, A=140
        )
        assert len(delta_ask) == len(q_grid)
        assert len(delta_bid) == len(q_grid)


# ---------------------------------------------------------------------------
# Finite vs infinite horizon comparison tests
# ---------------------------------------------------------------------------

class TestFiniteInfiniteComparison:
    """Tests for conceptual comparison between finite and infinite horizon."""

    def test_finite_effect_zero_at_maturity(self):
        """Finite-horizon inventory effect should be 0 at t=T."""
        comparison = compare_finite_infinite_horizon(0.1, 2.0, 1.5, q_max=5, T=1.0)
        assert abs(comparison["finite_horizon_effects"][1.0]) < 1e-12

    def test_infinite_effect_nonzero(self):
        """Infinite-horizon inventory effect should be nonzero."""
        comparison = compare_finite_infinite_horizon(0.1, 2.0, 1.5, q_max=5)
        assert comparison["infinite_horizon_effect"] > 0

    def test_finite_effect_decreases_over_time(self):
        """Finite-horizon effect should decrease as t increases."""
        comparison = compare_finite_infinite_horizon(0.1, 2.0, 1.5, q_max=5, T=1.0)
        effects = comparison["finite_horizon_effects"]
        t_values = sorted(effects.keys())
        for i in range(len(t_values) - 1):
            assert effects[t_values[i]] >= effects[t_values[i + 1]]

    def test_omega_satisfies_admissibility(self):
        """Suggested omega should satisfy admissibility."""
        comparison = compare_finite_infinite_horizon(0.1, 2.0, 1.5, q_max=5)
        gamma, sigma, omega, q_max = 0.1, 2.0, comparison["omega"], 5
        for q in range(-q_max, q_max + 1):
            min_omega = admissibility_condition(gamma, sigma, q)
            assert omega > min_omega


# ---------------------------------------------------------------------------
# Full experiment test
# ---------------------------------------------------------------------------

class TestFullExperiment:
    """Integration test for the full infinite-horizon experiment."""

    def test_run_experiment_4_completes(self):
        """Full experiment should complete without errors."""
        results = run_experiment_4(
            gamma_values=[0.1],
            sigma=2.0,
            k=1.5,
            A=140.0,
            q_max=3,
            output_dir="/tmp/test_exp4_results",
        )
        assert 0.1 in results

    def test_all_admissibility_satisfied(self):
        """All admissibility conditions should be satisfied."""
        results = run_experiment_4(
            gamma_values=[0.1, 0.5],
            sigma=2.0,
            k=1.5,
            A=140.0,
            q_max=3,
            output_dir="/tmp/test_exp4_results",
        )
        for gamma, res in results.items():
            assert res.omega_condition_satisfied, f"gamma={gamma}: admissibility not satisfied"

    def test_omega_increases_with_gamma(self):
        """Suggested omega should increase with gamma (for fixed sigma, q_max)."""
        results = run_experiment_4(
            gamma_values=[0.01, 0.1, 0.5],
            sigma=2.0,
            k=1.5,
            A=140.0,
            q_max=3,
            output_dir="/tmp/test_exp4_results",
        )
        omegas = [results[g].omega for g in [0.01, 0.1, 0.5]]
        assert omegas[0] < omegas[1] < omegas[2]
