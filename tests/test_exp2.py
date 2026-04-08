"""
Tests for Experiment 2: Monte Carlo with equation-faithful spread.

Verifies:
- Simulation parameter validation
- Reservation price formula
- Equation-faithful spread formula
- Execution intensity formula
- Single-path simulation properties
- Monte Carlo statistical properties
- Variance reduction of inventory strategy
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp2_mc_eq_spread import (
    SimParams,
    validate_sim_params,
    reservation_price,
    equation_faithful_spread,
    execution_intensity,
    simulate_path_inventory,
    simulate_path_symmetric,
    run_monte_carlo,
    DiagnosticsLog,
)


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestSimParamsValidation:
    """Tests for simulation parameter validation."""

    def test_valid_params(self):
        """Valid parameters should not raise."""
        p = SimParams(gamma=0.1)
        validate_sim_params(p)

    def test_gamma_zero_raises(self):
        """gamma=0 should raise ValueError."""
        p = SimParams(gamma=0.0)
        with pytest.raises(ValueError, match="gamma"):
            validate_sim_params(p)

    def test_sigma_negative_raises(self):
        """Negative sigma should raise ValueError."""
        p = SimParams(sigma=-1.0)
        with pytest.raises(ValueError, match="sigma"):
            validate_sim_params(p)

    def test_N_dt_equals_T(self):
        """N*dt must equal T."""
        p = SimParams(N=200, dt=0.005, T=1.0)
        validate_sim_params(p)  # Should not raise

    def test_N_dt_not_T_raises(self):
        """N*dt != T should raise AssertionError."""
        p = SimParams(N=100, dt=0.005, T=1.0)  # 100*0.005=0.5 != 1.0
        with pytest.raises(AssertionError):
            validate_sim_params(p)


# ---------------------------------------------------------------------------
# Formula tests
# ---------------------------------------------------------------------------

class TestFormulas:
    """Tests for core formula functions."""

    def test_reservation_price_zero_inventory(self):
        """q=0 => r = s."""
        r = reservation_price(s=100, q=0, t=0.4, gamma=0.1, sigma=2, T=1)
        assert abs(r - 100) < 1e-12

    def test_reservation_price_positive_inventory(self):
        """q>0 => r < s."""
        r = reservation_price(s=100, q=1, t=0.4, gamma=0.1, sigma=2, T=1)
        assert r < 100

    def test_reservation_price_at_maturity(self):
        """At t=T, r = s regardless of inventory."""
        r = reservation_price(s=100, q=5, t=1.0, gamma=0.1, sigma=2, T=1.0)
        assert abs(r - 100) < 1e-12

    def test_equation_faithful_spread_at_maturity(self):
        """At t=T, spread = (2/gamma)*ln(1+gamma/k)."""
        gamma, sigma, T, k = 0.1, 2.0, 1.0, 1.5
        sp = equation_faithful_spread(T, gamma, sigma, T, k)
        expected = (2.0 / gamma) * math.log(1.0 + gamma / k)
        assert abs(sp - expected) < 1e-12

    def test_equation_faithful_spread_decreases_over_time(self):
        """Spread should decrease as t increases."""
        gamma, sigma, T, k = 0.1, 2.0, 1.0, 1.5
        sp_t0 = equation_faithful_spread(0.0, gamma, sigma, T, k)
        sp_t05 = equation_faithful_spread(0.5, gamma, sigma, T, k)
        sp_tT = equation_faithful_spread(T, gamma, sigma, T, k)
        assert sp_t0 > sp_t05 > sp_tT

    def test_execution_intensity_decreases_with_delta(self):
        """Intensity should decrease as delta increases."""
        A, k = 140.0, 1.5
        lam1 = execution_intensity(0.5, A, k)
        lam2 = execution_intensity(1.0, A, k)
        lam3 = execution_intensity(2.0, A, k)
        assert lam1 > lam2 > lam3

    def test_execution_intensity_at_zero_delta(self):
        """At delta=0, intensity = A."""
        A, k = 140.0, 1.5
        lam = execution_intensity(0.0, A, k)
        assert abs(lam - A) < 1e-12

    def test_execution_intensity_positive(self):
        """Intensity must always be positive."""
        for delta in [-1.0, 0.0, 0.5, 1.0, 5.0]:
            lam = execution_intensity(delta, 140.0, 1.5)
            assert lam > 0


# ---------------------------------------------------------------------------
# Single-path simulation tests
# ---------------------------------------------------------------------------

class TestSinglePathSimulation:
    """Tests for single-path simulation properties."""

    def _make_path_inputs(self, N: int = 200, seed: int = 42):
        """Create deterministic path inputs for testing."""
        rng = np.random.default_rng(seed)
        price_innovations = rng.choice([-1.0, 1.0], size=N)
        ask_uniforms = rng.uniform(0, 1, size=N)
        bid_uniforms = rng.uniform(0, 1, size=N)
        return price_innovations, ask_uniforms, bid_uniforms

    def test_inventory_path_returns_correct_shapes(self):
        """Inventory path should return arrays of correct shape."""
        p = SimParams(gamma=0.1)
        price_inn, ask_u, bid_u = self._make_path_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_inventory(
            p, price_inn, ask_u, bid_u, diag
        )
        assert len(S_path) == p.N + 1
        assert len(r_path) == p.N + 1
        assert len(pa_path) == p.N + 1
        assert len(pb_path) == p.N + 1
        assert len(q_path) == p.N + 1

    def test_symmetric_path_returns_correct_shapes(self):
        """Symmetric path should return arrays of correct shape."""
        p = SimParams(gamma=0.1)
        price_inn, ask_u, bid_u = self._make_path_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_symmetric(
            p, price_inn, ask_u, bid_u, diag
        )
        assert len(S_path) == p.N + 1

    def test_initial_conditions_correct(self):
        """Initial state should match SimParams."""
        p = SimParams(gamma=0.1, s0=100.0, q0=0, X0=0.0)
        price_inn, ask_u, bid_u = self._make_path_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_inventory(
            p, price_inn, ask_u, bid_u, diag
        )
        assert abs(S_path[0] - 100.0) < 1e-12
        assert q_path[0] == 0

    def test_reservation_price_at_maturity_equals_mid(self):
        """At t=T, reservation price should equal mid-price."""
        p = SimParams(gamma=0.1)
        price_inn, ask_u, bid_u = self._make_path_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_inventory(
            p, price_inn, ask_u, bid_u, diag
        )
        # At t=T, r = S - q*gamma*sigma^2*0 = S
        assert abs(r_path[-1] - S_path[-1]) < 1e-10

    def test_symmetric_strategy_centered_at_mid(self):
        """Symmetric strategy: ask and bid should be equidistant from mid-price."""
        p = SimParams(gamma=0.1)
        price_inn, ask_u, bid_u = self._make_path_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_symmetric(
            p, price_inn, ask_u, bid_u, diag
        )
        # For symmetric strategy, pa - S = S - pb at each step
        for n in range(p.N):
            delta_a = pa_path[n] - S_path[n]
            delta_b = S_path[n] - pb_path[n]
            assert abs(delta_a - delta_b) < 1e-10, f"Step {n}: delta_a={delta_a}, delta_b={delta_b}"

    def test_inventory_integer_valued(self):
        """Inventory must always be integer-valued."""
        p = SimParams(gamma=0.1)
        price_inn, ask_u, bid_u = self._make_path_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_inventory(
            p, price_inn, ask_u, bid_u, diag
        )
        for q in q_path:
            assert q == int(q), f"Inventory {q} is not integer"

    def test_diagnostics_total_steps(self):
        """Diagnostics should count total steps correctly."""
        p = SimParams(gamma=0.1)
        price_inn, ask_u, bid_u = self._make_path_inputs()
        diag = DiagnosticsLog()
        simulate_path_inventory(p, price_inn, ask_u, bid_u, diag)
        assert diag.total_steps == p.N


# ---------------------------------------------------------------------------
# Monte Carlo statistical tests
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    """Tests for Monte Carlo simulation statistical properties."""

    def test_run_monte_carlo_completes(self):
        """Monte Carlo should complete without errors."""
        p = SimParams(gamma=0.1, n_paths=100, seed=42)
        results = run_monte_carlo(p)
        assert "inventory" in results
        assert "symmetric" in results

    def test_profits_array_length(self):
        """Profits array should have n_paths elements."""
        p = SimParams(gamma=0.1, n_paths=100, seed=42)
        results = run_monte_carlo(p)
        assert len(results["inventory"].profits) == 100
        assert len(results["symmetric"].profits) == 100

    def test_inventory_strategy_lower_profit_std(self):
        """Inventory strategy should have lower profit std than symmetric."""
        p = SimParams(gamma=0.5, n_paths=500, seed=42)
        results = run_monte_carlo(p)
        inv_std = results["inventory"].std_profit
        sym_std = results["symmetric"].std_profit
        assert inv_std < sym_std, f"Inventory std={inv_std} not < symmetric std={sym_std}"

    def test_inventory_strategy_lower_inventory_std(self):
        """Inventory strategy should have lower inventory std than symmetric."""
        p = SimParams(gamma=0.5, n_paths=500, seed=42)
        results = run_monte_carlo(p)
        inv_std = results["inventory"].std_inventory
        sym_std = results["symmetric"].std_inventory
        assert inv_std < sym_std, f"Inventory std_q={inv_std} not < symmetric std_q={sym_std}"

    def test_strategies_converge_at_low_gamma(self):
        """At low gamma, inventory and symmetric strategies should be more similar."""
        p_low = SimParams(gamma=0.01, n_paths=500, seed=42)
        p_high = SimParams(gamma=0.5, n_paths=500, seed=42)

        results_low = run_monte_carlo(p_low)
        results_high = run_monte_carlo(p_high)

        # Relative difference in std should be smaller for low gamma
        rel_diff_low = abs(
            results_low["inventory"].std_profit - results_low["symmetric"].std_profit
        ) / results_low["symmetric"].std_profit

        rel_diff_high = abs(
            results_high["inventory"].std_profit - results_high["symmetric"].std_profit
        ) / results_high["symmetric"].std_profit

        assert rel_diff_low < rel_diff_high, (
            f"Low gamma rel_diff={rel_diff_low} not < high gamma rel_diff={rel_diff_high}"
        )

    def test_sample_path_stored(self):
        """Sample path should be stored for inventory strategy."""
        p = SimParams(gamma=0.1, n_paths=10, seed=42)
        results = run_monte_carlo(p)
        assert results["inventory"].sample_path is not None
        assert "S" in results["inventory"].sample_path
        assert "r" in results["inventory"].sample_path

    def test_reproducibility_with_same_seed(self):
        """Same seed should produce identical results."""
        p1 = SimParams(gamma=0.1, n_paths=50, seed=123)
        p2 = SimParams(gamma=0.1, n_paths=50, seed=123)
        r1 = run_monte_carlo(p1)
        r2 = run_monte_carlo(p2)
        np.testing.assert_array_equal(r1["inventory"].profits, r2["inventory"].profits)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different results."""
        p1 = SimParams(gamma=0.1, n_paths=50, seed=1)
        p2 = SimParams(gamma=0.1, n_paths=50, seed=2)
        r1 = run_monte_carlo(p1)
        r2 = run_monte_carlo(p2)
        assert not np.array_equal(r1["inventory"].profits, r2["inventory"].profits)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases in Monte Carlo simulation."""

    def test_single_path(self):
        """Single path simulation should work."""
        p = SimParams(gamma=0.1, n_paths=1, seed=42)
        results = run_monte_carlo(p)
        assert len(results["inventory"].profits) == 1

    def test_high_gamma_strong_inventory_control(self):
        """High gamma should show stronger inventory control."""
        p = SimParams(gamma=0.5, n_paths=200, seed=42)
        results = run_monte_carlo(p)
        # Inventory strategy should have much lower inventory std
        inv_std = results["inventory"].std_inventory
        sym_std = results["symmetric"].std_inventory
        assert inv_std < sym_std * 0.8  # at least 20% reduction

    def test_terminal_profit_finite(self):
        """Terminal profits should be finite."""
        p = SimParams(gamma=0.1, n_paths=100, seed=42)
        results = run_monte_carlo(p)
        assert np.all(np.isfinite(results["inventory"].profits))
        assert np.all(np.isfinite(results["symmetric"].profits))
