"""
Tests for Experiment 3: Monte Carlo with table-faithful constant spread.

Verifies:
- Constant spread formula matches paper targets
- Simulation properties with constant spread
- Variance reduction of inventory strategy
- Comparison with paper published values
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp3_mc_const_spread import (
    SimParams,
    validate_sim_params,
    reservation_price,
    constant_spread,
    execution_intensity,
    simulate_path_inventory,
    simulate_path_symmetric,
    run_monte_carlo,
    DiagnosticsLog,
    PAPER_TARGETS,
)


# ---------------------------------------------------------------------------
# Constant spread formula tests
# ---------------------------------------------------------------------------

class TestConstantSpreadFormula:
    """Tests for the constant spread formula."""

    def test_spread_gamma_01(self):
        """gamma=0.1, k=1.5 should give spread ~1.29."""
        sp = constant_spread(0.1, 1.5)
        assert abs(sp - 1.29) < 0.01, f"spread={sp}, expected ~1.29"

    def test_spread_gamma_001(self):
        """gamma=0.01, k=1.5 should give spread ~1.33."""
        sp = constant_spread(0.01, 1.5)
        assert abs(sp - 1.33) < 0.01, f"spread={sp}, expected ~1.33"

    def test_spread_gamma_05(self):
        """gamma=0.5, k=1.5 should give spread ~1.15."""
        sp = constant_spread(0.5, 1.5)
        assert abs(sp - 1.15) < 0.01, f"spread={sp}, expected ~1.15"

    def test_spread_positive(self):
        """Spread must always be positive."""
        for gamma in [0.001, 0.01, 0.1, 0.5, 1.0]:
            sp = constant_spread(gamma, 1.5)
            assert sp > 0

    def test_spread_decreases_with_gamma(self):
        """Spread should decrease as gamma increases (for fixed k)."""
        k = 1.5
        sp_small = constant_spread(0.01, k)
        sp_medium = constant_spread(0.1, k)
        sp_large = constant_spread(0.5, k)
        assert sp_small > sp_medium > sp_large

    def test_spread_converges_to_2_over_k(self):
        """As gamma->0, spread -> 2/k."""
        k = 1.5
        limit = 2.0 / k
        sp = constant_spread(0.0001, k)
        assert abs(sp - limit) < 0.01

    def test_spread_formula_exact(self):
        """Verify exact formula: (2/gamma)*ln(1+gamma/k)."""
        gamma, k = 0.1, 1.5
        expected = (2.0 / gamma) * math.log(1.0 + gamma / k)
        sp = constant_spread(gamma, k)
        assert abs(sp - expected) < 1e-12

    def test_spread_matches_paper_targets(self):
        """Spread should match all paper target values."""
        for gamma in [0.1, 0.01, 0.5]:
            sp = constant_spread(gamma, 1.5)
            target = PAPER_TARGETS[gamma]["inventory"]["spread"]
            assert abs(sp - target) < 0.01, f"gamma={gamma}: spread={sp}, target={target}"


# ---------------------------------------------------------------------------
# Reservation price tests (same as exp2 but verify in exp3 context)
# ---------------------------------------------------------------------------

class TestReservationPrice:
    """Tests for reservation price in exp3 context."""

    def test_zero_inventory_equals_mid(self):
        """q=0 => r = s."""
        r = reservation_price(s=100, q=0, t=0.4, gamma=0.1, sigma=2, T=1)
        assert abs(r - 100) < 1e-12

    def test_positive_inventory_below_mid(self):
        """q>0 => r < s."""
        r = reservation_price(s=100, q=2, t=0.4, gamma=0.1, sigma=2, T=1)
        assert r < 100

    def test_negative_inventory_above_mid(self):
        """q<0 => r > s."""
        r = reservation_price(s=100, q=-2, t=0.4, gamma=0.1, sigma=2, T=1)
        assert r > 100

    def test_at_maturity_equals_mid(self):
        """At t=T, r = s."""
        r = reservation_price(s=100, q=5, t=1.0, gamma=0.1, sigma=2, T=1.0)
        assert abs(r - 100) < 1e-12


# ---------------------------------------------------------------------------
# Single-path simulation tests
# ---------------------------------------------------------------------------

class TestSinglePathSimulation:
    """Tests for single-path simulation with constant spread."""

    def _make_inputs(self, N: int = 200, seed: int = 42):
        rng = np.random.default_rng(seed)
        price_inn = rng.choice([-1.0, 1.0], size=N)
        ask_u = rng.uniform(0, 1, size=N)
        bid_u = rng.uniform(0, 1, size=N)
        return price_inn, ask_u, bid_u

    def test_inventory_path_shapes(self):
        """Path arrays should have correct shapes."""
        p = SimParams(gamma=0.1)
        sp = constant_spread(p.gamma, p.k)
        price_inn, ask_u, bid_u = self._make_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_inventory(
            p, sp, price_inn, ask_u, bid_u, diag
        )
        assert len(S_path) == p.N + 1
        assert len(q_path) == p.N + 1

    def test_symmetric_path_equidistant_quotes(self):
        """Symmetric strategy: ask and bid equidistant from mid-price."""
        p = SimParams(gamma=0.1)
        sp = constant_spread(p.gamma, p.k)
        price_inn, ask_u, bid_u = self._make_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_symmetric(
            p, sp, price_inn, ask_u, bid_u, diag
        )
        for n in range(p.N):
            delta_a = pa_path[n] - S_path[n]
            delta_b = S_path[n] - pb_path[n]
            assert abs(delta_a - delta_b) < 1e-10

    def test_symmetric_delta_equals_half_spread(self):
        """Symmetric strategy: delta = spread/2 at every step."""
        p = SimParams(gamma=0.1)
        sp = constant_spread(p.gamma, p.k)
        price_inn, ask_u, bid_u = self._make_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_symmetric(
            p, sp, price_inn, ask_u, bid_u, diag
        )
        half_sp = sp / 2.0
        for n in range(p.N):
            delta_a = pa_path[n] - S_path[n]
            assert abs(delta_a - half_sp) < 1e-10

    def test_inventory_quotes_shift_with_inventory(self):
        """Inventory strategy: quotes should shift based on inventory."""
        p = SimParams(gamma=0.5)  # high gamma for strong effect
        sp = constant_spread(p.gamma, p.k)
        # Use all-ask fills to build up inventory
        price_inn = np.ones(p.N)  # price always goes up
        ask_u = np.zeros(p.N)  # always fill ask (sell)
        bid_u = np.ones(p.N)   # never fill bid
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_inventory(
            p, sp, price_inn, ask_u, bid_u, diag
        )
        # After selling, inventory should be negative
        # Reservation price should be above mid-price
        # (short inventory raises reservation price)
        if q_T < 0:
            # r_T should be above S_T (but at T, r=S, so check earlier)
            mid_idx = p.N // 2
            r_mid = r_path[mid_idx]
            S_mid = S_path[mid_idx]
            q_mid = q_path[mid_idx]
            if q_mid < 0:
                assert r_mid > S_mid - 1e-10  # r > S for short inventory

    def test_inventory_integer_valued(self):
        """Inventory must always be integer-valued."""
        p = SimParams(gamma=0.1)
        sp = constant_spread(p.gamma, p.k)
        price_inn, ask_u, bid_u = self._make_inputs()
        diag = DiagnosticsLog()
        S_path, r_path, pa_path, pb_path, q_path, X_T, q_T = simulate_path_inventory(
            p, sp, price_inn, ask_u, bid_u, diag
        )
        for q in q_path:
            assert q == int(q)


# ---------------------------------------------------------------------------
# Monte Carlo statistical tests
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    """Tests for Monte Carlo simulation with constant spread."""

    def test_run_completes(self):
        """Monte Carlo should complete without errors."""
        p = SimParams(gamma=0.1, n_paths=100, seed=42)
        results = run_monte_carlo(p)
        assert "inventory" in results
        assert "symmetric" in results

    def test_spread_matches_paper_target(self):
        """Spread in results should match paper target."""
        p = SimParams(gamma=0.1, n_paths=100, seed=42)
        results = run_monte_carlo(p)
        target = PAPER_TARGETS[0.1]["inventory"]["spread"]
        assert abs(results["inventory"].spread - target) < 0.01

    def test_inventory_lower_profit_std(self):
        """Inventory strategy should have lower profit std."""
        p = SimParams(gamma=0.5, n_paths=500, seed=42)
        results = run_monte_carlo(p)
        assert results["inventory"].std_profit < results["symmetric"].std_profit

    def test_inventory_lower_inventory_std(self):
        """Inventory strategy should have lower inventory std."""
        p = SimParams(gamma=0.5, n_paths=500, seed=42)
        results = run_monte_carlo(p)
        assert results["inventory"].std_inventory < results["symmetric"].std_inventory

    def test_strategies_converge_at_low_gamma(self):
        """At low gamma, strategies should be more similar."""
        p_low = SimParams(gamma=0.01, n_paths=500, seed=42)
        p_high = SimParams(gamma=0.5, n_paths=500, seed=42)

        r_low = run_monte_carlo(p_low)
        r_high = run_monte_carlo(p_high)

        rel_diff_low = abs(
            r_low["inventory"].std_profit - r_low["symmetric"].std_profit
        ) / r_low["symmetric"].std_profit

        rel_diff_high = abs(
            r_high["inventory"].std_profit - r_high["symmetric"].std_profit
        ) / r_high["symmetric"].std_profit

        assert rel_diff_low < rel_diff_high

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        p1 = SimParams(gamma=0.1, n_paths=50, seed=99)
        p2 = SimParams(gamma=0.1, n_paths=50, seed=99)
        r1 = run_monte_carlo(p1)
        r2 = run_monte_carlo(p2)
        np.testing.assert_array_equal(r1["inventory"].profits, r2["inventory"].profits)

    def test_profits_finite(self):
        """All profits should be finite."""
        p = SimParams(gamma=0.1, n_paths=100, seed=42)
        results = run_monte_carlo(p)
        assert np.all(np.isfinite(results["inventory"].profits))
        assert np.all(np.isfinite(results["symmetric"].profits))

    def test_gamma_01_spread_matches_paper(self):
        """For gamma=0.1, spread should match paper's 1.29."""
        p = SimParams(gamma=0.1, n_paths=10, seed=42)
        results = run_monte_carlo(p)
        assert abs(results["inventory"].spread - 1.29) < 0.01

    def test_gamma_001_spread_matches_paper(self):
        """For gamma=0.01, spread should match paper's 1.33."""
        p = SimParams(gamma=0.01, n_paths=10, seed=42)
        results = run_monte_carlo(p)
        assert abs(results["inventory"].spread - 1.33) < 0.01

    def test_gamma_05_spread_matches_paper(self):
        """For gamma=0.5, spread should match paper's 1.15."""
        p = SimParams(gamma=0.5, n_paths=10, seed=42)
        results = run_monte_carlo(p)
        assert abs(results["inventory"].spread - 1.15) < 0.01


# ---------------------------------------------------------------------------
# Comparison with paper targets
# ---------------------------------------------------------------------------

class TestPaperTargetComparison:
    """Tests comparing results with paper's published values."""

    def test_gamma_01_inventory_profit_order_of_magnitude(self):
        """For gamma=0.1, inventory mean profit should be in reasonable range."""
        p = SimParams(gamma=0.1, n_paths=1000, seed=42)
        results = run_monte_carlo(p)
        inv = results["inventory"]
        # Paper reports ~62.94; allow wide tolerance due to RNG differences
        assert 30 < inv.mean_profit < 120, f"Mean profit {inv.mean_profit} out of range"

    def test_gamma_01_inventory_lower_std_than_symmetric(self):
        """For gamma=0.1, inventory std should be substantially lower than symmetric."""
        p = SimParams(gamma=0.1, n_paths=1000, seed=42)
        results = run_monte_carlo(p)
        inv_std = results["inventory"].std_profit
        sym_std = results["symmetric"].std_profit
        # Paper: inv_std ~5.89, sym_std ~13.43 (ratio ~2.3)
        assert sym_std > inv_std * 1.5, (
            f"Symmetric std {sym_std} not sufficiently larger than inventory std {inv_std}"
        )

    def test_gamma_01_inventory_lower_inventory_std(self):
        """For gamma=0.1, inventory std_q should be substantially lower than symmetric."""
        p = SimParams(gamma=0.1, n_paths=1000, seed=42)
        results = run_monte_carlo(p)
        inv_std_q = results["inventory"].std_inventory
        sym_std_q = results["symmetric"].std_inventory
        # Paper: inv_std_q ~2.80, sym_std_q ~8.66 (ratio ~3.1)
        assert sym_std_q > inv_std_q * 2.0, (
            f"Symmetric std_q {sym_std_q} not sufficiently larger than inventory std_q {inv_std_q}"
        )
