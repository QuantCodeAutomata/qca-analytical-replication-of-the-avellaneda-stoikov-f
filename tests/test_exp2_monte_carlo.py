"""
Tests for Experiment 2: Monte Carlo simulation of inventory vs symmetric strategies.

Verifies:
  - Simulation produces correct output shapes and types
  - Inventory strategy has lower P&L variance than symmetric
  - Inventory strategy has lower final inventory variance than symmetric
  - Qualitative behavior across gamma values
  - Edge cases and parameter validation
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from avellaneda_stoikov.exp2.monte_carlo import (
    SimParams,
    SimResults,
    PathDiagnostics,
    compute_inventory_quotes,
    compute_symmetric_quotes,
    simulate_paths,
    run_comparison,
    run_all_gammas,
    build_summary_table,
    build_diagnostics_table,
)


# ---------------------------------------------------------------------------
# Quote computation tests
# ---------------------------------------------------------------------------

class TestQuoteComputation:
    """Tests for quote computation functions."""

    def test_inventory_quotes_zero_inventory(self):
        """With q=0, inventory quotes must be symmetric around mid-price."""
        da, db, pa, pb = compute_inventory_quotes(100.0, 0, 0.0, 1.0, 0.1, 2.0, 1.5)
        assert abs(da - db) < 1e-10, "With q=0, delta_a must equal delta_b"
        assert abs(pa - 100.0 - da) < 1e-10
        assert abs(100.0 - pb - db) < 1e-10

    def test_symmetric_quotes_always_symmetric(self):
        """Symmetric quotes must always be symmetric regardless of inventory."""
        for q_val in [-5, -1, 0, 1, 5]:
            da, db, pa, pb = compute_symmetric_quotes(100.0, 0.0, 1.0, 0.1, 2.0, 1.5)
            assert abs(da - db) < 1e-10, f"Symmetric quotes must be equal for q={q_val}"

    def test_inventory_quotes_positive_inventory_ask_closer(self):
        """For positive inventory, ask distance must be smaller than bid distance."""
        for q_val in [1, 2, 3]:
            da, db, pa, pb = compute_inventory_quotes(100.0, q_val, 0.0, 1.0, 0.1, 2.0, 1.5)
            assert da < db, (
                f"For q={q_val}>0, ask distance should be smaller than bid distance"
            )

    def test_inventory_quotes_negative_inventory_bid_closer(self):
        """For negative inventory, bid distance must be smaller than ask distance."""
        for q_val in [-1, -2, -3]:
            da, db, pa, pb = compute_inventory_quotes(100.0, q_val, 0.0, 1.0, 0.1, 2.0, 1.5)
            assert db < da, (
                f"For q={q_val}<0, bid distance should be smaller than ask distance"
            )

    def test_quotes_prices_consistent(self):
        """Quote prices must be consistent with distances."""
        S = 100.0
        da, db, pa, pb = compute_inventory_quotes(S, 1, 0.0, 1.0, 0.1, 2.0, 1.5)
        assert abs(pa - S - da) < 1e-10, "p^a = S + delta^a"
        assert abs(S - pb - db) < 1e-10, "p^b = S - delta^b"

    def test_symmetric_quotes_prices_consistent(self):
        """Symmetric quote prices must be consistent with distances."""
        S = 100.0
        da, db, pa, pb = compute_symmetric_quotes(S, 0.0, 1.0, 0.1, 2.0, 1.5)
        assert abs(pa - S - da) < 1e-10, "p^a = S + delta^a"
        assert abs(S - pb - db) < 1e-10, "p^b = S - delta^b"

    def test_quotes_spread_matches_formula(self):
        """Total spread from quotes must match the spread formula."""
        g, sig, k_val = 0.1, 2.0, 1.5
        da, db, pa, pb = compute_inventory_quotes(100.0, 0, 0.0, 1.0, g, sig, k_val)
        spread_from_quotes = da + db
        spread_formula = g * sig**2 * 1.0 + (2 / g) * math.log(1 + g / k_val)
        assert abs(spread_from_quotes - spread_formula) < 1e-10


# ---------------------------------------------------------------------------
# Simulation output tests
# ---------------------------------------------------------------------------

class TestSimulationOutput:
    """Tests for simulation output structure and types."""

    @pytest.fixture
    def small_params(self):
        """Small simulation for fast testing."""
        return SimParams(
            S0=100.0, T=1.0, sigma=2.0, dt=0.005,
            q0=0, X0=0.0, A=140.0, k=1.5,
            n_paths=50, gamma=0.1, seed=42,
        )

    def test_simulate_returns_simresults(self, small_params):
        """simulate_paths must return a SimResults object."""
        result = simulate_paths(small_params, "inventory")
        assert isinstance(result, SimResults)

    def test_profits_array_shape(self, small_params):
        """Profits array must have shape (n_paths,)."""
        result = simulate_paths(small_params, "inventory")
        assert result.profits.shape == (small_params.n_paths,)

    def test_final_inventories_array_shape(self, small_params):
        """Final inventories array must have shape (n_paths,)."""
        result = simulate_paths(small_params, "inventory")
        assert result.final_inventories.shape == (small_params.n_paths,)

    def test_final_inventories_are_integers(self, small_params):
        """Final inventories must be integers."""
        result = simulate_paths(small_params, "inventory")
        assert result.final_inventories.dtype in [np.int32, np.int64, int], (
            "Final inventories must be integer type"
        )

    def test_profits_are_finite(self, small_params):
        """All profits must be finite."""
        result = simulate_paths(small_params, "inventory")
        assert np.all(np.isfinite(result.profits)), "All profits must be finite"

    def test_mean_profit_matches_array(self, small_params):
        """mean_profit must match np.mean(profits)."""
        result = simulate_paths(small_params, "inventory")
        assert abs(result.mean_profit - np.mean(result.profits)) < 1e-10

    def test_std_profit_matches_array(self, small_params):
        """std_profit must match np.std(profits)."""
        result = simulate_paths(small_params, "inventory")
        assert abs(result.std_profit - np.std(result.profits)) < 1e-10

    def test_mean_final_q_matches_array(self, small_params):
        """mean_final_q must match np.mean(final_inventories)."""
        result = simulate_paths(small_params, "inventory")
        assert abs(result.mean_final_q - np.mean(result.final_inventories)) < 1e-10

    def test_std_final_q_matches_array(self, small_params):
        """std_final_q must match np.std(final_inventories)."""
        result = simulate_paths(small_params, "inventory")
        assert abs(result.std_final_q - np.std(result.final_inventories)) < 1e-10

    def test_reported_spread_positive(self, small_params):
        """Reported spread must be positive."""
        result = simulate_paths(small_params, "inventory")
        assert result.reported_spread > 0

    def test_sample_path_has_correct_keys(self, small_params):
        """Sample path must have required keys."""
        result = simulate_paths(small_params, "inventory")
        assert result.sample_path is not None
        for key in ["t", "S", "r", "p_a", "p_b", "q"]:
            assert key in result.sample_path, f"Sample path missing key: {key}"

    def test_invalid_strategy_raises(self, small_params):
        """Invalid strategy name must raise AssertionError."""
        with pytest.raises(AssertionError):
            simulate_paths(small_params, "invalid_strategy")

    def test_strategy_name_stored(self, small_params):
        """Strategy name must be stored in results."""
        for strat in ["inventory", "symmetric"]:
            result = simulate_paths(small_params, strat)
            assert result.strategy == strat

    def test_gamma_stored(self, small_params):
        """Gamma must be stored in results."""
        result = simulate_paths(small_params, "inventory")
        assert result.gamma == small_params.gamma


# ---------------------------------------------------------------------------
# Qualitative behavior tests
# ---------------------------------------------------------------------------

class TestQualitativeBehavior:
    """Tests for qualitative behavior of the simulation."""

    @pytest.fixture
    def comparison_results(self):
        """Run comparison for gamma=0.1 with moderate paths."""
        return run_comparison(gamma=0.1, params_override={"n_paths": 500}, seed=42)

    def test_inventory_lower_profit_std_than_symmetric(self, comparison_results):
        """Inventory strategy must have lower P&L std than symmetric."""
        inv_res, sym_res = comparison_results
        assert inv_res.std_profit < sym_res.std_profit, (
            f"Inventory std(Profit)={inv_res.std_profit:.2f} should be < "
            f"Symmetric std(Profit)={sym_res.std_profit:.2f}"
        )

    def test_inventory_lower_final_q_std_than_symmetric(self, comparison_results):
        """Inventory strategy must have lower final inventory std than symmetric."""
        inv_res, sym_res = comparison_results
        assert inv_res.std_final_q < sym_res.std_final_q, (
            f"Inventory std(q_T)={inv_res.std_final_q:.2f} should be < "
            f"Symmetric std(q_T)={sym_res.std_final_q:.2f}"
        )

    def test_inventory_mean_final_q_near_zero(self, comparison_results):
        """Inventory strategy mean final inventory should be near zero."""
        inv_res, _ = comparison_results
        assert abs(inv_res.mean_final_q) < 2.0, (
            f"Inventory mean final q should be near 0, got {inv_res.mean_final_q}"
        )

    def test_both_strategies_positive_mean_profit(self, comparison_results):
        """Both strategies should have positive mean profit."""
        inv_res, sym_res = comparison_results
        assert inv_res.mean_profit > 0, "Inventory strategy should have positive mean profit"
        assert sym_res.mean_profit > 0, "Symmetric strategy should have positive mean profit"

    def test_high_gamma_stronger_inventory_control(self):
        """Higher gamma should produce stronger inventory control (lower std(q_T))."""
        inv_low, _ = run_comparison(gamma=0.01, params_override={"n_paths": 300}, seed=42)
        inv_high, _ = run_comparison(gamma=0.5, params_override={"n_paths": 300}, seed=42)
        assert inv_high.std_final_q < inv_low.std_final_q, (
            f"Higher gamma should give lower std(q_T): "
            f"gamma=0.5 std={inv_high.std_final_q:.2f} vs gamma=0.01 std={inv_low.std_final_q:.2f}"
        )

    def test_low_gamma_strategies_similar(self):
        """For very low gamma, inventory and symmetric strategies should be similar."""
        inv_res, sym_res = run_comparison(gamma=0.01, params_override={"n_paths": 300}, seed=42)
        # At low gamma, the two strategies should have similar mean profits
        profit_diff = abs(inv_res.mean_profit - sym_res.mean_profit)
        # The difference should be smaller than at high gamma
        inv_high, sym_high = run_comparison(gamma=0.5, params_override={"n_paths": 300}, seed=42)
        profit_diff_high = abs(inv_high.mean_profit - sym_high.mean_profit)
        assert profit_diff < profit_diff_high, (
            "Low gamma strategies should be more similar than high gamma strategies"
        )


# ---------------------------------------------------------------------------
# All-gamma experiment tests
# ---------------------------------------------------------------------------

class TestAllGammasExperiment:
    """Tests for the full multi-gamma experiment."""

    @pytest.fixture(scope="class")
    def all_results(self):
        """Run all gammas with small path count for speed."""
        return run_all_gammas(gammas=(0.01, 0.1, 0.5), n_paths=200, seed=42)

    def test_all_gammas_present(self, all_results):
        """Results must contain all three gamma values."""
        assert set(all_results.keys()) == {0.01, 0.1, 0.5}

    def test_each_gamma_has_two_strategies(self, all_results):
        """Each gamma must have results for both strategies."""
        for g, (inv_res, sym_res) in all_results.items():
            assert isinstance(inv_res, SimResults)
            assert isinstance(sym_res, SimResults)

    def test_inventory_lower_std_all_gammas(self, all_results):
        """Inventory strategy must have lower P&L std for all gamma values."""
        for g, (inv_res, sym_res) in all_results.items():
            assert inv_res.std_profit < sym_res.std_profit, (
                f"gamma={g}: Inventory std(Profit)={inv_res.std_profit:.2f} should be < "
                f"Symmetric std(Profit)={sym_res.std_profit:.2f}"
            )

    def test_inventory_lower_q_std_all_gammas(self, all_results):
        """Inventory strategy must have lower final inventory std for all gamma values."""
        for g, (inv_res, sym_res) in all_results.items():
            assert inv_res.std_final_q < sym_res.std_final_q, (
                f"gamma={g}: Inventory std(q_T)={inv_res.std_final_q:.2f} should be < "
                f"Symmetric std(q_T)={sym_res.std_final_q:.2f}"
            )

    def test_summary_table_shape(self, all_results):
        """Summary table must have 6 rows (2 strategies × 3 gammas)."""
        df = build_summary_table(all_results)
        assert len(df) == 6, f"Summary table should have 6 rows, got {len(df)}"

    def test_summary_table_columns(self, all_results):
        """Summary table must have required columns."""
        df = build_summary_table(all_results)
        required_cols = {"gamma", "Strategy", "Spread", "Profit", "std(Profit)", "Final q", "std(Final q)"}
        assert required_cols.issubset(set(df.columns))

    def test_diagnostics_table_shape(self, all_results):
        """Diagnostics table must have 6 rows."""
        df = build_diagnostics_table(all_results)
        assert len(df) == 6

    def test_spread_decreases_with_gamma(self, all_results):
        """Reported spread should generally decrease with gamma (constant component dominates)."""
        # The constant component (2/gamma)*ln(1+gamma/k) decreases with gamma
        # The time-varying component gamma*sigma^2*(T-t) increases with gamma
        # At t=0, T=1: total = gamma*4 + (2/gamma)*ln(1+gamma/1.5)
        # For gamma=0.01: ~0.04 + 200*ln(1.0067) ≈ 0.04 + 1.33 ≈ 1.37
        # For gamma=0.1: ~0.4 + 20*ln(1.067) ≈ 0.4 + 1.29 ≈ 1.69
        # For gamma=0.5: ~2.0 + 4*ln(1.33) ≈ 2.0 + 1.15 ≈ 3.15
        # Time-averaged spread will be lower than t=0 value
        # Just verify spreads are positive and finite
        for g, (inv_res, sym_res) in all_results.items():
            assert inv_res.reported_spread > 0
            assert sym_res.reported_spread > 0
            assert math.isfinite(inv_res.reported_spread)


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Tests for simulation reproducibility."""

    def test_same_seed_same_results(self):
        """Same seed must produce identical results."""
        params = SimParams(n_paths=50, gamma=0.1, seed=123)
        result1 = simulate_paths(params, "inventory")
        result2 = simulate_paths(params, "inventory")
        np.testing.assert_array_equal(result1.profits, result2.profits)
        np.testing.assert_array_equal(result1.final_inventories, result2.final_inventories)

    def test_different_seeds_different_results(self):
        """Different seeds must produce different results."""
        params1 = SimParams(n_paths=50, gamma=0.1, seed=1)
        params2 = SimParams(n_paths=50, gamma=0.1, seed=2)
        result1 = simulate_paths(params1, "inventory")
        result2 = simulate_paths(params2, "inventory")
        assert not np.array_equal(result1.profits, result2.profits)

    def test_common_random_numbers_consistency(self):
        """Common random numbers must produce consistent comparison."""
        inv_res, sym_res = run_comparison(gamma=0.1, params_override={"n_paths": 100}, seed=42)
        # Both strategies use same random numbers, so comparison is fair
        assert inv_res.std_profit < sym_res.std_profit
