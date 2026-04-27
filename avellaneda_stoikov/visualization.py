"""
Visualization module for the Avellaneda-Stoikov replication experiments.

Generates all plots for Experiments 1, 2, and 3.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Experiment 1 plots
# ---------------------------------------------------------------------------

def plot_exp1_reservation_prices(
    s0: float = 100.0,
    T0: float = 1.0,
    sigma0: float = 2.0,
    k0: float = 1.5,
    gammas: Tuple[float, ...] = (0.01, 0.1, 0.5),
    inventories: Tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3),
    t0: float = 0.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot reservation prices (r^b, r^a, r_avg) vs inventory for different gamma.

    Parameters
    ----------
    s0 : mid-price
    T0 : horizon
    sigma0 : volatility
    k0 : intensity decay
    gammas : risk aversion values
    inventories : inventory levels
    t0 : current time
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    tau = T0 - t0
    fig, axes = plt.subplots(1, len(gammas), figsize=(15, 5), sharey=False)
    if len(gammas) == 1:
        axes = [axes]

    for ax, g in zip(axes, gammas):
        r_b_vals = [s0 + ((-1 - 2 * q) * g * sigma0**2 * tau) / 2 for q in inventories]
        r_a_vals = [s0 + ((1 - 2 * q) * g * sigma0**2 * tau) / 2 for q in inventories]
        r_avg_vals = [(ra + rb) / 2 for ra, rb in zip(r_a_vals, r_b_vals)]

        ax.plot(inventories, r_b_vals, "b-o", label=r"$r^b$ (reservation bid)", markersize=5)
        ax.plot(inventories, r_a_vals, "r-s", label=r"$r^a$ (reservation ask)", markersize=5)
        ax.plot(inventories, r_avg_vals, "g-^", label=r"$\bar{r}$ (average)", markersize=5, linewidth=2)
        ax.axhline(s0, color="k", linestyle="--", alpha=0.5, label=f"mid-price s={s0}")
        ax.set_xlabel("Inventory q", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.set_title(f"γ = {g}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Exp 1: Reservation Prices vs Inventory\n"
        f"(s={s0}, σ={sigma0}, T={T0}, t={t0})",
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_exp1_spread_vs_time(
    T0: float = 1.0,
    sigma0: float = 2.0,
    k0: float = 1.5,
    gammas: Tuple[float, ...] = (0.01, 0.1, 0.5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the approximate total spread psi(t) vs time for different gamma.

    Parameters
    ----------
    T0 : horizon
    sigma0 : volatility
    k0 : intensity decay
    gammas : risk aversion values
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    t_grid = np.linspace(0, T0, 200)
    fig, ax = plt.subplots(figsize=(9, 5))

    for g in gammas:
        psi = g * sigma0**2 * (T0 - t_grid) + (2 / g) * np.log(1 + g / k0)
        ax.plot(t_grid, psi, label=f"γ = {g}", linewidth=2)

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Spread ψ(t)", fontsize=12)
    ax.set_title(
        "Exp 1: Approximate Total Spread vs Time\n"
        f"ψ(t) = γσ²(T-t) + (2/γ)ln(1+γ/k)  [σ={sigma0}, k={k0}, T={T0}]",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_exp1_quote_distances(
    s0: float = 100.0,
    T0: float = 1.0,
    sigma0: float = 2.0,
    k0: float = 1.5,
    gamma0: float = 0.1,
    inventories: Tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3),
    t0: float = 0.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot optimal quote distances delta^a and delta^b vs inventory.

    Parameters
    ----------
    s0 : mid-price
    T0 : horizon
    sigma0 : volatility
    k0 : intensity decay
    gamma0 : risk aversion
    inventories : inventory levels
    t0 : current time
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    tau = T0 - t0
    adj = np.log(1 + gamma0 / k0) / gamma0

    delta_a_vals = [((1 - 2 * q) * gamma0 * sigma0**2 * tau) / 2 + adj for q in inventories]
    delta_b_vals = [((1 + 2 * q) * gamma0 * sigma0**2 * tau) / 2 + adj for q in inventories]
    spread_vals = [da + db for da, db in zip(delta_a_vals, delta_b_vals)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(inventories, delta_a_vals, "r-s", label=r"$\delta^a$ (ask distance)", markersize=7)
    ax.plot(inventories, delta_b_vals, "b-o", label=r"$\delta^b$ (bid distance)", markersize=7)
    ax.plot(inventories, spread_vals, "g-^", label=r"$\delta^a + \delta^b$ (spread)", markersize=7, linewidth=2)
    ax.axhline(
        (2 / gamma0) * np.log(1 + gamma0 / k0),
        color="purple", linestyle=":", alpha=0.7,
        label=f"(2/γ)ln(1+γ/k) = {(2/gamma0)*np.log(1+gamma0/k0):.3f}",
    )
    ax.set_xlabel("Inventory q", fontsize=12)
    ax.set_ylabel("Quote Distance", fontsize=12)
    ax.set_title(
        f"Exp 1: Optimal Quote Distances vs Inventory\n"
        f"(γ={gamma0}, σ={sigma0}, k={k0}, T={T0}, t={t0})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Experiment 2 plots
# ---------------------------------------------------------------------------

def plot_exp2_sample_path(
    sample_path: Dict,
    gamma: float,
    strategy: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a representative single path: mid-price, reservation price, bid/ask quotes.

    Parameters
    ----------
    sample_path : dict with keys 't', 'S', 'r', 'p_a', 'p_b', 'q'
    gamma : risk aversion
    strategy : strategy name
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    t = sample_path["t"]
    S = sample_path["S"]
    r = sample_path["r"]
    p_a = sample_path["p_a"]
    p_b = sample_path["p_b"]
    q = sample_path["q"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t, S, "k-", label="Mid-price S", linewidth=1.5, alpha=0.8)
    ax1.plot(t, r, "g--", label="Reservation price r", linewidth=1.5)
    ax1.plot(t[:-1], p_a[:-1], "r-", label="Ask quote p^a", linewidth=1, alpha=0.7)
    ax1.plot(t[:-1], p_b[:-1], "b-", label="Bid quote p^b", linewidth=1, alpha=0.7)
    ax1.fill_between(t[:-1], p_b[:-1], p_a[:-1], alpha=0.1, color="gray", label="Quoted spread")
    ax1.set_ylabel("Price", fontsize=12)
    ax1.set_title(
        f"Exp 2: Sample Path — {strategy.capitalize()} Strategy (γ={gamma})",
        fontsize=13,
    )
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.step(t, q, "purple", where="post", linewidth=1.5)
    ax2.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time t", fontsize=12)
    ax2.set_ylabel("Inventory q", fontsize=12)
    ax2.set_title("Inventory Path", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_exp2_profit_histograms(
    inv_results,
    sym_results,
    gamma: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot overlaid histograms of terminal profit for both strategies.

    Parameters
    ----------
    inv_results : SimResults for inventory strategy
    sym_results : SimResults for symmetric strategy
    gamma : risk aversion
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(
        min(inv_results.profits.min(), sym_results.profits.min()),
        max(inv_results.profits.max(), sym_results.profits.max()),
        50,
    )

    ax.hist(
        inv_results.profits, bins=bins, alpha=0.6, color="steelblue",
        label=f"Inventory: μ={inv_results.mean_profit:.1f}, σ={inv_results.std_profit:.1f}",
        density=True,
    )
    ax.hist(
        sym_results.profits, bins=bins, alpha=0.6, color="tomato",
        label=f"Symmetric: μ={sym_results.mean_profit:.1f}, σ={sym_results.std_profit:.1f}",
        density=True,
    )

    ax.axvline(inv_results.mean_profit, color="steelblue", linestyle="--", linewidth=2)
    ax.axvline(sym_results.mean_profit, color="tomato", linestyle="--", linewidth=2)

    ax.set_xlabel("Terminal Profit X_T + q_T·S_T", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Exp 2: Terminal Profit Distribution (γ={gamma})\n"
        f"N={len(inv_results.profits)} paths",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_exp2_all_gammas_comparison(
    all_results: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot profit histograms for all gamma values in a grid.

    Parameters
    ----------
    all_results : dict from run_all_gammas
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    gammas = sorted(all_results.keys())
    fig, axes = plt.subplots(1, len(gammas), figsize=(15, 5))
    if len(gammas) == 1:
        axes = [axes]

    for ax, g in zip(axes, gammas):
        inv_res, sym_res = all_results[g]
        bins = np.linspace(
            min(inv_res.profits.min(), sym_res.profits.min()),
            max(inv_res.profits.max(), sym_res.profits.max()),
            40,
        )
        ax.hist(inv_res.profits, bins=bins, alpha=0.6, color="steelblue",
                label=f"Inventory\nμ={inv_res.mean_profit:.1f}\nσ={inv_res.std_profit:.1f}",
                density=True)
        ax.hist(sym_res.profits, bins=bins, alpha=0.6, color="tomato",
                label=f"Symmetric\nμ={sym_res.mean_profit:.1f}\nσ={sym_res.std_profit:.1f}",
                density=True)
        ax.set_title(f"γ = {g}", fontsize=13)
        ax.set_xlabel("Terminal Profit", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 2: Terminal Profit Distributions — Inventory vs Symmetric", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_exp2_inventory_histograms(
    all_results: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot final inventory histograms for all gamma values.

    Parameters
    ----------
    all_results : dict from run_all_gammas
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    gammas = sorted(all_results.keys())
    fig, axes = plt.subplots(1, len(gammas), figsize=(15, 5))
    if len(gammas) == 1:
        axes = [axes]

    for ax, g in zip(axes, gammas):
        inv_res, sym_res = all_results[g]
        all_q = np.concatenate([inv_res.final_inventories, sym_res.final_inventories])
        bins = np.arange(all_q.min() - 0.5, all_q.max() + 1.5, 1)

        ax.hist(inv_res.final_inventories, bins=bins, alpha=0.6, color="steelblue",
                label=f"Inventory\nμ={inv_res.mean_final_q:.2f}\nσ={inv_res.std_final_q:.2f}",
                density=True)
        ax.hist(sym_res.final_inventories, bins=bins, alpha=0.6, color="tomato",
                label=f"Symmetric\nμ={sym_res.mean_final_q:.2f}\nσ={sym_res.std_final_q:.2f}",
                density=True)
        ax.set_title(f"γ = {g}", fontsize=13)
        ax.set_xlabel("Final Inventory q_T", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 2: Final Inventory Distributions — Inventory vs Symmetric", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Experiment 3 plots
# ---------------------------------------------------------------------------

def plot_exp3_intensity_comparison(
    deltas: Optional[np.ndarray] = None,
    A: float = 140.0,
    k: float = 1.5,
    B: float = 1.0,
    alpha: float = 1.4,
    beta: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot exponential vs power-law execution intensities.

    Parameters
    ----------
    deltas : array of quote distances
    A : exponential scale
    k : exponential decay
    B : power-law scale
    alpha : order size tail exponent
    beta : impact exponent
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    if deltas is None:
        deltas = np.linspace(0.01, 3.0, 300)

    lambda_exp = A * np.exp(-k * deltas)
    lambda_power = B * deltas ** (-alpha / beta)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Linear scale
    axes[0].plot(deltas, lambda_exp, "b-", label=f"Exponential: A·exp(-k·δ)\nA={A}, k={k}", linewidth=2)
    axes[0].plot(deltas, lambda_power, "r--", label=f"Power-law: B·δ^(-α/β)\nB={B}, α={alpha}, β={beta}", linewidth=2)
    axes[0].set_xlabel("Quote distance δ", fontsize=12)
    axes[0].set_ylabel("Intensity λ(δ)", fontsize=12)
    axes[0].set_title("Execution Intensities (Linear Scale)", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, min(lambda_exp.max(), lambda_power.max()) * 1.1)
    axes[0].grid(True, alpha=0.3)

    # Log scale
    axes[1].semilogy(deltas, lambda_exp, "b-", label="Exponential", linewidth=2)
    axes[1].semilogy(deltas, lambda_power, "r--", label="Power-law", linewidth=2)
    axes[1].set_xlabel("Quote distance δ", fontsize=12)
    axes[1].set_ylabel("Intensity λ(δ) [log scale]", fontsize=12)
    axes[1].set_title("Execution Intensities (Log Scale)", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Exp 3: Execution Intensity Functions from Microstructure Derivations", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_exp3_infinite_horizon_prices(
    s: float = 100.0,
    gammas: Tuple[float, ...] = (0.01, 0.1, 0.5),
    sigma: float = 2.0,
    q_max: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot infinite-horizon reservation prices vs inventory for different gamma.

    Parameters
    ----------
    s : mid-price
    gammas : risk aversion values
    sigma : volatility
    q_max : max inventory
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    from avellaneda_stoikov.exp3.intensity_and_infinite_horizon import (
        infinite_horizon_inventory_scan,
    )

    inventories = np.arange(-q_max, q_max + 1)
    fig, axes = plt.subplots(1, len(gammas), figsize=(15, 5), sharey=False)
    if len(gammas) == 1:
        axes = [axes]

    for ax, g in zip(axes, gammas):
        scan = infinite_horizon_inventory_scan(s, g, sigma, q_max, inventories)
        omega = scan["omega"]

        ax.plot(inventories, scan["r_bar_a"], "r-s", label=r"$\bar{r}^a$", markersize=6)
        ax.plot(inventories, scan["r_bar_b"], "b-o", label=r"$\bar{r}^b$", markersize=6)
        ax.plot(inventories, scan["r_bar_avg"], "g-^", label=r"$\bar{r}_{avg}$", markersize=6, linewidth=2)
        ax.axhline(s, color="k", linestyle="--", alpha=0.5, label=f"mid-price s={s}")
        ax.set_xlabel("Inventory q", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.set_title(f"γ={g}, ω={omega:.4f}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Exp 3: Infinite-Horizon Stationary Reservation Prices vs Inventory\n"
        f"(s={s}, σ={sigma}, q_max={q_max})",
        fontsize=14,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_exp3_finite_vs_infinite(
    s: float = 100.0,
    gamma: float = 0.1,
    sigma: float = 2.0,
    T: float = 1.0,
    t: float = 0.0,
    q_max: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare finite-horizon and infinite-horizon average reservation prices.

    Parameters
    ----------
    s : mid-price
    gamma : risk aversion
    sigma : volatility
    T : finite horizon
    t : current time
    q_max : max inventory
    save_path : path to save figure

    Returns
    -------
    matplotlib Figure
    """
    from avellaneda_stoikov.exp3.intensity_and_infinite_horizon import (
        compare_finite_vs_infinite_horizon,
    )

    comp = compare_finite_vs_infinite_horizon(s, gamma, sigma, T, t, q_max)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(comp["q"], comp["r_finite"], "b-o", label=f"Finite-horizon r(s,q,t)\n(τ={comp['tau']:.2f})", linewidth=2, markersize=6)
    ax.plot(comp["q"], comp["r_inf_avg"], "r-s", label=f"Infinite-horizon r̄_avg\n(ω={comp['omega']:.4f})", linewidth=2, markersize=6)
    ax.axhline(s, color="k", linestyle="--", alpha=0.5, label=f"mid-price s={s}")
    ax.set_xlabel("Inventory q", fontsize=12)
    ax.set_ylabel("Average Reservation Price", fontsize=12)
    ax.set_title(
        f"Exp 3: Finite vs Infinite Horizon Reservation Prices\n"
        f"(γ={gamma}, σ={sigma}, s={s})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
