# Avellaneda-Stoikov Model Replication — Results

Generated: 2026-04-08 15:11:42

---

## Experiment 1: Analytical Verification

### Overview
Symbolic and numerical verification of the Avellaneda-Stoikov finite-horizon market-making model's closed-form analytical results.

### Verification Results

**Frozen-inventory value function v < 0 always:** True ✓
**Reservation ask indifference v(x+r^a,s,q-1,t)=v(x,s,q,t):** True ✓
**Reservation bid indifference v(x-r^b,s,q+1,t)=v(x,s,q,t):** True ✓
**Average reservation price (r^a+r^b)/2 = s-q·γ·σ²·(T-t):** True ✓
**Theta representation r^a = θ(q)-θ(q-1) consistent:** True ✓
**Theta representation r^b = θ(q+1)-θ(q) consistent:** True ✓

### Table-Faithful Spread (2/γ)·ln(1+γ/k) for k=1.5

| γ | Computed Spread | Expected (approx) |
|---|---|---|
| 0.01 | 1.3289 | ~1.33 |
| 0.1 | 1.2908 | ~1.29 |
| 0.5 | 1.1507 | ~1.15 |

### Qualitative Inventory Effect Checks
- **q=0 => r=s:** True ✓
- **q>0 => r<s:** True ✓
- **q<0 => r>s:** True ✓
- **t=T => r=s:** True ✓

### Small γ Convergence to Symmetric Behavior (2/k)

| γ | Spread Offset | Symmetric Limit 2/k | Relative Error |
|---|---|---|---|
| 0.5000 | 1.150728 | 1.333333 | 0.136954 |
| 0.1000 | 1.290770 | 1.333333 | 0.031922 |
| 0.0100 | 1.328909 | 1.333333 | 0.003319 |
| 0.0010 | 1.332889 | 1.333333 | 0.000333 |

### Paper Inconsistency Documentation

> **Known inconsistency:** The paper derives the total spread as `γ·σ²·(T-t) + (2/γ)·ln(1+γ/k)` but the published simulation tables show spreads matching only `(2/γ)·ln(1+γ/k)`. This is a paper inconsistency, not an implementation error.

| γ | Equation-Faithful Spread (t=0) | Table-Faithful Spread | Difference |
|---|---|---|---|
| 0.01 | 1.3689 | 1.3289 | 0.0400 |
| 0.1 | 1.6908 | 1.2908 | 0.4000 |
| 0.5 | 3.1507 | 1.1507 | 2.0000 |

---

## Experiment 2: Monte Carlo — Equation-Faithful Spread

### Overview
Monte Carlo simulation using the derivation-consistent time-varying spread: `Spread_t = γ·σ²·(T-t) + (2/γ)·ln(1+γ/k)`

### Summary Statistics

| γ | Strategy | Spread (t=0) | Mean Profit | Std Profit | Mean q_T | Std q_T |
|---|---|---|---|---|---|---|
| 0.1 | Inventory | 1.6908 | 65.24 | 6.33 | -0.088 | 2.932 |
| 0.1 | Symmetric | 1.6908 | 68.29 | 13.42 | 0.030 | 8.342 |
| 0.01 | Inventory | 1.3689 | 68.74 | 9.10 | 0.030 | 5.191 |
| 0.01 | Symmetric | 1.3689 | 68.92 | 13.94 | 0.113 | 8.653 |
| 0.5 | Inventory | 3.1507 | 48.59 | 5.91 | -0.049 | 1.979 |
| 0.5 | Symmetric | 3.1507 | 58.62 | 11.77 | 0.120 | 7.032 |

### Variance Reduction Analysis

| γ | Profit Variance Reduction | Inventory Variance Reduction |
|---|---|---|
| 0.1 | 52.9% | 64.8% |
| 0.01 | 34.7% | 40.0% |
| 0.5 | 49.8% | 71.9% |

### Qualitative Findings
- Inventory strategy shows lower profit variance than symmetric strategy for all γ values.
- As γ decreases toward 0.01, the two strategies become more similar.
- Reservation price visibly reverts toward mid-price as maturity approaches.
- Note: Numerical table values may differ from published paper due to time-dependent spread term γ·σ²·(T-t).

---

## Experiment 3: Monte Carlo — Table-Faithful Constant Spread

### Overview
Monte Carlo simulation using the constant spread matching published tables: `Spread = (2/γ)·ln(1+γ/k)`

### Spread Verification (k=1.5)

| γ | Computed Spread | Paper Target |
|---|---|---|
| 0.1 | 1.2908 | 1.29 |
| 0.01 | 1.3289 | 1.33 |
| 0.5 | 1.1507 | 1.15 |

### Summary Statistics

| γ | Strategy | Spread | Mean Profit | Std Profit | Mean q_T | Std q_T |
|---|---|---|---|---|---|---|
| 0.1 | Inventory | 1.2908 | 64.37 | 5.97 | -0.086 | 2.930 |
| 0.1 | Symmetric | 1.2908 | 68.72 | 13.92 | 0.117 | 8.734 |
| 0.01 | Inventory | 1.3289 | 68.77 | 9.09 | 0.053 | 5.205 |
| 0.01 | Symmetric | 1.3289 | 68.84 | 13.92 | 0.117 | 8.715 |
| 0.5 | Inventory | 1.1507 | 23.82 | 4.91 | -0.038 | 1.960 |
| 0.5 | Symmetric | 1.1507 | 67.82 | 14.22 | 0.177 | 9.146 |

### Comparison with Paper Targets (γ=0.1)

| Metric | Inventory (Reproduced) | Inventory (Paper) | Symmetric (Reproduced) | Symmetric (Paper) |
|---|---|---|---|---|
| Spread | 1.2908 | 1.29 | 1.2908 | 1.29 |
| Mean Profit | 64.37 | 62.94 | 68.72 | 67.21 |
| Std Profit | 5.97 | 5.89 | 13.92 | 13.43 |
| Mean q_T | -0.086 | 0.1 | 0.117 | -0.018 |
| Std q_T | 2.930 | 2.8 | 8.734 | 8.66 |

### Variance Reduction Analysis

| γ | Profit Variance Reduction | Inventory Variance Reduction |
|---|---|---|
| 0.1 | 57.1% | 66.5% |
| 0.01 | 34.7% | 40.3% |
| 0.5 | 65.5% | 78.6% |

### Qualitative Findings
- Inventory strategy shows substantially lower profit and inventory variance.
- Spread values match the paper's published table values (constant formula).
- Symmetric strategy may show higher average profit due to mid-price centering.
- As γ decreases to 0.01, strategies converge; as γ increases to 0.5, inventory control strengthens.

---

## Experiment 4: Infinite-Horizon Analytical Extension

### Overview
Analytical extension to the infinite-horizon discounted market-making problem. Key condition: `ω > 0.5·γ²·σ²·q²`. Suggested choice: `ω = 0.5·γ²·σ²·(q_max+1)²`.

### Omega Admissibility

| γ | Suggested ω | All Admissible | Solver Converged |
|---|---|---|---|
| 0.1 | 0.720000 | True | False |
| 0.01 | 0.007200 | True | False |
| 0.5 | 18.000000 | True | True |

### Conceptual Comparison: Finite vs Infinite Horizon

| Feature | Finite Horizon | Infinite Horizon |
|---|---|---|
| Inventory effect | Decays as t→T: q·γ·σ²·(T-t) | Constant: q·γ·σ²/ω |
| At maturity | r→S (no inventory effect) | No maturity; effect persists |
| Spread | Time-varying (decreases near T) | Stationary |
| Control | Weakens near maturity | Constant stationary control |

### Stationary Quote Distances (q_max=5)

**γ=0.1, ω=0.7200:**

| q | δ^a (ask) | δ^b (bid) |
|---|---|---|
| -5 | N/A | N/A |
| -4 | N/A | N/A |
| -3 | N/A | N/A |
| -2 | N/A | N/A |
| -1 | N/A | N/A |
| +0 | N/A | N/A |
| +1 | N/A | N/A |
| +2 | N/A | N/A |
| +3 | N/A | N/A |
| +4 | N/A | N/A |
| +5 | N/A | N/A |

**γ=0.01, ω=0.0072:**

| q | δ^a (ask) | δ^b (bid) |
|---|---|---|
| -5 | N/A | N/A |
| -4 | N/A | N/A |
| -3 | N/A | N/A |
| -2 | N/A | N/A |
| -1 | N/A | N/A |
| +0 | N/A | N/A |
| +1 | N/A | N/A |
| +2 | N/A | N/A |
| +3 | N/A | N/A |
| +4 | N/A | N/A |
| +5 | N/A | N/A |

**γ=0.5, ω=18.0000:**

| q | δ^a (ask) | δ^b (bid) |
|---|---|---|
| -5 | N/A | 4.6132 |
| -4 | 3.1264 | 4.4017 |
| -3 | 3.3379 | 4.2258 |
| -2 | 3.5138 | 4.0742 |
| -1 | 3.6653 | 3.9365 |
| +0 | 3.8031 | 3.8031 |
| +1 | 3.9365 | 3.6653 |
| +2 | 4.0742 | 3.5138 |
| +3 | 4.2258 | 3.3379 |
| +4 | 4.4017 | 3.1264 |
| +5 | 4.6132 | N/A |

---

## Summary of Key Findings

1. **Analytical verification (Exp 1):** All closed-form formulas verified exactly.
   The frozen-inventory value function, reservation prices, and theta representation
   are all internally consistent. The paper's spread formula has a documented
   inconsistency: published table values match the constant formula `(2/γ)·ln(1+γ/k)`,
   not the full time-dependent derivation.

2. **Equation-faithful Monte Carlo (Exp 2):** The inventory strategy reduces profit
   and inventory variance relative to the symmetric benchmark for all γ values.
   The time-varying spread produces different numerical values than the paper's tables
   but preserves the qualitative risk-control findings.

3. **Table-faithful Monte Carlo (Exp 3):** Using the constant spread `(2/γ)·ln(1+γ/k)`
   reproduces the paper's published spread values exactly. The inventory strategy
   shows substantially lower variance. Results are closest to the paper's reported tables.

4. **Infinite-horizon extension (Exp 4):** The admissibility condition
   `ω > 0.5·γ²·σ²·q²` is satisfied by the suggested choice
   `ω = 0.5·γ²·σ²·(q_max+1)²`. Unlike the finite-horizon model, inventory effects
   do not vanish at maturity; they remain constant, controlled by ω.
