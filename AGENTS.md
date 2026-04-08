# Repository: Avellaneda-Stoikov Market-Making Model Replication

## Overview
This repository implements analytical and Monte Carlo replication of the Avellaneda-Stoikov (2008)
finite-horizon market-making model. It covers:
- Exp 1: Symbolic/analytical verification of closed-form formulas
- Exp 2: Monte Carlo simulation with equation-faithful (time-varying) spread
- Exp 3: Monte Carlo simulation with table-faithful (constant) spread
- Exp 4: Infinite-horizon analytical extension

## Structure
```
src/
  exp1_analytical.py      # Symbolic verification using sympy
  exp2_mc_eq_spread.py    # Monte Carlo, equation-faithful spread
  exp3_mc_const_spread.py # Monte Carlo, table-faithful constant spread
  exp4_infinite_horizon.py# Infinite-horizon analytical extension
tests/
  test_exp1.py
  test_exp2.py
  test_exp3.py
  test_exp4.py
results/
  RESULTS.md              # All metrics and findings
  *.png                   # Plots
```

## Key Parameters (Avellaneda-Stoikov)
- s0=100, T=1, sigma=2, dt=0.005, N=200
- A=140, k=1.5, q0=0, X0=0
- gamma in {0.1, 0.01, 0.5}
- 1000 paths per experiment

## Key Formulas
- Reservation price: r(s,q,t) = s - q*gamma*sigma^2*(T-t)
- Equation-faithful spread: gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)
- Table-faithful spread: (2/gamma)*ln(1 + gamma/k)
- Frozen-inventory value: v = -exp(-gamma*x)*exp(-gamma*q*s)*exp(gamma^2*q^2*sigma^2*(T-t)/2)

## Libraries Used
- sympy: symbolic math (Context7 confirmed)
- numpy: numerical computation
- matplotlib/seaborn: visualization
- pytest: testing

## Notes
- The paper has an inconsistency: published table spreads match constant formula, not full time-varying formula
- Exp 2 uses equation-faithful spread (may not match tables)
- Exp 3 uses constant spread (matches published tables)
- Simultaneous bid/ask fills are allowed (independent Bernoulli draws)
- Mid-price uses ±sigma*sqrt(dt) with prob 0.5 (not Gaussian Euler)
