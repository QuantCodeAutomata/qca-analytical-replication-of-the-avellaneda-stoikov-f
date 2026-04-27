# Repository: Avellaneda-Stoikov Market-Making Model Replication

## Overview
This repository replicates the analytical and simulation results from Avellaneda & Stoikov (2008),
"High-frequency trading in a limit order book."

## Structure
```
avellaneda_stoikov/
  exp1/analytical_model.py   - Symbolic + numeric finite-horizon model
  exp2/monte_carlo.py        - Monte Carlo simulation
  exp3/intensity_and_infinite_horizon.py - Intensity derivations + infinite-horizon
  visualization.py           - All plotting functions
tests/
  test_exp1_analytical.py
  test_exp2_monte_carlo.py
  test_exp3_intensity.py
run_all.py                   - Master experiment runner
results/                     - Output figures and RESULTS.md
```

## Key Implementation Notes

### Experiment 1
- Uses sympy for symbolic algebra (Context7 confirmed)
- All formulas verified symbolically via `run_symbolic_verification()`
- Paper parameters: s=100, T=1, sigma=2, k=1.5, gamma in {0.01, 0.1, 0.5}
- Key formulas:
  - r^b = s + ((-1-2q)*gamma*sigma^2*(T-t))/2
  - r^a = s + ((1-2q)*gamma*sigma^2*(T-t))/2
  - r_avg = s - q*gamma*sigma^2*(T-t)
  - spread = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)

### Experiment 2
- Mid-price: binary step S_{n+1} = S_n ± sigma*sqrt(dt) (not Gaussian)
- Fill probabilities: Bernoulli with p = lambda*dt, capped at 1.0
- Common random numbers used for fair comparison
- Ambiguities documented in AMBIGUITY_LOG string
- Paper spread values appear to use only constant component (2/gamma)*ln(1+gamma/k)

### Experiment 3
- Exponential intensity: lambda(delta) = A*exp(-k*delta) from log-impact + power-law sizes
- Power-law intensity: lambda(delta) = B*delta^(-alpha/beta) from power-law impact
- Infinite-horizon: omega = (1/2)*gamma^2*sigma^2*(q_max+1)^2 for admissibility
- Admissibility: 2*omega - gamma^2*q^2*sigma^2 > 0

## Testing
- Run: `pytest tests/ -v`
- All tests use real code paths (no mocks)
- Edge cases: zero inventory, zero sigma, t=T, gamma->0

## Dependencies
- sympy: symbolic algebra
- numpy: numerical computation
- pandas: data tables
- matplotlib + seaborn: visualization
- pytest: testing
