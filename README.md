# Avellaneda-Stoikov Market-Making Model: Analytical Replication

This repository replicates the analytical and simulation results from the Avellaneda-Stoikov (2008) paper
*"High-frequency trading in a limit order book"*.

## Structure

```
avellaneda_stoikov/
  exp1/   - Analytical finite-horizon model (reservation prices, HJB, FOC)
  exp2/   - Monte Carlo simulation: inventory vs symmetric strategies
  exp3/   - Intensity derivations and infinite-horizon extension
tests/    - pytest test suite
results/  - Output figures and RESULTS.md
run_all.py - Master script to run all experiments
```

## Experiments

### Exp 1: Analytical Finite-Horizon Model
Symbolic and numeric verification of:
- Frozen-inventory utility-indifference reservation prices
- HJB transformation under exponential utility ansatz
- First-order optimal quote conditions
- Approximate closed-form finite-horizon quoting formulas

### Exp 2: Monte Carlo Simulation
Replication of the paper's simulation comparing:
- Inventory-based quoting strategy
- Symmetric benchmark strategy
Across gamma ∈ {0.01, 0.1, 0.5}

### Exp 3: Intensity Derivations & Infinite-Horizon
- Derivation of exponential intensity from log-impact + power-law order sizes
- Derivation of power-law intensity from power-law impact
- Infinite-horizon stationary reservation price formulas

## Usage

```bash
pip install -r requirements.txt
python run_all.py
pytest tests/ -v
```

## Parameters (Paper defaults)
- S₀ = 100, T = 1, σ = 2, dt = 0.005
- A = 140, k = 1.5
- γ ∈ {0.01, 0.1, 0.5}
- N_paths = 1000
