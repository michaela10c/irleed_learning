# IRLEED Scaling Ambiguity Experiments

## Overview

This project studies scaling ambiguity in IRLEED (Inverse Reinforcement Learning by Estimating Expertise of Demonstrators).

We investigate the following question:

Why does reward recovery succeed while expertise (beta) estimation fails?

We evaluate three experimental settings:
1. Homogeneous setting (single demonstrator, epsilon OFF)
2. Heterogeneous setting (multiple demonstrators, epsilon OFF)
3. Epsilon-only setting (shared beta, epsilon ON)

---

## Codebase Structure

The implementation builds on the original IRLEED codebase.

```
src/
├── irl_maxent/        (Maximum Entropy IRL baseline)
│   ├── gridworld.py
│   ├── maxent.py
│   ├── optimizer.py
│   ├── solver.py
│   ├── trajectory.py
│   └── plot.py
│
└── mix_irl/           (IRLEED + extensions)
    ├── irl.py
    ├── irleed.py
    ├── helpers.py
    └── trajectory.py
```

Module roles:

- `irl_maxent/`  
  Standard MaxEnt IRL implementation used as a baseline.

- `mix_irl/`  
  IRLEED implementation and extensions used in this project:
  - mixture modeling across components
  - beta (expertise) estimation
  - epsilon-based reward perturbations
  - multi-seed experiment pipeline

---

## Environment

- Gridworld (7 x 7)
- 3 terminal reward corners
- Discount factor gamma = 0.9
- Maximum Entropy IRL framework

---

## Results Directory Structure

All outputs are stored under:

`results/gridworld_simple/`

### 1. Standard IRLEED Experiments (ε OFF)

`results/gridworld_simple/irleed/env_1/`

Contains:

Homogeneous:
`demo_beta_0.100/noeps/baseline.p`  
`demo_beta_1.000/noeps/baseline.p`  
`demo_beta_5.000/noeps/baseline.p`  
`demo_beta_10.000/noeps/baseline.p`

Heterogeneous:
`demo_betas_0.300_1.000_5.000/noeps/baseline.p`

These experiments explicitly disable epsilon (`noeps`) to study the baseline IRLEED behavior.

---

### 2. Epsilon-Only Experiments (ε ON)

`results/gridworld_simple/irleed_eps_only/env_1/demo_betas_1.000_1.000_1.000/eps/lam_2.000.p`

This setting enables epsilon while fixing beta across components.

---

## Experiments

### 1. Homogeneous Setting (ε OFF)

Setup:
- `n_components = 1`
- beta values = 0.1, 1.0, 5.0, 10.0
- epsilon disabled

Purpose:
- Test identifiability with a single demonstrator

Result:
- beta is not recovered correctly
- converges to similar values

Conclusion:
Scaling ambiguity:
(theta, beta) is equivalent to (c * theta, beta / c)

---

### 2. Heterogeneous Setting (ε OFF)

Setup:
- `n_components = 3`
- beta values = 0.3, 1.0, 5.0
- epsilon disabled

Purpose:
- Test whether demonstrator heterogeneity resolves ambiguity

Result:
- reward recovery improves
- policies collapse to similar behavior

Conclusion:
No distinct behavioral modes are recovered.  
Scaling ambiguity persists even with heterogeneous data.

---

### 3. Epsilon-Only Setting (ε ON)

Setup:
- `n_components = 3`
- beta = 1.0 (shared across all components)
- epsilon learned per component
- lambda = 2.0
- 100 seeds, 1000 steps

Purpose:
- Test whether epsilon alone can explain heterogeneity

---

## How to Run

Run epsilon-only experiment:

```
python run_mix.py \
  --save_dir gridworld_simple/irleed_eps_only \
  --n_components 3 \
  --demo_betas 1.0 1.0 1.0 \
  --max_steps 1000
```

To reproduce homogeneous or heterogeneous experiments:
- change `n_components`
- change `demo_betas`
- keep epsilon disabled (default behavior)

---

## Output Files

Each `.p` file contains:
- learned theta (reward)
- learned beta (expertise)
- epsilon (if enabled)
- training histories
- aggregated multi-seed results

---

## Important Note on Seeds

- 100 seeds launched
- 2 seeds failed due to numerical instability

Cause:
- overflow in weighting
- invalid normalization
- NaN probabilities during trajectory sampling

Example error:
ValueError: probabilities contain NaN

Final results:
98 valid seeds

This does not affect qualitative conclusions.

---

## Results Summary

### Beta

- converges to ~1.34 to 1.35
- nearly identical across components

Conclusion:
beta is globally rescaled (scaling ambiguity)

---

### Theta

- U-shaped error curve

Interpretation:
scaling drift, not overfitting

---

### Epsilon

- learned successfully
- structured spatial patterns
- similar across components

Conclusion:
epsilon adjusts reward but does not create distinct behaviors

---

### Policy

- nearly identical across components

---

### Visitation

- minor variation only
- no clear behavioral decomposition

---

## Final Conclusion

- reward recovery works
- expertise (beta) is not identifiable
- components collapse to shared behavior

Scaling ambiguity persists:
- without epsilon (baseline IRLEED)
- with epsilon (epsilon-only setting)

---

## Files

run_mix.py          main experiment script  
src/irl_maxent/     MaxEnt IRL  
src/mix_irl/        IRLEED + extensions  
results/            outputs  

---

## Notes

- builds on the original IRLEED framework
- homogeneous and heterogeneous experiments explicitly disable epsilon
- epsilon-only experiment isolates epsilon as the only source of heterogeneity
- multi-seed evaluation used for robustness
- some seeds may fail due to numerical instability
