# IRLEED Scaling Ambiguity Experiments

## Overview

This project studies scaling ambiguity in IRLEED (Inverse Reinforcement Learning by Estimating Expertise of Demonstrators).

We investigate the following question:

Why does reward recovery succeed while expertise (beta) estimation fails?

We evaluate three experimental settings:
1. Homogeneous setting (single demonstrator)
2. Heterogeneous setting (multiple demonstrators with different beta values)
3. Epsilon-only setting (shared beta, learned reward perturbations epsilon)

---

## Environment

- Gridworld (7 x 7)
- 3 terminal reward corners
- Discount factor: gamma = 0.9
- Maximum entropy IRL framework

---

## Experiments

### 1. Homogeneous Setting

Setup:
- Number of components: 1
- Demonstrator beta values tested: {0.1, 1.0, 5.0, 10}

Purpose:
- Test identifiability with a single demonstrator

Result:
- The model does not recover the true beta
- Instead, it converges to a similar value across runs

This demonstrates scaling ambiguity:
(theta, beta) is equivalent to (c * theta, beta / c)

---

### 2. Heterogeneous Setting

Setup:
- Number of components: 3
- Demonstrator betas differ across components

Purpose:
- Test whether heterogeneity resolves ambiguity

Result:
- Reward recovery improves
- Learned policies across components become nearly identical

Conclusion:
- The model does not recover distinct behavioral modes
- Scaling ambiguity persists

---

### 3. Epsilon-Only Setting

Setup:
- Number of components: 3
- Shared beta across all demonstrators:
  beta = 1.0
- Learn component-specific reward perturbations epsilon
- 100 seeds, 1000 training steps

---

## How to Run

Run the epsilon-only experiment:

```
python run_mix.py \
  --save_dir gridworld_simple/irleed_eps_only \
  --n_components 3 \
  --demo_betas 1.0 1.0 1.0 \
  --max_steps 1000
```

Other experiment settings (homogeneous / heterogeneous) can be run by modifying:
- number of components
- demo_betas

---

## Output

Results are saved under:

```
gridworld_simple/irleed_eps_only/
```

Each seed produces a result file. Aggregated statistics are computed across seeds.

---

## Important Note on Seeds

- 100 seeds were launched
- 2 seeds failed due to numerical instability:
  - Overflow in weighting computation
  - Invalid policy normalization
  - NaN probabilities during trajectory sampling

Example error:
```
ValueError: probabilities contain NaN
```

Final results are computed over:

98 valid seeds

This does not affect qualitative conclusions.

---

## Results Summary

### Beta Behavior
- Learned beta converges to approximately 1.34 to 1.35 across all components
- Very small spread across components
- No differentiation via rationality
- Confirms scaling ambiguity

---

### Reward (Theta) Behavior
- Error follows a U-shaped curve
- Decreases early, then increases later

Interpretation:
- Indicates scaling drift, not classical overfitting

---

### Epsilon Behavior
- Epsilon is successfully learned
- Exhibits structured spatial patterns
- However:
  - Similar across components
  - Does not induce distinct strategies

---

### Policy Behavior
- Inferred policies are nearly identical across components
- No behavioral separation

---

### Visitation Distributions
- Minor variation across components
- No clear decomposition into distinct modes

---

## Final Conclusion

Across all settings:

- Reward recovery is robust
- Expertise estimation is not identifiable
- Learned components collapse to shared behavior

Scaling ambiguity persists even with heterogeneous data and learned epsilon.

---

## Files

- run_mix.py: main experiment script
- src/mix_irl/: IRLEED implementation
- gridworld_simple/: experiment outputs

---

## Notes

- Results in the report and poster correspond to multi-seed aggregated runs
- No additional tools or notebooks are required to reproduce results
- A small number of seeds may fail due to numerical instability
