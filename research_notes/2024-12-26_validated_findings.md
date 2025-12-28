# Validated Research Findings

**Date:** 2024-12-26
**Environment:** CartPole-v1 (500 max steps) - **Standard Gymnasium 1.2.3**
**Goal:** Validate all previous research findings with correct environment

## Critical Discovery: Previous Tests Used Broken Environment

### The Problem

Previous research notes (2024-12-25 and 2024-12-26) were conducted using a **custom, broken Gymnasium installation** located at `/home/jackavery/Documents/inverted-pendulum/Gymnasium`.

This custom environment had critical modifications to CartPole:

```python
# In custom cartpole.py:
force = 0  # Line 178: Actions have NO effect!
terminated = False  # Line 258: Episode NEVER fails!
x = max(-1, min(1, x))  # Lines 211, 241: Cart clamped
```

### Impact on Findings

| Previous Finding | Status | Explanation |
|-----------------|--------|-------------|
| "Instant convergence at Episode 4" | **INVALID** | Impossible to fail in broken env |
| "500 steps from Episode 0" | **INVALID** | Episodes never terminated |
| "Errors explode to inf" | **PARTIALLY VALID** | True, but masked by impossible-to-fail env |
| "Replay buffers harm performance" | **VALID** | Re-confirmed with correct env |
| "Goldilocks stabilizes errors" | **VALID** | Re-confirmed with correct env |

## Validated Results (Correct Gymnasium 1.2.3)

### Test Configuration
- **Gymnasium version:** 1.2.3 (standard, from PyPI)
- **Episodes per run:** 100
- **Runs per configuration:** 3
- **Convergence criteria:** 5 consecutive 500-step episodes (2500 cumulative)

### 1. Baseline Performance

| Metric | Value |
|--------|-------|
| Final avg (last 10 ep) | 242.9 ± 50.0 |
| Max episode length | 500 |
| Convergence rate | 0% (in 100 episodes) |

**Key insight:** Agent CAN achieve 500-step episodes but doesn't reliably converge within 100 episodes. The task is genuinely challenging.

### 2. Replay Buffer Impact

| Configuration | Final Avg | Change vs Baseline |
|--------------|-----------|-------------------|
| Baseline (online) | 242.6 ± 36.2 | — |
| Replay both modules | 122.7 ± 62.0 | **-49.4%** |
| Local targets | 95.8 ± 16.9 | **-60.5%** |
| Buffer states + global goal | 49.4 ± 11.0 | **-79.6%** |

**CONFIRMED:** Replay buffers catastrophically harm this architecture.

**Root cause (unchanged):**
1. Prediction module used as "differentiable simulator" for gradient-based policy optimization
2. Buffer samples are from OLD policy distributions
3. Prediction accuracy degrades for current policy actions
4. Gradient bridge becomes inaccurate

### 3. Gradient Clipping

| Clip Norm | Final Avg | Std Dev |
|-----------|-----------|---------|
| None | 224.8 | **70.6** |
| 0.5 | 231.7 | 34.7 |
| 1.0 | 188.9 | 36.9 |
| 2.0 | 201.4 | **21.4** |
| 5.0 | 181.5 | 23.5 |

**Key insight:** Gradient clipping **reduces variance** but doesn't significantly improve average performance. Previous claim of "~15% improvement" was an artifact.

**Recommendation:** Use `clip_norm=2.0` for lowest variance.

### 4. Learning Rate Strategies

| Strategy | Final Avg | Max Error |
|----------|-----------|-----------|
| Baseline | 212.2 ± 26.6 | 1.78e+02 |
| LR Decay | 227.2 ± 74.3 | 2.19e+02 |
| Goldilocks | 200.1 ± 83.8 | **4.14e+01** |
| Surprise | **24.3 ± 5.7** | **5.00e+11** |

**CONFIRMED:**
- Goldilocks strategy keeps errors stable (4.14e+01 vs 1.78e+02)
- Surprise strategy causes catastrophic failure (errors explode to 5e11)
- "Episode 4 convergence" was invalid (broken environment artifact)

### 5. Learning Rate Values

| LR | Final Avg | Max Episode | Stability |
|----|-----------|-------------|-----------|
| 0.01 | 25.1 ± 22.1 | 54 | Too slow |
| 0.05 | 102.1 ± 66.4 | 117 | High variance |
| 0.1 | 169.4 ± 8.9 | 187 | **Most stable** |
| 0.5 | 150.6 ± 76.1 | **500** | Can reach max |
| 1.0 | 21.2 ± 4.1 | 109 | Too aggressive |

**Key insight:** LR=0.5 can achieve max episode length but has high variance. LR=0.1 is more stable but performance capped at ~180 steps.

### 6. Updates Per Step (NEW FINDING)

| Updates | Final Avg | Max Episode | Convergence Rate |
|---------|-----------|-------------|------------------|
| 1 | 137.4 ± 7.6 | 292 | 0% |
| 3 | **274.5 ± 66.0** | 500 | **33%** |
| 5 | 237.3 ± 19.1 | 500 | 0% |
| 10 | 169.0 ± 12.9 | 461 | 0% |

**NEW DISCOVERY:** 3 updates per step is optimal!
- Highest average performance (274.5)
- Only configuration to achieve convergence (33%)
- 5 updates (default) is slightly suboptimal

## Updated Recommendations

### Optimal Configuration

```python
TrainingConfig(
    action_update_steps=3,  # NEW: 3 is optimal, not 5
    prediction_update_steps=3,
    action_learning_rate=0.5,
    prediction_learning_rate=0.5,
    use_adaptive_lr=True,
    gradient_clip_norm=2.0,  # For variance reduction
    use_replay_buffer=False,  # Critical: must be online
)
```

### Summary of Valid Findings

| Finding | Status | Notes |
|---------|--------|-------|
| Online learning required | **VALID** | Replay buffers cause -49% to -80% degradation |
| Gradient clipping helps | **PARTIALLY** | Reduces variance, not average performance |
| Goldilocks stabilizes errors | **VALID** | Confirmed with correct environment |
| Surprise strategy fails | **VALID** | Catastrophic error explosion confirmed |
| Instant convergence | **INVALID** | Artifact of broken environment |
| 3 updates optimal | **NEW** | Previously untested, discovered in validation |

### Future Research Directions

1. **Hyperparameter optimization:** Systematic search over update steps and learning rates
2. **Longer training:** Run 500+ episodes to measure true convergence rates
3. **Different environments:** Validate on Acrobot, MountainCar, Pendulum
4. **Target networks:** May enable safe replay buffer usage
5. **Goldilocks tuning:** Optimize target_surprise and width parameters

## Visualizations

See `output/validation/` for:
- `comprehensive_validation.png` - Summary of all validation results
- `environment_comparison.png` - Broken vs correct environment comparison
- `validation_summary.json` - Raw data

## Code Locations

- `examples/validate_research_findings.py` - Main validation script
- `examples/validate_replay_buffer.py` - Replay buffer validation
- `examples/create_validation_plots.py` - Plot generation
- Custom broken gymnasium: `/home/jackavery/Documents/inverted-pendulum/Gymnasium/`
- Standard gymnasium: `venv/lib/python3.10/site-packages/gymnasium/`

## Lessons Learned

1. **Always verify environment installation** - System Python vs venv can have different packages
2. **Editable installs are dangerous** - `pip install -e` can silently override standard packages
3. **"Too good" results should be investigated** - Instant convergence was a red flag
4. **Validate assumptions** - The agent wasn't learning, it was impossible to fail
