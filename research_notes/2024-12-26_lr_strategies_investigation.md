# Learning Rate Strategies Investigation

**Date:** 2024-12-26
**Environment:** CartPole-v1 (500 max steps)
**Goal:** Compare different learning rate scheduling strategies for stability and performance

## Strategies Tested

### 1. Baseline (No Scheduling)
- Uses adaptive LR (gradient normalization) + gradient clipping
- Learning rate: 0.5 for both modules

### 2. Exponential Decay
- `lr = base_lr * decay_rate^(step / decay_steps)`
- Parameters: decay_rate=0.9995, decay_steps=100

### 3. Goldilocks (Surprise-Based)
- Bell curve centered on target_surprise
- Low surprise → low LR
- Moderate surprise → high LR (optimal learning)
- High surprise → low LR (too chaotic)
- Parameters: target_surprise=0.05, width=0.05

### 4. Inverse Surprise
- Higher prediction error → higher learning rate
- `lr = base_lr * (baseline + scale * tanh(surprise / saturation))`
- Parameters: scale=1.5, saturation=0.5, baseline=0.5

## Key Finding: Numerical Instability

### The Problem
During long episodes (500 steps), the prediction module's predictions diverge from reality:

```
Step 0:  goal_err=9.2e-04, pred_err=0.0
Step 55: goal_err=6.5e+06, pred_err=2.4e+05
Step 57: goal_err=3.1e+50, pred_err=2.2e+14
Step 58: goal_err=inf,     pred_err=inf
```

The errors explode from ~1e-3 to infinity in just ~60 steps!

### Root Cause Analysis
1. **Long episode accumulation**: 500 steps × 5 updates × 2 modules = 5000 weight updates per episode
2. **Feedback loop**: Prediction errors feed into action policy gradients, which change actions, which change predictions
3. **Gradient clipping only clips gradients, not the loss values themselves**

### Why the Agent Still Works
Despite `inf` errors, the agent achieves 500-step episodes because:
1. **Gradient clipping** prevents weight explosion
2. **Action selection** is a simple threshold on raw output - robust to large internal values
3. **The policy learned quickly** (first few episodes) before instability set in

## Strategy Comparison Results

| Strategy | Convergence | Error Behavior |
|----------|-------------|----------------|
| Baseline | Episode 4 | Explodes to 5e11 (clipped) |
| LR Decay | Episode 4 | Explodes to 5e11 (clipped) |
| Goldilocks | Episode 4 | Stays in range 10-170 MSE |
| Surprise | Episode 4 | Explodes to 5e11 (clipped) |

### Critical Discovery: Goldilocks Prevents Instability

The Goldilocks strategy is the ONLY one that maintains stable error values:
- Goal errors: 10-44 MSE (vs 5e11 for others)
- Prediction errors: 10-170 MSE (vs 5e11 for others)

**Why it works:**
When prediction errors become large (surprising), Goldilocks REDUCES the learning rate:
```
lr_multiplier = exp(-((surprise - target) / width)^2)
```
If surprise >> target, multiplier → 0, preventing further destabilization.

## Fixes Implemented

### 1. MSE Loss Overflow Protection
Added `max_diff` parameter to clip differences before squaring:
```python
diff = np.clip(diff, -self.max_diff, self.max_diff)  # Default: 1e6
squared = 0.5 * np.power(diff, 2)  # Max: 5e11 instead of inf
```

### 2. Plot Handling for Inf/NaN
Updated visualization to filter non-finite values:
```python
goal_errors_clean = [e for e in goal_errors if np.isfinite(e)]
```

## Recommendations

### For This Architecture
1. **Use Goldilocks scheduling** - provides natural stability mechanism
2. **Keep gradient clipping** - essential backup protection
3. **Consider lower base learning rates** for long episodes

### For Future Research
1. **Weight decay**: May help prevent accumulation
2. **Episode-based resets**: Reset optimizer state between episodes
3. **Prediction module regularization**: Prevent drift during long episodes
4. **Target networks**: Like DQN, use frozen copy for stability

## Visualizations

See `output/lr_comparison/` for:
- `comparison.png` - Side-by-side strategy comparison
- `{strategy}/training_metrics.png` - Individual strategy details

## Code Locations

- `utils/optimizers.py`: LRScheduler classes (ExponentialDecay, SurpriseBasedLR, InverseSurpriseLR)
- `core/losses.py`: MSELoss with overflow protection
- `agents/goal_state_agent.py`: `_get_scheduled_lr()` method
- `examples/compare_lr_strategies.py`: Comparison script
- `examples/visualize_lr_comparison.py`: Visualization script

## Key Insight

> The Goldilocks learning rate strategy acts as an automatic stability mechanism: when the model becomes "too surprised" (high prediction error), it reduces learning rate to prevent runaway feedback loops. This makes it uniquely suitable for this tightly-coupled architecture where prediction errors directly influence action policy updates.

