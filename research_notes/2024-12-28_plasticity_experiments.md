# Plasticity Experiments: Preventing Catastrophic Forgetting

**Date:** 2024-12-28
**Author:** Jack Avery (with Claude Code assistance)

---

## Background

### The Goal State Architecture

The goal state agent uses a novel architecture for reinforcement learning:

1. **Prediction Module**: Learns to predict next_state from (state, action)
2. **Action Policy**: Learns to select actions that minimize distance to goal state
3. **Gradient Flow**: The key insight is that gradients from the goal error flow *through* the prediction module's weights to update the action policy

This creates a tight coupling where the prediction module serves as a "world model" that enables the action policy to plan toward the goal.

### The Problem: Catastrophic Forgetting

During training on CartPole-v1, we observed a puzzling phenomenon:

1. The agent learns to balance perfectly early in training (often by episode 30)
2. Continued training causes performance to **degrade**
3. By episode 100+, the agent often performs worse than at episode 30

Evidence from our analysis:

```
Episode | Training Length | Frozen Policy Test Performance
--------|-----------------|------------------------------
30      | 500 steps       | 500.0 ± 0.0 (PERFECT!)
50      | 477 steps       | 299.3 ± 80.7
99      | 471 steps       | 151.6 ± 118.9
```

The agent that achieved perfect balance at episode 30 has **forgotten how to balance** by episode 99, despite continued training showing good episode lengths.

### Root Cause Analysis

Investigation revealed the mechanism:

1. **State Distribution Shift**: During long balanced episodes (500 steps), the agent only experiences "easy" states where the pole is nearly upright
2. **Continuous Weight Updates**: Each step modifies weights toward the current (easy) state distribution
3. **Loss of Recovery Knowledge**: The agent forgets how to recover from difficult states (large pole angles/velocities) because it rarely sees them during successful episodes

The training performance metric (episode length) masks this issue because the agent can still achieve long episodes from easy starting conditions, while having lost the ability to recover from perturbations.

---

## Motivation: The Stability-Plasticity Dilemma

This is a classic problem in neural network learning:

- **Plasticity**: The ability to learn new information
- **Stability**: The ability to retain previously learned information

In online learning (like our goal state agent), high plasticity enables quick learning but also quick forgetting. We need a mechanism to reduce plasticity when the agent has "learned enough" while maintaining it for novel situations.

### Initial Hypothesis: Achievement-Gated Plasticity

The goal state architecture provides natural signals about learning progress:

1. **Goal Error**: How close is the current state to the goal? (pole upright, zero velocity)
2. **Prediction Error**: How accurate is the world model?
3. **State Novelty**: Is this a familiar or unfamiliar state?

**Hypothesis**: Reduce plasticity (learning rate) when goal is consistently achieved; increase it when surprised or in novel states.

---

## Experiments

### Strategy 1: Per-Step Goal Error Gating

**Implementation**: Track exponential moving average of per-step goal achievement. Reduce plasticity as achievement increases.

**Result**: FAILED

**Why**: Even failing episodes have many low-error steps before the pole falls. A 150-step episode still has 150 steps where the pole was briefly upright. The metric conflates "pole was upright at this moment" with "can balance indefinitely."

### Strategy 2: Episode-Level Plasticity Gating

**Implementation**: `AchievementGatedPlasticity` class
- Track episode success rate (episodes >= 450 steps count as success)
- Reduce base plasticity as success EMA increases
- Add temporary boosts for novel states and surprising predictions

```python
class AchievementGatedPlasticity:
    def end_episode(self, episode_length):
        is_success = episode_length / 500 >= 0.9
        self.achievement_ema = (1 - alpha) * self.achievement_ema + alpha * is_success
        self.plasticity = min_plasticity + (1 - achievement_ema) * (max - min)
```

**Result**: MIXED (48% perfect rate vs 32% baseline)

**Why**:
- Helps some seeds by preventing over-training
- Hurts others by reducing plasticity before the agent has fully learned
- Difficult to tune thresholds that work across different random seeds

### Strategy 3: Best Checkpoint (Winner)

**Implementation**: `BestCheckpoint` class
- Periodically evaluate current weights on held-out test episodes
- Save checkpoint if it performs better than previous best
- Return best checkpoint at end of training (not final weights)

```python
class BestCheckpoint:
    def maybe_checkpoint(self, agent, test_env):
        if self.episode_count % self.check_interval == 0:
            test_score = self._evaluate(agent, test_env)
            if test_score > self.best_score:
                self.best_weights = agent.get_weights()
                self.best_score = test_score
```

**Result**: EXCELLENT (99.5% perfect rate)

**Why**:
- Evaluates on held-out episodes (not training performance)
- Makes no commitment during training - keeps all options open
- Robust regardless of training dynamics
- Simple with no threshold tuning required

---

## Results

### 10-Seed Validation (150 training episodes, 20 test episodes per seed)

| Strategy | Test Avg | Perfect Rate | Seeds with 20/20 |
|----------|----------|--------------|------------------|
| Baseline | 258.7 | 64/200 (32%) | 2/10 |
| Episode-Gated | 277.1 | 97/200 (48%) | 2/10 |
| **Best Checkpoint** | **499.9** | **199/200 (99.5%)** | **9/10** |

### Per-Seed Breakdown

```
Seed      Baseline    Episode-Gated    Best Checkpoint
------------------------------------------------------
42        500.0 (P)   119.1            498.6
123       144.7       28.9             500.0 (P)
456       239.7       389.2            500.0 (P)
789       159.8       500.0 (P)        500.0 (P)
1000      24.6        64.8             500.0 (P)
2024      500.0 (P)   478.1            500.0 (P)
3000      192.1       500.0 (P)        500.0 (P)
4000      220.9       476.4            500.0 (P)
5000      148.5       204.9            500.0 (P)
6000      456.9       9.7              500.0 (P)

(P) = Perfect 20/20 test episodes
```

Key observations:
- Baseline is highly variable (some seeds perfect, others terrible)
- Episode-gated is also variable and can catastrophically fail (seed 6000: 9.7 avg)
- Best checkpoint is consistently excellent across all seeds

---

## Analysis

### Why Plasticity Gating is Unreliable

1. **Premature Reduction**: May reduce plasticity before the agent has learned to handle all situations
2. **Prevents Refinement**: Even good policies benefit from continued learning on edge cases
3. **Threshold Sensitivity**: What works for one seed may fail for another
4. **Feedback Loop**: Reducing plasticity due to success can prevent learning the final refinements needed for robust performance

### Why Best Checkpoint Works

1. **Decoupled Evaluation**: Tests on held-out episodes, not training performance
2. **No Early Commitment**: Keeps learning at full plasticity throughout
3. **Retrospective Selection**: Chooses the best policy after the fact
4. **Robust to Dynamics**: Works regardless of when/how the agent learns

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Plasticity Gating | No extra computation | Unreliable, needs careful tuning |
| Best Checkpoint | Robust, simple, reliable | Extra evaluation overhead (~5%) |

The overhead of best checkpoint is minimal: evaluating 5 episodes every 10 training episodes adds ~3-5% to training time but delivers dramatically better results.

---

## Implementation

### Files Created

- `utils/plasticity.py`: Contains all plasticity-related utilities
  - `PlasticityMetrics`: Dataclass for tracking metrics
  - `AchievementGatedPlasticity`: Episode-level plasticity gating with novelty boosts
  - `BestCheckpoint`: Periodic checkpoint evaluation and selection
  - `PerformanceGatedPlasticity`: Simpler episode-length based gating

- `examples/test_achievement_gated_plasticity.py`: Comprehensive comparison script

### Recommended Usage

```python
from goal_state_agent.utils.plasticity import BestCheckpoint
import gymnasium as gym

# Create checkpoint tracker
checkpoint = BestCheckpoint(
    check_interval=10,    # Evaluate every 10 episodes
    test_episodes=5,      # Use 5 test episodes per evaluation
    test_seed_offset=1000 # Seed offset for test reproducibility
)

# Create test environment (separate from training)
test_env = gym.make("CartPole-v1")

# Training loop
for ep in range(max_episodes):
    # ... normal training code ...

    # Check for new best checkpoint at end of episode
    score = checkpoint.maybe_checkpoint(agent, test_env)
    if score is not None:
        print(f"Checkpoint at ep {ep}: score={score:.1f}")

# Retrieve best weights
best_weights = checkpoint.get_best_weights()
if best_weights is not None:
    agent.set_weights(best_weights[0], best_weights[1])

print(f"Best checkpoint score: {checkpoint.best_score:.1f}")
```

---

## Conclusions

1. **The stability-plasticity dilemma is real** in the goal state architecture. The tight coupling that enables learning also enables forgetting.

2. **Dynamic plasticity gating is unreliable**. Attempts to predict "when to stop learning" based on performance metrics are sensitive to random seeds and hyperparameters.

3. **Best checkpoint is the robust solution**. By evaluating on held-out episodes and keeping the best weights, we achieve 99.5% perfect rate across diverse seeds.

4. **This aligns with standard practice** in deep RL, where checkpoint selection based on validation performance is common. The goal state architecture's simplicity makes it particularly susceptible to forgetting, but the solution is equally simple.

### Key Insight

> Don't try to predict when to stop learning. Instead, keep track of the best policy seen during training and use that.

**Final improvement: 32% → 99.5% perfect episode rate with a simple checkpoint mechanism.**
