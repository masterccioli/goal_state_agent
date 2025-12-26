# Replay Buffer Investigation for Goal State Agent

**Date:** 2024-12-25
**Environment:** CartPole-v1 (500 max steps)
**Goal:** Stabilize training for longer episodes

## Problem Statement

The goal state agent architecture experiences instability on CartPole-v1 compared to CartPole-v0:
- CartPole-v0: 200 max steps, converges reliably
- CartPole-v1: 500 max steps, high variance, catastrophic drops

**Root cause:** Long episodes accumulate many weight updates (500 steps × 5 updates × 2 modules = 5000 updates), causing weight explosion.

## Approaches Tested

### 1. Experience Replay Buffer (Standard Approach)

**Hypothesis:** Break temporal correlation by sampling random past transitions.

**Implementation:**
- `TransitionBuffer`: stores `(state, action, next_state)` for prediction module
- `GoalTransitionBuffer`: stores `(state, next_state, goal_distance)` for action policy

**Results:**

| Configuration | Avg Final Steps | Notes |
|--------------|-----------------|-------|
| Baseline (online) | ~200 | High variance |
| Buffer for both modules | ~12 | Complete failure |
| Buffer for prediction only | ~12 | Complete failure |
| Small recency buffer (50) | ~50 | Still broken |

**Conclusion:** Replay buffers catastrophically harm this architecture.

### 2. Local Targets (Achieved Next-States)

**Hypothesis:** Use achieved `next_state` as target instead of global goal. These are "achievable" because they actually happened.

**Implementation:**
- Sample `(state, next_state)` pairs where `next_state` was close to goal
- Train action policy: `P(state, π(state))` should match `next_state`

**Results:** Avg ~70-120 steps (worse than baseline)

**Analysis:** Training toward OLD policy's achievements limits the CURRENT policy from discovering better trajectories.

### 3. Buffer States + Global Goal

**Hypothesis:** Sample diverse states from buffer but still optimize toward global goal.

**Results:** Avg ~40-70 steps (worse than baseline)

**Analysis:** Linear action policy cannot generalize across diverse states simultaneously.

### 4. Gradient Clipping Only

**Hypothesis:** Limit gradient magnitude to prevent weight explosion.

**Results:**
| clip_norm | Avg Final Steps | Convergence Rate |
|-----------|-----------------|------------------|
| None | ~180 | 1/10 |
| 0.5 | ~260 | 1/5 |
| 1.0 | ~280 | 2/10 |
| 2.0 | ~290 | 2/10 |

**Conclusion:** Gradient clipping provides ~15% improvement, modest but consistent.

## Root Cause Analysis

### Why Replay Buffers Fail

1. **Prediction Module Coupling**
   - Action policy gradients flow THROUGH prediction module weights
   - Formula: `∂cost/∂W_action = ∂cost/∂predicted × ∂predicted/∂action × ∂action/∂W_action`
   - The `∂predicted/∂action` term comes from prediction module weights

2. **Distribution Shift**
   - Prediction module trained on: `(state, action_old) → next_state`
   - Action policy computes: `P(state, π_current(state))`
   - If `π_current ≠ π_old`, prediction may be inaccurate

3. **Online Learning Advantage**
   - Prediction module always trained on MOST RECENT transition
   - Stays tuned to current policy's action distribution
   - Gradient bridge remains accurate

4. **Linear Policy Limitation**
   - Linear policy makes tradeoffs between states
   - Training on diverse past states = suboptimal compromise
   - Online learning = specialize for current state distribution

### Experimental Verification

Traced through training dynamics:

**With buffer (broken):**
- Episode 0: 44 steps, weights A=3.8 P=2.7
- Episode 19: 47 steps, weights A=5.6 P=5.6
- NO improvement over 20 episodes

**Without buffer (working):**
- Episode 0: 10 steps, weights A=2.7 P=3.8
- Episode 19: 500 steps, weights A=9.6 P=10.6
- Clear learning progression

## Recommendations

### For This Architecture

1. **Keep online learning** - essential for gradient coupling
2. **Enable gradient clipping** - `gradient_clip_norm=1.0`
3. **Do NOT use replay buffers** - fundamentally incompatible

### For Future Research

If replay buffers are desired, consider:
1. **Target networks** (like DQN) - separate prediction module for gradient computation
2. **Importance sampling** - weight old transitions by policy similarity
3. **Model-based replay** - use prediction module to generate synthetic transitions

## Code Location

- `utils/replay_buffer.py`: TransitionBuffer, GoalTransitionBuffer classes
- `agents/goal_state_agent.py`: AgentConfig options, update methods
- `configs/config.py`: TrainingConfig with buffer settings (disabled by default)

## Key Insight

> The goal state agent architecture uses the prediction module as a "differentiable simulator" for gradient-based policy optimization. This tight coupling requires the prediction module to be accurate for the CURRENT policy's actions, which online learning provides but replay sampling destroys.
