# Alternative Architectures for Goal State Agent

**Date:** 2024-12-26
**Context:** Investigating why the agent unlearns and exploring solutions

## The Core Problem: Catastrophic Forgetting in Online Learning

### Evidence from Analysis

```
Checkpoint | Training Length | Frozen Policy Performance
ep 30      | 500 steps       | 500.0±0.0 (PERFECT!)
ep 50      | 477 steps       | 299.3±80.7
ep 99      | 471 steps       | 151.6±118.9
```

The agent learns a **perfect policy by episode 30** but then forgets it through continued training.

### Root Cause

1. **State distribution shift**: During long balanced episodes, the agent only sees "easy" states (low variance in pole angle/velocity)
2. **Continuous weight updates**: Each step modifies weights toward current (easy) states
3. **No protection for learned knowledge**: All weights are equally plastic

## Solution 1: Early Stopping / Weight Freezing

### Concept
Stop learning when a performance threshold is reached.

### Implementation
```python
class AdaptiveFreezing:
    def __init__(self, freeze_threshold=450, window=5):
        self.freeze_threshold = freeze_threshold
        self.window = window
        self.frozen = False

    def should_freeze(self, recent_lengths):
        if len(recent_lengths) >= self.window:
            avg = np.mean(recent_lengths[-self.window:])
            if avg >= self.freeze_threshold:
                self.frozen = True
        return self.frozen
```

### Pros/Cons
- ✅ Simple, preserves perfect policy
- ✅ Zero computational overhead once frozen
- ❌ Cannot adapt to environment changes
- ❌ Requires tuning freeze threshold

---

## Solution 2: Instance-Based Learning (k-NN)

### Concept
Instead of learning weights, store (state, action, outcome) tuples and use nearest neighbors for action selection.

### Why It's Appealing for This Problem
1. **No forgetting**: Past experiences are stored explicitly
2. **Interpretable**: Can inspect which past experiences drive decisions
3. **Continuous learning**: New experiences added without modifying old ones
4. **Natural handling of non-stationary environments**

### Architecture Options

#### 2a. k-NN Action Policy
```python
class KNNPolicy:
    def __init__(self, k=5):
        self.experiences = []  # [(state, action, reward)]
        self.k = k

    def add_experience(self, state, action, reward):
        self.experiences.append((state, action, reward))

    def get_action(self, state):
        # Find k nearest neighbors
        distances = [np.linalg.norm(state - exp[0]) for exp in self.experiences]
        nearest_idx = np.argsort(distances)[:self.k]

        # Weight by distance and outcome
        action_votes = {}
        for idx in nearest_idx:
            exp = self.experiences[idx]
            weight = 1.0 / (distances[idx] + 1e-6) * exp[2]  # reward-weighted
            action_votes[exp[1]] = action_votes.get(exp[1], 0) + weight

        return max(action_votes, key=action_votes.get)
```

#### 2b. Locally Weighted Regression (LWR)
```python
class LocallyWeightedPolicy:
    """Fit linear model locally using weighted neighbors."""

    def __init__(self, bandwidth=0.1):
        self.experiences = []  # [(state, action)]
        self.bandwidth = bandwidth

    def get_action(self, query_state):
        # Gaussian kernel weights
        weights = []
        for state, action in self.experiences:
            dist = np.linalg.norm(query_state - state)
            w = np.exp(-dist**2 / (2 * self.bandwidth**2))
            weights.append(w)

        # Weighted least squares to fit local linear model
        # ... solve for action that minimizes goal distance
```

#### 2c. Case-Based Reasoning with Goal Awareness
```python
class GoalAwareCBR:
    """Store transitions with goal distance improvement."""

    def __init__(self, goal_indices, goal_values):
        self.cases = []  # [(state, action, next_state, goal_improvement)]
        self.goal_indices = goal_indices
        self.goal_values = goal_values

    def add_case(self, state, action, next_state):
        curr_dist = self._goal_distance(state)
        next_dist = self._goal_distance(next_state)
        improvement = curr_dist - next_dist  # Positive = got closer
        self.cases.append((state, action, next_state, improvement))

    def get_action(self, query_state, k=10):
        # Find similar past states
        similar = self._find_similar(query_state, k)

        # Prefer actions that historically improved goal distance
        best_action = None
        best_score = float('-inf')

        for state, action, next_state, improvement in similar:
            similarity = 1.0 / (np.linalg.norm(query_state - state) + 1e-6)
            score = similarity * improvement
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
```

### Pros/Cons
- ✅ **No catastrophic forgetting** - experiences stored explicitly
- ✅ **Interpretable** - can trace decisions to specific experiences
- ✅ **Handles non-stationarity** - new experiences don't overwrite old
- ❌ **Memory grows** - O(n) storage where n = total experiences
- ❌ **Query time** - O(n) or O(log n) with spatial indexing
- ❌ **Curse of dimensionality** - distance metrics less meaningful in high-D
- ❌ **No gradient flow** - breaks the goal-state architecture's core mechanism

### Hybrid Approach: Instance-Based + Goal-State Gradient
```python
class HybridGoalStateAgent:
    """Combine instance memory with gradient-based optimization."""

    def __init__(self):
        self.memory = CaseMemory()
        self.action_policy = LinearLayer(...)
        self.prediction_module = LinearLayer(...)

    def get_action(self, state):
        # Start with instance-based suggestion
        memory_action = self.memory.suggest_action(state)

        if memory_action is not None and self.memory.confidence(state) > 0.8:
            return memory_action
        else:
            # Fall back to gradient-based policy
            return self.action_policy.forward(state)

    def update(self, state, action, next_state):
        # Always store in memory
        self.memory.add(state, action, next_state)

        # Only update weights if memory is uncertain about this region
        if self.memory.confidence(state) < 0.5:
            self._gradient_update(state, action, next_state)
```

---

## Solution 3: Elastic Weight Consolidation (EWC)

### Concept
Identify "important" weights and penalize changes to them.

### How It Works
1. After learning a good policy, compute Fisher Information for each weight
2. Weights with high Fisher Information are "important"
3. Add penalty term: `L_total = L_task + λ * Σ F_i * (θ_i - θ*_i)²`

### Implementation Sketch
```python
class EWCAgent:
    def __init__(self):
        self.action_policy = LinearLayer(...)
        self.fisher_information = None
        self.optimal_weights = None

    def consolidate(self):
        """Call after learning good policy."""
        # Compute Fisher Information (diagonal approximation)
        # This identifies which weights were crucial for the task
        self.fisher_information = self._compute_fisher()
        self.optimal_weights = self.action_policy.W.copy()

    def compute_loss(self, prediction, target):
        task_loss = mse(prediction, target)

        if self.fisher_information is not None:
            # Penalize deviation from consolidated weights
            ewc_loss = 0.5 * np.sum(
                self.fisher_information * (self.action_policy.W - self.optimal_weights)**2
            )
            return task_loss + self.lambda_ewc * ewc_loss
        return task_loss
```

### Pros/Cons
- ✅ Principled approach from neuroscience
- ✅ Maintains plasticity for unimportant weights
- ❌ Requires knowing when to consolidate
- ❌ Fisher computation can be expensive
- ❌ May still forget if task distribution changes significantly

---

## Solution 4: Dual-Memory / Fast-Slow Weights

### Concept
Maintain two sets of weights:
- **Fast weights**: Highly plastic, adapt quickly to current situation
- **Slow weights**: Stable, encode long-term knowledge

### Implementation
```python
class DualMemoryAgent:
    def __init__(self, fast_lr=0.5, slow_lr=0.01, blend=0.7):
        self.fast_weights = init_weights()
        self.slow_weights = init_weights()
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.blend = blend

    def get_action(self, state):
        # Blend fast and slow predictions
        fast_action = state @ self.fast_weights
        slow_action = state @ self.slow_weights
        return self.blend * slow_action + (1 - self.blend) * fast_action

    def update(self, gradient):
        # Fast weights: quick adaptation
        self.fast_weights -= self.fast_lr * gradient

        # Slow weights: gradual consolidation
        # Could also use EMA: slow = 0.99 * slow + 0.01 * fast
        self.slow_weights -= self.slow_lr * gradient
```

### Pros/Cons
- ✅ Best of both worlds: stability + plasticity
- ✅ Simple to implement
- ❌ Requires tuning blend ratio
- ❌ Slow weights may still drift over very long training

---

## Solution 5: Episodic Control with Sparse Updates

### Concept
Only update weights when encountering "novel" states (far from known experiences).

### Implementation
```python
class NoveltyBasedAgent:
    def __init__(self, novelty_threshold=0.1):
        self.state_memory = []
        self.novelty_threshold = novelty_threshold

    def is_novel(self, state):
        if not self.state_memory:
            return True
        distances = [np.linalg.norm(state - s) for s in self.state_memory]
        return min(distances) > self.novelty_threshold

    def update(self, state, action, next_state):
        if self.is_novel(state):
            self.state_memory.append(state)
            self._gradient_update(state, action, next_state)
        # else: skip update, state already well-covered
```

### Pros/Cons
- ✅ Prevents over-training on common states
- ✅ Maintains diverse state coverage
- ❌ Memory grows (though more slowly)
- ❌ May miss important updates in "familiar" regions

---

## Recommendation: Start with Simple Solutions

Given the evidence, I recommend trying solutions in this order:

### Tier 1: Immediate (simplest)
1. **Early stopping with weight freezing**
   - Freeze weights when rolling avg > 450 for 5 episodes
   - Test: Does frozen policy maintain performance indefinitely?

2. **Performance-based learning rate**
   - `lr = base_lr * (1 - avg_episode_length / 500)`
   - More successful → less learning

### Tier 2: Instance-based (user interest)
3. **Hybrid: Instance memory + gradient policy**
   - Store good (state, action) pairs
   - Use memory when confident, gradient when uncertain
   - Best of both worlds

4. **Pure k-NN for comparison**
   - See how well pure instance-based works
   - Baseline for hybrid approaches

### Tier 3: More complex
5. **Dual-memory (fast/slow weights)**
6. **EWC after first convergent episode**

## Expected Outcomes

| Solution | Implementation Effort | Expected Improvement |
|----------|----------------------|---------------------|
| Early stopping | Low | High (maintains perfect policy) |
| Performance-LR decay | Low | Medium |
| Hybrid instance+gradient | Medium | High |
| Pure k-NN | Medium | Unknown (different paradigm) |
| Dual-memory | Medium | Medium-High |
| EWC | High | Medium |

## Key Insight

> The goal-state architecture with gradient flow through the prediction module is elegant but inherently unstable for online learning. The tight coupling that enables learning also enables forgetting. Instance-based methods or weight freezing may be necessary complements.
