#!/usr/bin/env python3
"""Analyze why the agent unlearns during long episodes.

Key hypothesis: The agent learns a good policy early, but continued
weight updates during long successful episodes cause it to "forget"
how to recover from disturbed states.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import gymnasium as gym
from goal_state_agent import Trainer, TrainingConfig
from goal_state_agent.agents.goal_state_agent import GoalStateAgent, AgentConfig


def test_frozen_policy(agent: GoalStateAgent, num_episodes: int = 10) -> dict:
    """Test agent with frozen weights (no learning)."""
    env = gym.make("CartPole-v1")
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=100 + ep)
        steps = 0

        for _ in range(500):
            action = agent.get_action(state, training=False)
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break

        episode_lengths.append(steps)

    env.close()
    return {
        "mean": np.mean(episode_lengths),
        "std": np.std(episode_lengths),
        "min": min(episode_lengths),
        "max": max(episode_lengths),
        "lengths": episode_lengths,
    }


def analyze_policy_at_checkpoints():
    """Save checkpoints during training and test frozen performance."""
    print("="*70)
    print("EXPERIMENT 1: Frozen Policy Performance at Checkpoints")
    print("="*70)
    print("Training agent and saving weights at key episodes...")

    config = TrainingConfig(
        max_episodes=100,
        action_update_steps=3,
        prediction_update_steps=3,
        use_adaptive_lr=True,
        gradient_clip_norm=2.0,
        seed=42,
    )

    trainer = Trainer(config=config)
    agent = trainer.agent

    env = gym.make("CartPole-v1")
    checkpoints = {}
    episode_lengths = []

    for ep in range(100):
        state, _ = env.reset(seed=42 + ep)
        agent.reset()
        steps = 0

        for _ in range(500):
            action, goal_error, pred_error = agent.step(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            steps += 1
            if terminated or truncated:
                break

        episode_lengths.append(steps)

        # Save checkpoints at key moments
        checkpoint_eps = [5, 10, 15, 20, 30, 50, 75, 99]
        if ep in checkpoint_eps:
            action_W, pred_W = agent.get_weights()
            checkpoints[ep] = {
                "action_weights": action_W.copy(),
                "prediction_weights": pred_W.copy(),
                "episode_length": steps,
            }
            print(f"  Saved checkpoint at episode {ep} (length: {steps})")

        # Also save if we hit a good episode
        if steps >= 400 and "first_400" not in checkpoints:
            action_W, pred_W = agent.get_weights()
            checkpoints["first_400"] = {
                "action_weights": action_W.copy(),
                "prediction_weights": pred_W.copy(),
                "episode_length": steps,
                "episode": ep,
            }
            print(f"  Saved 'first_400' checkpoint at episode {ep} (length: {steps})")

    env.close()

    # Test each checkpoint with frozen weights
    print("\n" + "-"*50)
    print("Testing frozen policy at each checkpoint (10 test episodes):")
    print("-"*50)

    results = {}
    for key, ckpt in sorted(checkpoints.items(), key=lambda x: (isinstance(x[0], str), x[0])):
        # Create agent config matching training
        agent_config = AgentConfig(
            state_dim=4,
            action_dim=1,
            goal_indices=np.array([2, 3]),
            goal_values=np.array([0.0, 0.0]),
        )
        test_agent = GoalStateAgent(config=agent_config)
        test_agent.set_weights(ckpt["action_weights"], ckpt["prediction_weights"])

        perf = test_frozen_policy(test_agent, num_episodes=10)
        results[key] = perf

        label = f"ep {key:2d}" if isinstance(key, int) else f"{key}"
        print(f"  {label:12s} (trained {ckpt['episode_length']:3d} steps): "
              f"frozen_avg={perf['mean']:.1f}±{perf['std']:.1f}, "
              f"min={perf['min']}, max={perf['max']}")

    return results, episode_lengths, checkpoints


def analyze_weight_magnitude():
    """Track how weight magnitude changes during training."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Weight Magnitude Over Training")
    print("="*70)

    config = TrainingConfig(
        max_episodes=100,
        action_update_steps=3,
        prediction_update_steps=3,
        use_adaptive_lr=True,
        gradient_clip_norm=2.0,
        seed=42,
    )

    trainer = Trainer(config=config)
    agent = trainer.agent

    env = gym.make("CartPole-v1")
    weight_data = []

    for ep in range(100):
        action_W, pred_W = agent.get_weights()
        start_action_norm = np.linalg.norm(action_W)
        start_pred_norm = np.linalg.norm(pred_W)

        state, _ = env.reset(seed=42 + ep)
        agent.reset()
        steps = 0

        for _ in range(500):
            action, _, _ = agent.step(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            steps += 1
            if terminated or truncated:
                break

        action_W, pred_W = agent.get_weights()
        end_action_norm = np.linalg.norm(action_W)
        end_pred_norm = np.linalg.norm(pred_W)

        weight_data.append({
            "episode": ep,
            "steps": steps,
            "action_norm_start": start_action_norm,
            "action_norm_end": end_action_norm,
            "action_change": end_action_norm - start_action_norm,
            "pred_norm_end": end_pred_norm,
        })

    env.close()

    # Analyze correlation
    steps_arr = np.array([w["steps"] for w in weight_data])
    changes_arr = np.array([w["action_change"] for w in weight_data])

    # Find long episodes (200+ steps)
    long_eps = [w for w in weight_data if w["steps"] >= 200]
    short_eps = [w for w in weight_data if w["steps"] < 50]

    print(f"\nLong episodes (200+ steps): {len(long_eps)}")
    if long_eps:
        avg_change = np.mean([w["action_change"] for w in long_eps])
        print(f"  Average weight change: {avg_change:.4f}")

    print(f"\nShort episodes (<50 steps): {len(short_eps)}")
    if short_eps:
        avg_change = np.mean([w["action_change"] for w in short_eps])
        print(f"  Average weight change: {avg_change:.4f}")

    # Correlation
    if len(steps_arr) > 1:
        corr = np.corrcoef(steps_arr, changes_arr)[0, 1]
        print(f"\nCorrelation(episode_length, weight_change): {corr:.3f}")

    return weight_data


def analyze_state_distribution():
    """Analyze how state distribution changes during long vs short episodes."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: State Distribution Analysis")
    print("="*70)

    env = gym.make("CartPole-v1")

    # Collect states from random policy (diverse states)
    random_states = []
    for _ in range(20):
        state, _ = env.reset()
        for _ in range(50):
            random_states.append(state.copy())
            action = env.action_space.sample()
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    # Collect states from a "balanced" policy (trained agent in steady state)
    config = TrainingConfig(
        max_episodes=30,
        action_update_steps=3,
        prediction_update_steps=3,
        use_adaptive_lr=True,
        gradient_clip_norm=2.0,
        seed=42,
    )
    trainer = Trainer(config=config)
    trainer.train(verbose=False)

    agent = trainer.agent
    balanced_states = []

    for _ in range(5):
        state, _ = env.reset(seed=200)
        agent.reset()
        for _ in range(500):
            action = agent.get_action(state, training=False)
            balanced_states.append(state.copy())
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    env.close()

    if len(balanced_states) == 0:
        print("Agent couldn't balance - no balanced states to analyze")
        return

    random_arr = np.array(random_states)
    balanced_arr = np.array(balanced_states)

    print("\nState Distribution Comparison:")
    print("-" * 60)
    labels = ["cart_pos", "cart_vel", "pole_angle", "pole_vel"]
    print(f"{'Variable':12s} {'Random σ':>12s} {'Balanced σ':>12s} {'Ratio':>10s}")
    print("-" * 60)
    for i, label in enumerate(labels):
        random_std = np.std(random_arr[:, i])
        balanced_std = np.std(balanced_arr[:, i])
        ratio = random_std / max(balanced_std, 1e-6)
        print(f"{label:12s} {random_std:12.4f} {balanced_std:12.4f} {ratio:10.1f}x")

    print("\nKey insight: During balanced episodes, state variance is MUCH lower.")
    print("The agent only sees 'easy' states and forgets 'hard' recovery states.")


def main():
    print("="*70)
    print("LEARNING DYNAMICS ANALYSIS")
    print("="*70)
    print("Investigating why the agent 'unlearns' during long episodes\n")

    # Run analyses
    frozen_results, episode_lengths, checkpoints = analyze_policy_at_checkpoints()
    weight_data = analyze_weight_magnitude()
    analyze_state_distribution()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: The Stability-Plasticity Dilemma")
    print("="*70)

    print("""
THE CORE PROBLEM:

1. EARLY TRAINING (episodes 1-20):
   - Agent sees diverse states (falling, recovering, balanced)
   - Learns robust policy that handles ALL situations
   - When frozen, this policy works WELL

2. LONG EPISODES (200+ steps):
   - Agent only sees "easy" balanced states (low variance)
   - Weights keep updating toward these easy states
   - Weights DRIFT away from recovery policy
   - More updates = more forgetting

3. AFTER DISTURBANCE:
   - Agent encounters "hard" states it hasn't seen recently
   - Current weights optimized for balanced states ONLY
   - Policy fails for recovery → episode ends quickly

THIS IS THE STABILITY-PLASTICITY DILEMMA:
- Too much plasticity: forgets old knowledge (current problem)
- Too much stability: can't learn new things

EVIDENCE FROM EXPERIMENTS:
""")

    # Print key findings
    if "first_400" in checkpoints:
        first_good = checkpoints["first_400"]
        ep = first_good.get("episode", "?")
        perf = frozen_results.get("first_400", {})
        print(f"  - First 400+ step episode at ep {ep}")
        print(f"  - When frozen at that point: avg={perf.get('mean', 0):.1f} steps")

    if 99 in frozen_results:
        final_perf = frozen_results[99]
        print(f"  - After 100 episodes of training: frozen avg={final_perf['mean']:.1f} steps")

    print("""
POSSIBLE SOLUTIONS:

1. EARLY STOPPING / WEIGHT FREEZING
   - Stop learning when performance is "good enough"
   - Pros: Simple, preserves good policy
   - Cons: Can't adapt to changes

2. LEARNING RATE DECAY
   - Reduce LR as performance improves
   - Tested: Goldilocks helps stability but doesn't prevent drift

3. INSTANCE-BASED LEARNING (user interested)
   - Store (state, action, outcome) tuples
   - k-NN: find similar past states, use their actions
   - Pros: NO forgetting, can add new experiences
   - Cons: Memory grows, retrieval time, curse of dimensionality

4. ELASTIC WEIGHT CONSOLIDATION (EWC)
   - Protect "important" weights from changing
   - Pros: Reduces catastrophic forgetting
   - Cons: Needs to identify which weights matter

5. DUAL-MEMORY SYSTEM
   - Fast weights for current learning
   - Slow weights for stable knowledge
   - Pros: Best of both worlds
   - Cons: More complex architecture

6. EPISODIC MEMORY (hybrid)
   - Store key experiences explicitly
   - Replay during training to prevent forgetting
   - Pros: Maintains diverse state coverage
   - Cons: Similar to replay buffer issues with gradient flow

RECOMMENDED NEXT STEPS:
1. Implement early stopping based on rolling performance
2. Explore instance-based (k-NN) policy as alternative
3. Test weight freezing after first convergent episode
""")


if __name__ == "__main__":
    main()
