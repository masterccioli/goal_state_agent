#!/usr/bin/env python3
"""Test early stopping with weight freezing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import gymnasium as gym
from goal_state_agent import Trainer, TrainingConfig
from goal_state_agent.agents.goal_state_agent import GoalStateAgent, AgentConfig


def train_with_early_stopping(freeze_threshold=450, window=5, max_episodes=200):
    """Train agent and freeze weights when performance exceeds threshold."""

    config = TrainingConfig(
        max_episodes=max_episodes,
        action_update_steps=3,
        prediction_update_steps=3,
        use_adaptive_lr=True,
        gradient_clip_norm=2.0,
        seed=42,
    )

    trainer = Trainer(config=config)
    agent = trainer.agent
    env = gym.make("CartPole-v1")

    episode_lengths = []
    frozen = False
    freeze_episode = None
    frozen_weights = None

    for ep in range(max_episodes):
        state, _ = env.reset(seed=42 + ep)
        agent.reset()
        steps = 0

        for _ in range(500):
            if frozen:
                # Frozen: just get action, no updates
                action = agent.get_action(state, training=False)
            else:
                # Learning: full step with updates
                action, _, _ = agent.step(state)

            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            steps += 1
            if terminated or truncated:
                break

        episode_lengths.append(steps)

        # Check if we should freeze
        if not frozen and len(episode_lengths) >= window:
            avg = np.mean(episode_lengths[-window:])
            if avg >= freeze_threshold:
                frozen = True
                freeze_episode = ep
                action_W, pred_W = agent.get_weights()
                frozen_weights = (action_W.copy(), pred_W.copy())
                print(f"  FROZEN at episode {ep} (avg last {window}: {avg:.1f})")

    env.close()

    return {
        "episode_lengths": episode_lengths,
        "frozen": frozen,
        "freeze_episode": freeze_episode,
        "frozen_weights": frozen_weights,
        "final_avg": np.mean(episode_lengths[-10:]),
    }


def test_frozen_weights(weights, num_episodes=50):
    """Test frozen weights extensively."""
    agent_config = AgentConfig(
        state_dim=4,
        action_dim=1,
        goal_indices=np.array([2, 3]),
        goal_values=np.array([0.0, 0.0]),
    )
    agent = GoalStateAgent(config=agent_config)
    agent.set_weights(weights[0], weights[1])

    env = gym.make("CartPole-v1")
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset(seed=200 + ep)  # Different seeds from training
        steps = 0

        for _ in range(500):
            action = agent.get_action(state, training=False)
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break

        episode_lengths.append(steps)

    env.close()
    return episode_lengths


def main():
    print("="*70)
    print("EARLY STOPPING WITH WEIGHT FREEZING")
    print("="*70)
    print()

    # Test different freeze thresholds
    thresholds = [400, 450, 480]

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing freeze_threshold={threshold}")
        print("="*60)

        result = train_with_early_stopping(
            freeze_threshold=threshold,
            window=5,
            max_episodes=200
        )

        if result["frozen"]:
            print(f"\nTraining frozen at episode {result['freeze_episode']}")
            print(f"Final avg (last 10 training eps): {result['final_avg']:.1f}")

            # Test frozen weights extensively
            print(f"\nTesting frozen weights on 50 new episodes...")
            test_lengths = test_frozen_weights(result["frozen_weights"], num_episodes=50)

            print(f"  Mean: {np.mean(test_lengths):.1f}")
            print(f"  Std:  {np.std(test_lengths):.1f}")
            print(f"  Min:  {min(test_lengths)}")
            print(f"  Max:  {max(test_lengths)}")
            print(f"  Perfect (500): {sum(1 for l in test_lengths if l == 500)}/50")

        else:
            print(f"Did not reach freeze threshold in 200 episodes")
            print(f"Final avg: {result['final_avg']:.1f}")

    # Compare with no freezing (continued learning)
    print(f"\n{'='*60}")
    print("COMPARISON: Training without freezing (200 episodes)")
    print("="*60)

    config = TrainingConfig(
        max_episodes=200,
        action_update_steps=3,
        prediction_update_steps=3,
        use_adaptive_lr=True,
        gradient_clip_norm=2.0,
        seed=42,
    )

    trainer = Trainer(config=config)
    metrics = trainer.train(verbose=False)

    print(f"Final avg (last 10): {np.mean(metrics.episode_lengths[-10:]):.1f}")
    print(f"Convergence: {trainer.converged}")

    # Test final weights
    final_weights = trainer.agent.get_weights()
    test_lengths = test_frozen_weights(final_weights, num_episodes=50)

    print(f"\nTesting final weights on 50 new episodes:")
    print(f"  Mean: {np.mean(test_lengths):.1f}")
    print(f"  Std:  {np.std(test_lengths):.1f}")
    print(f"  Perfect (500): {sum(1 for l in test_lengths if l == 500)}/50")


if __name__ == "__main__":
    main()
