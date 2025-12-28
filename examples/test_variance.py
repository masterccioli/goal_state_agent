#!/usr/bin/env python3
"""Test variance across multiple seeds."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import gymnasium as gym
from goal_state_agent import Trainer, TrainingConfig
from goal_state_agent.agents.goal_state_agent import GoalStateAgent, AgentConfig


def test_weights(weights, num_episodes=20):
    agent_config = AgentConfig(
        state_dim=4, action_dim=1,
        goal_indices=np.array([2, 3]),
        goal_values=np.array([0.0, 0.0]),
    )
    agent = GoalStateAgent(config=agent_config)
    agent.set_weights(weights[0], weights[1])

    env = gym.make('CartPole-v1')
    lengths = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=500 + ep)
        steps = 0
        for _ in range(500):
            action = agent.get_action(state, training=False)
            state, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                break
        lengths.append(steps)
    env.close()
    return lengths


def main():
    print('Testing multiple seeds to understand variance...')
    print('='*70)

    all_results = []
    for seed in [42, 123, 456, 789, 1000]:
        config = TrainingConfig(
            max_episodes=100,
            action_update_steps=3,
            prediction_update_steps=3,
            use_adaptive_lr=True,
            gradient_clip_norm=2.0,
            seed=seed,
        )
        trainer = Trainer(config=config)
        metrics = trainer.train(verbose=False)

        # Test final weights
        final_weights = trainer.agent.get_weights()
        test_lengths = test_weights(final_weights, 20)

        perfect = sum(1 for l in test_lengths if l == 500)
        all_results.append({
            'seed': seed,
            'final_train_avg': np.mean(metrics.episode_lengths[-10:]),
            'test_mean': np.mean(test_lengths),
            'test_perfect': perfect,
            'converged': trainer.converged,
        })

        print(f'Seed {seed}: train_avg={all_results[-1]["final_train_avg"]:.1f}, '
              f'test_avg={all_results[-1]["test_mean"]:.1f}, '
              f'perfect={perfect}/20, conv={trainer.converged}')

    print()
    print('SUMMARY:')
    print(f'  Training avg:  {np.mean([r["final_train_avg"] for r in all_results]):.1f} '
          f'± {np.std([r["final_train_avg"] for r in all_results]):.1f}')
    print(f'  Test avg:      {np.mean([r["test_mean"] for r in all_results]):.1f} '
          f'± {np.std([r["test_mean"] for r in all_results]):.1f}')
    print(f'  Perfect rate:  {np.mean([r["test_perfect"] for r in all_results]):.1f}/20')
    print(f'  Convergence:   {sum(1 for r in all_results if r["converged"])}/5')


if __name__ == "__main__":
    main()
