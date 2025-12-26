#!/usr/bin/env python3
"""Compare different learning rate strategies for the goal state agent.

This script runs multiple training sessions with different learning rate
configurations and compares their performance on CartPole-v1.

Strategies tested:
1. Baseline (no scheduling)
2. Exponential decay
3. Goldilocks (moderate surprise = more learning)
4. Inverse surprise (high surprise = more learning)
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from goal_state_agent import (
    Trainer, TrainingConfig,
    CARTPOLE_CLIPPED,
    CARTPOLE_LR_DECAY,
    CARTPOLE_GOLDILOCKS,
    CARTPOLE_SURPRISE,
)


def run_comparison(num_runs: int = 3, max_episodes: int = 200):
    """Run comparison of learning rate strategies.

    Args:
        num_runs: Number of runs per configuration.
        max_episodes: Max episodes per run.
    """
    configs = {
        "baseline": TrainingConfig(
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            max_episodes=max_episodes,
            seed=None,  # Different seed each run
        ),
        "lr_decay": TrainingConfig(
            lr_scheduler_type="decay",
            lr_decay_rate=0.9995,
            lr_decay_steps=100,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            max_episodes=max_episodes,
        ),
        "goldilocks": TrainingConfig(
            lr_scheduler_type="goldilocks",
            lr_target_surprise=0.05,
            lr_surprise_width=0.05,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            max_episodes=max_episodes,
        ),
        "surprise": TrainingConfig(
            lr_scheduler_type="surprise",
            lr_surprise_scale=1.5,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            max_episodes=max_episodes,
        ),
    }

    results = {name: [] for name in configs}

    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        for run in range(num_runs):
            print(f"\n  Run {run + 1}/{num_runs}...")

            # Set different seed for each run
            config.seed = 42 + run

            trainer = Trainer(config=config)
            metrics = trainer.train(verbose=False)

            # Get final performance (average of last 10 episodes)
            final_avg = np.mean(metrics.episode_lengths[-10:])
            converged = trainer.converged
            conv_ep = trainer.convergence_episode

            results[name].append({
                "final_avg": final_avg,
                "converged": converged,
                "convergence_episode": conv_ep,
                "episode_lengths": metrics.episode_lengths,
            })

            status = f"converged at ep {conv_ep}" if converged else "did not converge"
            print(f"    Final avg: {final_avg:.1f} ({status})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    for name in configs:
        runs = results[name]
        final_avgs = [r["final_avg"] for r in runs]
        conv_rate = sum(1 for r in runs if r["converged"]) / len(runs)

        print(f"{name}:")
        print(f"  Final avg steps: {np.mean(final_avgs):.1f} +/- {np.std(final_avgs):.1f}")
        print(f"  Convergence rate: {conv_rate*100:.0f}%")
        if any(r["converged"] for r in runs):
            conv_eps = [r["convergence_episode"] for r in runs if r["converged"]]
            print(f"  Avg convergence episode: {np.mean(conv_eps):.1f}")
        print()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare LR strategies")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per config")
    parser.add_argument("--episodes", type=int, default=200, help="Max episodes per run")
    args = parser.parse_args()

    results = run_comparison(num_runs=args.runs, max_episodes=args.episodes)
