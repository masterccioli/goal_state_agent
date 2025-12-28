#!/usr/bin/env python3
"""Validate replay buffer claims with correct gymnasium."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from goal_state_agent import Trainer, TrainingConfig


def run_experiments(name: str, config: TrainingConfig, num_runs: int = 3):
    """Run experiments and return summary."""
    results = []

    for run in range(num_runs):
        config.seed = 42 + run
        trainer = Trainer(config=config)
        metrics = trainer.train(verbose=False)

        final_avg = np.mean(metrics.episode_lengths[-10:])
        max_ep = max(metrics.episode_lengths)
        converged = trainer.converged

        results.append({
            "final_avg": final_avg,
            "max_ep": max_ep,
            "converged": converged,
            "conv_ep": trainer.convergence_episode,
        })

        print(f"  Run {run+1}: final_avg={final_avg:.1f}, max={max_ep}, conv={converged}")

    avg = np.mean([r["final_avg"] for r in results])
    std = np.std([r["final_avg"] for r in results])
    conv_rate = sum(1 for r in results if r["converged"]) / len(results)

    return {"name": name, "avg": avg, "std": std, "conv_rate": conv_rate, "results": results}


def main():
    print("="*70)
    print("REPLAY BUFFER VALIDATION")
    print("="*70)
    print("Claim: Replay buffers catastrophically harm this architecture")
    print()

    max_episodes = 100
    num_runs = 3

    all_results = []

    # 1. Baseline (online learning, no buffer)
    print("1. BASELINE (Online Learning)")
    print("-"*50)
    config = TrainingConfig(
        max_episodes=max_episodes,
        use_replay_buffer=False,
        use_adaptive_lr=True,
        gradient_clip_norm=1.0,
    )
    all_results.append(run_experiments("baseline", config, num_runs))
    print()

    # 2. Replay buffer for both modules
    print("2. REPLAY BUFFER (Both Modules)")
    print("-"*50)
    config = TrainingConfig(
        max_episodes=max_episodes,
        use_replay_buffer=True,
        use_local_targets=False,
        use_buffer_states=False,
        replay_buffer_capacity=10000,
        replay_batch_size=32,
        min_buffer_size=100,
        use_adaptive_lr=True,
        gradient_clip_norm=1.0,
    )
    all_results.append(run_experiments("replay_both", config, num_runs))
    print()

    # 3. Local targets (achieved next-states)
    print("3. LOCAL TARGETS (Achieved Next-States)")
    print("-"*50)
    config = TrainingConfig(
        max_episodes=max_episodes,
        use_replay_buffer=True,
        use_local_targets=True,
        local_target_percentile=25.0,
        replay_buffer_capacity=5000,
        replay_batch_size=16,
        min_buffer_size=50,
        use_adaptive_lr=True,
        gradient_clip_norm=1.0,
    )
    all_results.append(run_experiments("local_targets", config, num_runs))
    print()

    # 4. Buffer states + global goal
    print("4. BUFFER STATES + GLOBAL GOAL")
    print("-"*50)
    config = TrainingConfig(
        max_episodes=max_episodes,
        use_replay_buffer=True,
        use_local_targets=False,
        use_buffer_states=True,
        replay_buffer_capacity=5000,
        replay_batch_size=16,
        min_buffer_size=50,
        use_adaptive_lr=True,
        gradient_clip_norm=1.0,
    )
    all_results.append(run_experiments("buffer_states", config, num_runs))
    print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    for r in all_results:
        print(f"{r['name']:20s}: avg={r['avg']:.1f}±{r['std']:.1f}, conv={r['conv_rate']*100:.0f}%")

    print()
    print("CLAIM VALIDATION:")
    baseline = all_results[0]
    for r in all_results[1:]:
        diff = r['avg'] - baseline['avg']
        pct = (diff / baseline['avg']) * 100
        status = "CONFIRMED" if diff < -20 else "NOT CONFIRMED"
        print(f"  {r['name']}: {diff:+.1f} steps ({pct:+.1f}%) - {status}")


if __name__ == "__main__":
    main()
