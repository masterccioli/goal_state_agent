#!/usr/bin/env python3
"""Test the optimal configuration discovered through validation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from goal_state_agent import Trainer, TrainingConfig


def main():
    print('Testing OPTIMAL configuration (updates=3)')
    print('='*60)

    results = []
    for run in range(5):
        config = TrainingConfig(
            max_episodes=150,
            action_update_steps=3,
            prediction_update_steps=3,
            action_learning_rate=0.5,
            prediction_learning_rate=0.5,
            use_adaptive_lr=True,
            gradient_clip_norm=2.0,
            use_replay_buffer=False,
            seed=42 + run,
        )

        trainer = Trainer(config=config)
        metrics = trainer.train(verbose=False)

        final_avg = np.mean(metrics.episode_lengths[-10:])
        max_ep = max(metrics.episode_lengths)
        converged = trainer.converged
        conv_ep = trainer.convergence_episode

        results.append({
            'final_avg': final_avg,
            'max_ep': max_ep,
            'converged': converged,
            'conv_ep': conv_ep,
        })

        status = f'CONVERGED at ep {conv_ep}' if converged else 'not converged'
        print(f'Run {run+1}: final_avg={final_avg:.1f}, max={max_ep}, {status}')

    print()
    print('SUMMARY:')
    print(f'  Final avg: {np.mean([r["final_avg"] for r in results]):.1f} ± {np.std([r["final_avg"] for r in results]):.1f}')
    print(f'  Max episode: {np.mean([r["max_ep"] for r in results]):.0f}')
    print(f'  Convergence rate: {sum(1 for r in results if r["converged"])/len(results)*100:.0f}%')
    if any(r['converged'] for r in results):
        conv_eps = [r['conv_ep'] for r in results if r['converged']]
        print(f'  Avg convergence episode: {np.mean(conv_eps):.0f}')


if __name__ == "__main__":
    main()
