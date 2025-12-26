#!/usr/bin/env python3
"""Generate visualization plots comparing different LR strategies.

Creates side-by-side plots of training metrics for each learning rate
scheduling strategy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from goal_state_agent import Trainer, TrainingConfig
from goal_state_agent.visualization import plot_training_metrics


def run_and_visualize(name: str, config: TrainingConfig, output_dir: Path):
    """Run training and generate plots for a single configuration."""
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    trainer = Trainer(config=config)
    metrics = trainer.train(verbose=False)

    # Summary stats
    final_avg = np.mean(metrics.episode_lengths[-10:])
    converged = trainer.converged
    conv_ep = trainer.convergence_episode

    print(f"  Final avg: {final_avg:.1f}")
    print(f"  Converged: {converged}" + (f" at episode {conv_ep}" if converged else ""))

    # Generate plots
    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_training_metrics(
        episode_lengths=metrics.episode_lengths,
        goal_errors=metrics.goal_errors,
        prediction_errors=metrics.prediction_errors,
        save_dir=save_dir,
        show=False,
    )

    print(f"  Plots saved to: {save_dir}")

    return {
        "name": name,
        "episode_lengths": metrics.episode_lengths,
        "goal_errors": metrics.goal_errors,
        "prediction_errors": metrics.prediction_errors,
        "converged": converged,
        "convergence_episode": conv_ep,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize LR strategy comparison")
    parser.add_argument("--episodes", type=int, default=100, help="Max episodes")
    parser.add_argument("--output-dir", type=str, default="output/lr_comparison",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define configurations
    configs = {
        "baseline": TrainingConfig(
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            max_episodes=args.episodes,
            seed=args.seed,
        ),
        "lr_decay": TrainingConfig(
            lr_scheduler_type="decay",
            lr_decay_rate=0.9995,
            lr_decay_steps=100,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            max_episodes=args.episodes,
            seed=args.seed,
        ),
        "goldilocks": TrainingConfig(
            lr_scheduler_type="goldilocks",
            lr_target_surprise=0.05,
            lr_surprise_width=0.05,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            max_episodes=args.episodes,
            seed=args.seed,
        ),
        "surprise": TrainingConfig(
            lr_scheduler_type="surprise",
            lr_surprise_scale=1.5,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            max_episodes=args.episodes,
            seed=args.seed,
        ),
    }

    print("=" * 60)
    print("LR Strategy Comparison - Visualization")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")

    results = []
    for name, config in configs.items():
        result = run_and_visualize(name, config, output_dir)
        results.append(result)

    # Create combined comparison plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Learning Rate Strategy Comparison", fontsize=14, fontweight='bold')

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

        for idx, result in enumerate(results):
            name = result["name"]
            lengths = result["episode_lengths"]
            color = colors[idx]

            # Episode lengths
            axes[0, 0].plot(lengths, label=name, color=color, alpha=0.8)

            # Rolling average
            window = min(10, len(lengths))
            if len(lengths) >= window:
                rolling = np.convolve(lengths, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(rolling, label=name, color=color, linewidth=2)

            # Goal errors
            axes[1, 0].plot(result["goal_errors"], label=name, color=color, alpha=0.7)

            # Prediction errors
            axes[1, 1].plot(result["prediction_errors"], label=name, color=color, alpha=0.7)

        axes[0, 0].set_title("Episode Lengths")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Steps")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Rolling Average (10 episodes)")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Avg Steps")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Goal Errors")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("MSE")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Prediction Errors")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("MSE")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        comparison_path = output_dir / "comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n{'='*60}")
        print(f"Combined comparison plot saved to: {comparison_path}")
        print(f"{'='*60}")

    except ImportError:
        print("\nmatplotlib not available for combined plot")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
