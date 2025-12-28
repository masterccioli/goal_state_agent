#!/usr/bin/env python3
"""Validate research findings with correct gymnasium environment.

This script systematically tests all claims from research notes against
the standard gymnasium CartPole (not the broken custom version).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Verify we're using correct gymnasium
import gymnasium
print(f"Gymnasium version: {gymnasium.__version__}")
print(f"Gymnasium location: {gymnasium.__file__}")

from goal_state_agent import Trainer, TrainingConfig


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    name: str
    episode_lengths: List[int]
    goal_errors: List[float]
    prediction_errors: List[float]
    converged: bool
    convergence_episode: Optional[int]
    final_avg: float
    max_episode_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "converged": self.converged,
            "convergence_episode": self.convergence_episode,
            "final_avg": self.final_avg,
            "max_episode_length": self.max_episode_length,
            "total_episodes": len(self.episode_lengths),
        }


def run_experiment(name: str, config: TrainingConfig, verbose: bool = True) -> ExperimentResult:
    """Run a single experiment and return results."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

    trainer = Trainer(config=config)
    metrics = trainer.train(verbose=False)

    final_avg = np.mean(metrics.episode_lengths[-10:]) if len(metrics.episode_lengths) >= 10 else np.mean(metrics.episode_lengths)
    max_len = max(metrics.episode_lengths)

    result = ExperimentResult(
        name=name,
        episode_lengths=metrics.episode_lengths,
        goal_errors=metrics.goal_errors,
        prediction_errors=metrics.prediction_errors,
        converged=trainer.converged,
        convergence_episode=trainer.convergence_episode,
        final_avg=final_avg,
        max_episode_length=max_len,
    )

    if verbose:
        print(f"  Episodes: {len(metrics.episode_lengths)}")
        print(f"  Final avg (last 10): {final_avg:.1f}")
        print(f"  Max episode length: {max_len}")
        print(f"  Converged: {trainer.converged}", end="")
        if trainer.converged:
            print(f" at episode {trainer.convergence_episode}")
        else:
            print()

    return result


def validate_baseline(output_dir: Path, num_runs: int = 5, max_episodes: int = 100) -> Dict[str, List[ExperimentResult]]:
    """Validate baseline performance on CartPole-v1."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: BASELINE PERFORMANCE")
    print("="*70)
    print("Testing basic agent performance with default settings")

    results = []

    for run in range(num_runs):
        config = TrainingConfig(
            env_name="CartPole-v1",
            max_episodes=max_episodes,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
            seed=42 + run,
        )
        result = run_experiment(f"baseline_run{run+1}", config)
        results.append(result)

    # Summary
    final_avgs = [r.final_avg for r in results]
    max_lengths = [r.max_episode_length for r in results]
    conv_rate = sum(1 for r in results if r.converged) / len(results)

    print(f"\nBASELINE SUMMARY ({num_runs} runs):")
    print(f"  Final avg: {np.mean(final_avgs):.1f} ± {np.std(final_avgs):.1f}")
    print(f"  Max episode: {np.mean(max_lengths):.1f} ± {np.std(max_lengths):.1f}")
    print(f"  Convergence rate: {conv_rate*100:.0f}%")

    return {"baseline": results}


def validate_gradient_clipping(output_dir: Path, num_runs: int = 3, max_episodes: int = 100) -> Dict[str, List[ExperimentResult]]:
    """Validate gradient clipping effectiveness."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: GRADIENT CLIPPING EFFECTIVENESS")
    print("="*70)
    print("Claim: Gradient clipping provides ~15% improvement")

    clip_values = [None, 0.5, 1.0, 2.0, 5.0]
    all_results = {}

    for clip in clip_values:
        name = f"clip_{clip}" if clip else "no_clip"
        results = []

        for run in range(num_runs):
            config = TrainingConfig(
                env_name="CartPole-v1",
                max_episodes=max_episodes,
                use_adaptive_lr=True,
                gradient_clip_norm=clip,
                seed=42 + run,
            )
            result = run_experiment(f"{name}_run{run+1}", config)
            results.append(result)

        all_results[name] = results

    # Summary
    print(f"\nGRADIENT CLIPPING SUMMARY:")
    for name, results in all_results.items():
        final_avgs = [r.final_avg for r in results]
        conv_rate = sum(1 for r in results if r.converged) / len(results)
        print(f"  {name}: avg={np.mean(final_avgs):.1f}±{np.std(final_avgs):.1f}, conv={conv_rate*100:.0f}%")

    return all_results


def validate_lr_strategies(output_dir: Path, num_runs: int = 3, max_episodes: int = 100) -> Dict[str, List[ExperimentResult]]:
    """Validate learning rate strategy comparison."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: LEARNING RATE STRATEGIES")
    print("="*70)
    print("Claims: Goldilocks prevents instability, other strategies explode")

    strategies = {
        "baseline": TrainingConfig(
            env_name="CartPole-v1",
            max_episodes=max_episodes,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
        ),
        "lr_decay": TrainingConfig(
            env_name="CartPole-v1",
            max_episodes=max_episodes,
            lr_scheduler_type="decay",
            lr_decay_rate=0.9995,
            lr_decay_steps=100,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
        ),
        "goldilocks": TrainingConfig(
            env_name="CartPole-v1",
            max_episodes=max_episodes,
            lr_scheduler_type="goldilocks",
            lr_target_surprise=0.05,
            lr_surprise_width=0.05,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
        ),
        "surprise": TrainingConfig(
            env_name="CartPole-v1",
            max_episodes=max_episodes,
            lr_scheduler_type="surprise",
            lr_surprise_scale=1.5,
            use_adaptive_lr=True,
            gradient_clip_norm=1.0,
        ),
    }

    all_results = {}

    for name, base_config in strategies.items():
        results = []

        for run in range(num_runs):
            # Copy config and update seed
            from dataclasses import asdict
            config_dict = asdict(base_config)
            config_dict["seed"] = 42 + run
            config = TrainingConfig(**config_dict)
            result = run_experiment(f"{name}_run{run+1}", config)
            results.append(result)

        all_results[name] = results

    # Summary
    print(f"\nLR STRATEGIES SUMMARY:")
    for name, results in all_results.items():
        final_avgs = [r.final_avg for r in results]
        max_lengths = [r.max_episode_length for r in results]

        # Check error stability
        all_goal_errors = []
        for r in results:
            all_goal_errors.extend([e for e in r.goal_errors if np.isfinite(e)])

        max_error = max(all_goal_errors) if all_goal_errors else float('nan')

        print(f"  {name}: avg={np.mean(final_avgs):.1f}±{np.std(final_avgs):.1f}, "
              f"max_ep={np.mean(max_lengths):.0f}, max_err={max_error:.2e}")

    return all_results


def validate_learning_rates(output_dir: Path, num_runs: int = 3, max_episodes: int = 100) -> Dict[str, List[ExperimentResult]]:
    """Test different base learning rates."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: LEARNING RATE VALUES")
    print("="*70)
    print("Testing different base learning rates for the correct environment")

    lr_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    all_results = {}

    for lr in lr_values:
        name = f"lr_{lr}"
        results = []

        for run in range(num_runs):
            config = TrainingConfig(
                env_name="CartPole-v1",
                max_episodes=max_episodes,
                action_learning_rate=lr,
                prediction_learning_rate=lr,
                use_adaptive_lr=True,
                gradient_clip_norm=1.0,
                seed=42 + run,
            )
            result = run_experiment(f"{name}_run{run+1}", config)
            results.append(result)

        all_results[name] = results

    # Summary
    print(f"\nLEARNING RATE SUMMARY:")
    for name, results in all_results.items():
        final_avgs = [r.final_avg for r in results]
        max_lengths = [r.max_episode_length for r in results]
        conv_rate = sum(1 for r in results if r.converged) / len(results)
        print(f"  {name}: avg={np.mean(final_avgs):.1f}±{np.std(final_avgs):.1f}, "
              f"max={np.mean(max_lengths):.0f}, conv={conv_rate*100:.0f}%")

    return all_results


def validate_updates_per_step(output_dir: Path, num_runs: int = 3, max_episodes: int = 100) -> Dict[str, List[ExperimentResult]]:
    """Test different update frequencies."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: UPDATES PER STEP")
    print("="*70)
    print("Testing impact of update frequency on learning")

    update_values = [1, 3, 5, 10]
    all_results = {}

    for updates in update_values:
        name = f"updates_{updates}"
        results = []

        for run in range(num_runs):
            config = TrainingConfig(
                env_name="CartPole-v1",
                max_episodes=max_episodes,
                action_update_steps=updates,
                prediction_update_steps=updates,
                use_adaptive_lr=True,
                gradient_clip_norm=1.0,
                seed=42 + run,
            )
            result = run_experiment(f"{name}_run{run+1}", config)
            results.append(result)

        all_results[name] = results

    # Summary
    print(f"\nUPDATES PER STEP SUMMARY:")
    for name, results in all_results.items():
        final_avgs = [r.final_avg for r in results]
        max_lengths = [r.max_episode_length for r in results]
        print(f"  {name}: avg={np.mean(final_avgs):.1f}±{np.std(final_avgs):.1f}, max={np.mean(max_lengths):.0f}")

    return all_results


def generate_plots(all_experiments: Dict[str, Dict[str, List[ExperimentResult]]], output_dir: Path):
    """Generate comprehensive plots from all experiments."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Research Findings Validation (Correct Gymnasium)", fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot each experiment type
    exp_names = list(all_experiments.keys())

    for idx, (exp_name, exp_results) in enumerate(all_experiments.items()):
        if idx >= 6:
            break
        ax = axes.flat[idx]

        for i, (config_name, results) in enumerate(exp_results.items()):
            # Plot mean episode lengths
            all_lengths = []
            max_len = max(len(r.episode_lengths) for r in results)

            for r in results:
                padded = r.episode_lengths + [r.episode_lengths[-1]] * (max_len - len(r.episode_lengths))
                all_lengths.append(padded)

            mean_lengths = np.mean(all_lengths, axis=0)
            std_lengths = np.std(all_lengths, axis=0)

            ax.plot(mean_lengths, label=config_name, color=colors[i % 10], alpha=0.8)
            ax.fill_between(range(len(mean_lengths)),
                           mean_lengths - std_lengths,
                           mean_lengths + std_lengths,
                           color=colors[i % 10], alpha=0.2)

        ax.set_title(exp_name.replace("_", " ").title())
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "validation_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate research findings with correct gymnasium")
    parser.add_argument("--runs", type=int, default=3, help="Runs per config")
    parser.add_argument("--episodes", type=int, default=100, help="Max episodes per run")
    parser.add_argument("--output-dir", type=str, default="output/validation", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer runs")
    args = parser.parse_args()

    if args.quick:
        args.runs = 2
        args.episodes = 50

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("RESEARCH FINDINGS VALIDATION")
    print("="*70)
    print(f"Gymnasium version: {gymnasium.__version__}")
    print(f"Runs per config: {args.runs}")
    print(f"Episodes per run: {args.episodes}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    all_experiments = {}

    # Run all validation experiments
    all_experiments["baseline"] = validate_baseline(output_dir, args.runs, args.episodes)
    all_experiments["gradient_clipping"] = validate_gradient_clipping(output_dir, args.runs, args.episodes)
    all_experiments["lr_strategies"] = validate_lr_strategies(output_dir, args.runs, args.episodes)
    all_experiments["learning_rates"] = validate_learning_rates(output_dir, args.runs, args.episodes)
    all_experiments["updates_per_step"] = validate_updates_per_step(output_dir, args.runs, args.episodes)

    # Generate plots
    generate_plots(all_experiments, output_dir)

    # Save raw results
    summary = {}
    for exp_name, exp_results in all_experiments.items():
        summary[exp_name] = {}
        for config_name, results in exp_results.items():
            summary[exp_name][config_name] = {
                "runs": len(results),
                "final_avg_mean": float(np.mean([r.final_avg for r in results])),
                "final_avg_std": float(np.std([r.final_avg for r in results])),
                "max_episode_mean": float(np.mean([r.max_episode_length for r in results])),
                "convergence_rate": sum(1 for r in results if r.converged) / len(results),
            }

    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
