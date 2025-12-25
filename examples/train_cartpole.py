#!/usr/bin/env python3
"""Example script demonstrating goal state agent training on CartPole.

This script replicates the results from the original notebooks with a clean,
configurable interface.

Usage:
    # Train with default settings
    python train_cartpole.py

    # Train with custom learning rate
    python train_cartpole.py --action-lr 0.1 --prediction-lr 0.1

    # Train with Adam optimizer
    python train_cartpole.py --optimizer adam --action-lr 0.001

    # Train with different goal state (upright only, ignoring velocity)
    python train_cartpole.py --goal-indices 2 --goal-values 0.0
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from goal_state_agent import (
    Trainer,
    TrainingConfig,
    CARTPOLE_DEFAULT,
    CARTPOLE_FAST,
    CARTPOLE_ADAM,
    save_config,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a goal state agent on CartPole",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training settings
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Maximum training episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )

    # Learning rates
    parser.add_argument(
        "--action-lr",
        type=float,
        default=0.5,
        help="Action policy learning rate",
    )
    parser.add_argument(
        "--prediction-lr",
        type=float,
        default=0.5,
        help="Prediction module learning rate",
    )

    # Update steps
    parser.add_argument(
        "--action-steps",
        type=int,
        default=5,
        help="Gradient steps per action update",
    )
    parser.add_argument(
        "--prediction-steps",
        type=int,
        default=5,
        help="Gradient steps per prediction update",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        choices=["builtin", "sgd", "adam"],
        default="builtin",
        help="Optimizer type",
    )
    parser.add_argument(
        "--no-adaptive-lr",
        action="store_true",
        help="Disable adaptive learning rate scaling",
    )

    # Goal state
    parser.add_argument(
        "--goal-indices",
        type=int,
        nargs="+",
        default=[2, 3],
        help="State indices for goal (2=angle, 3=angular_velocity)",
    )
    parser.add_argument(
        "--goal-values",
        type=float,
        nargs="+",
        default=[0.0, 0.0],
        help="Target values for goal dimensions",
    )

    # Preset configs
    parser.add_argument(
        "--preset",
        choices=["default", "fast", "adam"],
        default=None,
        help="Use a preset configuration",
    )

    # Output
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes after training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Start with preset or default config
    if args.preset == "fast":
        config = CARTPOLE_FAST
    elif args.preset == "adam":
        config = CARTPOLE_ADAM
    else:
        config = CARTPOLE_DEFAULT

    # Override with command line arguments
    config = TrainingConfig(
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps,
        action_learning_rate=args.action_lr,
        prediction_learning_rate=args.prediction_lr,
        action_update_steps=args.action_steps,
        prediction_update_steps=args.prediction_steps,
        optimizer_type=args.optimizer,
        use_adaptive_lr=not args.no_adaptive_lr,
        goal_indices=args.goal_indices,
        goal_values=args.goal_values,
        save_dir=args.save_dir,
        seed=args.seed,
        log_interval=1 if not args.quiet else 10,
    )

    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print("Goal State Agent Training")
        print("=" * 60)
        print(f"Environment: {config.env_name}")
        print(f"Max episodes: {config.max_episodes}")
        print(f"Action learning rate: {config.action_learning_rate}")
        print(f"Prediction learning rate: {config.prediction_learning_rate}")
        print(f"Optimizer: {config.optimizer_type}")
        print(f"Adaptive LR: {config.use_adaptive_lr}")
        print(f"Goal indices: {config.goal_indices}")
        print(f"Goal values: {config.goal_values}")
        print("=" * 60)
        print()

    # Save config
    save_config(config, Path(config.save_dir) / "config.json")

    # Create trainer and train
    trainer = Trainer(config=config)
    metrics = trainer.train(verbose=not args.quiet)

    # Save final metrics
    metrics.save(str(Path(config.save_dir) / "final_metrics.json"))

    # Evaluate
    if not args.quiet:
        print("\n" + "=" * 60)
        print("Evaluation")
        print("=" * 60)

    eval_results = trainer.evaluate(num_episodes=args.eval_episodes)

    print(f"Mean episode length: {eval_results['mean_episode_length']:.1f}")
    print(f"Std episode length: {eval_results['std_episode_length']:.1f}")
    print(f"Min/Max: {eval_results['min_episode_length']}/{eval_results['max_episode_length']}")

    # Summary
    if trainer.converged:
        print(f"\nSuccess! Converged at episode {trainer.convergence_episode}")
    else:
        print(f"\nDid not converge within {config.max_episodes} episodes")

    return 0 if trainer.converged else 1


if __name__ == "__main__":
    sys.exit(main())
