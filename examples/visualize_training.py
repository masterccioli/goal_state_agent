#!/usr/bin/env python3
"""Example script demonstrating training visualization and video recording.

This script trains a goal state agent and produces:
1. Training metric plots (episode lengths, errors)
2. Videos of the agent at different stages of learning

Usage:
    # Basic training with plots
    python -m goal_state_agent.examples.visualize_training

    # With video recording
    python -m goal_state_agent.examples.visualize_training --record-video

    # Specify output directory
    python -m goal_state_agent.examples.visualize_training --output-dir results/run1

    # See all options
    python -m goal_state_agent.examples.visualize_training --help
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and visualize goal state agent learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=100,
        help="Maximum training episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files (plots, videos)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record videos of agent at different training stages",
    )
    parser.add_argument(
        "--record-episodes",
        type=int,
        nargs="+",
        default=None,
        help="Specific episodes to record (default: auto-spaced)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (just save)",
    )
    parser.add_argument(
        "--create-montage",
        action="store_true",
        help="Create a montage video from recorded episodes (requires moviepy)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Goal State Agent - Training Visualization")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Record video: {args.record_video}")
    print("=" * 60)
    print()

    if args.record_video:
        # Use the video recording function which trains and records
        from goal_state_agent.visualization import record_learning_progression

        results = record_learning_progression(
            max_episodes=args.max_episodes,
            record_episodes=args.record_episodes,
            video_folder=str(output_dir / "videos"),
            verbose=True,
        )

        episode_lengths = results["episode_lengths"]
        goal_errors = results["goal_errors"]
        prediction_errors = results["prediction_errors"]

        print(f"\nRecorded {len(results['recorded_episodes'])} episodes")
        for rec in results["recorded_episodes"]:
            print(f"  Episode {rec['episode']}: {rec['steps']} steps -> {rec['path']}")

        # Create montage if requested
        if args.create_montage:
            from goal_state_agent.visualization.video import create_learning_montage

            print("\nCreating learning montage...")
            montage_path = create_learning_montage(
                video_folder=str(output_dir / "videos"),
                output_path=str(output_dir / "learning_montage.mp4"),
            )
            if montage_path:
                print(f"Montage saved to: {montage_path}")

    else:
        # Standard training without video
        from goal_state_agent import Trainer, TrainingConfig

        config = TrainingConfig(
            max_episodes=args.max_episodes,
            log_interval=10,
            save_interval=0,
        )

        trainer = Trainer(config=config)
        metrics = trainer.train(verbose=True)

        episode_lengths = metrics.episode_lengths
        goal_errors = metrics.goal_errors
        prediction_errors = metrics.prediction_errors

    # Generate plots
    if not args.no_plots:
        print("\nGenerating training plots...")

        from goal_state_agent.visualization import plot_training_metrics

        plot_training_metrics(
            episode_lengths=episode_lengths,
            goal_errors=goal_errors,
            prediction_errors=prediction_errors,
            save_dir=output_dir / "figures",
            show=not args.no_show,
        )

        print(f"Plots saved to: {output_dir / 'figures'}")

    print("\nDone!")
    print(f"All outputs saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
