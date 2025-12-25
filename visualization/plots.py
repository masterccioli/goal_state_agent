"""Training visualization plots."""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


def plot_episode_lengths(
    episode_lengths: List[int],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (8, 4),
    title: str = "Episode Length Over Training",
) -> Optional[plt.Figure]:
    """Plot episode lengths over training.

    Args:
        episode_lengths: List of steps per episode.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the figure.
        figsize: Figure size as (width, height).
        title: Plot title.

    Returns:
        The matplotlib Figure object, or None if matplotlib is unavailable.
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(episode_lengths, linewidth=1.5, color="#2E86AB")

    # Add rolling average
    if len(episode_lengths) > 10:
        window = min(10, len(episode_lengths) // 5)
        rolling_avg = np.convolve(
            episode_lengths,
            np.ones(window) / window,
            mode='valid'
        )
        offset = window // 2
        ax.plot(
            range(offset, offset + len(rolling_avg)),
            rolling_avg,
            linewidth=2,
            color="#E94F37",
            label=f"{window}-episode moving average",
            alpha=0.8,
        )
        ax.legend()

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_errors(
    goal_errors: List[float],
    prediction_errors: List[float],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (10, 4),
) -> Optional[plt.Figure]:
    """Plot goal state and prediction errors over training.

    Args:
        goal_errors: Average goal error per episode.
        prediction_errors: Average prediction error per episode.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the figure.
        figsize: Figure size as (width, height).

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Goal state error
    ax1 = axes[0]
    ax1.plot(goal_errors, linewidth=1.5, color="#2E86AB")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("MSE")
    ax1.set_title("Goal State Error")
    ax1.grid(True, alpha=0.3)

    # Prediction error
    ax2 = axes[1]
    # Filter out zeros (no prediction on first step)
    pred_errors_filtered = [e for e in prediction_errors if e > 0]
    ax2.plot(pred_errors_filtered, linewidth=1.5, color="#E94F37")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("MSE")
    ax2.set_title("Prediction Module Error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_training_metrics(
    episode_lengths: List[int],
    goal_errors: List[float],
    prediction_errors: List[float],
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (12, 8),
) -> Optional[plt.Figure]:
    """Create a comprehensive training metrics dashboard.

    Args:
        episode_lengths: List of steps per episode.
        goal_errors: Average goal error per episode.
        prediction_errors: Average prediction error per episode.
        save_dir: Directory to save figures. If None, figures are not saved.
        show: Whether to display the figure.
        figsize: Figure size as (width, height).

    Returns:
        The matplotlib Figure object.
    """
    _check_matplotlib()

    fig = plt.figure(figsize=figsize)

    # Create grid: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Episode lengths (top, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(episode_lengths, linewidth=1, color="#2E86AB", alpha=0.7, label="Episode length")

    if len(episode_lengths) > 10:
        window = min(10, len(episode_lengths) // 5)
        rolling_avg = np.convolve(
            episode_lengths,
            np.ones(window) / window,
            mode='valid'
        )
        offset = window // 2
        ax1.plot(
            range(offset, offset + len(rolling_avg)),
            rolling_avg,
            linewidth=2,
            color="#E94F37",
            label=f"{window}-ep moving avg",
        )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Steps")
    ax1.set_title("Episode Length Over Training")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Goal state error (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(goal_errors, linewidth=1.5, color="#2E86AB")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("MSE")
    ax2.set_title("Goal State Error")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log') if max(goal_errors) / (min(goal_errors) + 1e-10) > 100 else None

    # Prediction error (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    pred_errors_filtered = [e for e in prediction_errors if e > 0]
    if pred_errors_filtered:
        ax3.plot(pred_errors_filtered, linewidth=1.5, color="#E94F37")
        ax3.set_yscale('log') if max(pred_errors_filtered) / (min(pred_errors_filtered) + 1e-10) > 100 else None
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("MSE")
    ax3.set_title("Prediction Module Error")
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Goal State Agent Training Metrics", fontsize=14, fontweight="bold", y=1.02)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save combined figure
        plt.savefig(save_dir / "training_metrics.png", dpi=150, bbox_inches="tight")

        # Also save individual plots
        plot_episode_lengths(
            episode_lengths,
            save_path=save_dir / "episode_lengths.png",
            show=False,
        )
        plot_errors(
            goal_errors,
            prediction_errors,
            save_path=save_dir / "errors.png",
            show=False,
        )

    if show:
        plt.show()
    else:
        plt.close()

    return fig
