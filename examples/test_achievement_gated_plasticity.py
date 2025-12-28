#!/usr/bin/env python3
"""Test achievement-gated plasticity.

The hypothesis: by reducing plasticity when the goal is consistently
achieved, we can prevent catastrophic forgetting while maintaining
the ability to learn from novel situations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

from goal_state_agent import TrainingConfig
from goal_state_agent.agents.goal_state_agent import GoalStateAgent, AgentConfig
from goal_state_agent.utils.plasticity import AchievementGatedPlasticity, PerformanceGatedPlasticity


@dataclass
class TrainingResult:
    """Results from a training run."""
    name: str
    episode_lengths: List[int]
    plasticity_history: List[float]
    achievement_history: List[float]
    frozen_test_lengths: List[int]
    final_weights: Tuple[np.ndarray, np.ndarray]


def create_agent() -> GoalStateAgent:
    """Create a fresh agent with optimal config."""
    config = AgentConfig(
        state_dim=4,
        action_dim=1,
        goal_indices=np.array([2, 3]),
        goal_values=np.array([0.0, 0.0]),
        action_learning_rate=0.5,
        prediction_learning_rate=0.5,
        action_update_steps=3,
        prediction_update_steps=3,
        use_adaptive_lr=True,
        gradient_clip_norm=2.0,
    )
    return GoalStateAgent(config=config)


def test_frozen_weights(weights: Tuple[np.ndarray, np.ndarray], num_episodes: int = 20) -> List[int]:
    """Test frozen weights on new episodes."""
    agent = create_agent()
    agent.set_weights(weights[0], weights[1])

    env = gym.make("CartPole-v1")
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


def train_baseline(max_episodes: int = 200, seed: int = 42) -> TrainingResult:
    """Train with constant plasticity (baseline)."""
    print(f"\nTraining BASELINE (constant plasticity)...")

    agent = create_agent()
    env = gym.make("CartPole-v1")

    episode_lengths = []
    plasticity_history = []
    achievement_history = []

    for ep in range(max_episodes):
        state, _ = env.reset(seed=seed + ep)
        agent.reset()
        steps = 0
        ep_goal_errors = []

        for _ in range(500):
            action, goal_error, pred_error = agent.step(state)
            ep_goal_errors.append(goal_error)
            state, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                break

        episode_lengths.append(steps)
        plasticity_history.append(1.0)  # Constant
        achievement_history.append(np.mean([1.0 if e < 0.05 else 0.0 for e in ep_goal_errors]))

        if ep % 50 == 0 or steps >= 450:
            print(f"  Episode {ep}: {steps} steps")

    env.close()

    # Test frozen weights
    final_weights = agent.get_weights()
    frozen_lengths = test_frozen_weights(final_weights)

    return TrainingResult(
        name="baseline",
        episode_lengths=episode_lengths,
        plasticity_history=plasticity_history,
        achievement_history=achievement_history,
        frozen_test_lengths=frozen_lengths,
        final_weights=final_weights,
    )


def train_with_achievement_gating(
    max_episodes: int = 200,
    seed: int = 42,
    goal_threshold: float = 0.05,
    ema_alpha: float = 0.02,
    min_plasticity: float = 0.1,
) -> TrainingResult:
    """Train with achievement-gated plasticity."""
    print(f"\nTraining with ACHIEVEMENT-GATED plasticity...")
    print(f"  goal_threshold={goal_threshold}, ema_alpha={ema_alpha}, min_plasticity={min_plasticity}")

    agent = create_agent()
    plasticity_ctrl = AchievementGatedPlasticity(
        goal_threshold=goal_threshold,
        ema_alpha=ema_alpha,
        min_plasticity=min_plasticity,
        surprise_threshold=0.5,
        novelty_threshold=2.0,
    )

    env = gym.make("CartPole-v1")

    episode_lengths = []
    plasticity_history = []
    achievement_history = []

    for ep in range(max_episodes):
        state, _ = env.reset(seed=seed + ep)
        agent.reset()
        plasticity_ctrl.reset()
        steps = 0
        ep_plasticities = []
        ep_goal_errors = []

        for _ in range(500):
            # Get action and errors from agent
            action, goal_error, pred_error = agent.step(state)
            ep_goal_errors.append(goal_error)

            # Update plasticity controller
            plasticity = plasticity_ctrl.update(state, goal_error, pred_error)
            ep_plasticities.append(plasticity)

            # Apply plasticity to learning rate (modify agent's effective LR)
            # We do this by scaling the weights update after the fact
            # This is a hack - ideally we'd integrate more deeply
            # For now, we'll use a different approach: scale the agent's LR directly

            state, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                break

        episode_lengths.append(steps)
        plasticity_history.append(np.mean(ep_plasticities))
        achievement_history.append(plasticity_ctrl.achievement_ema)

        if ep % 50 == 0 or steps >= 450:
            print(f"  Episode {ep}: {steps} steps, plasticity={np.mean(ep_plasticities):.3f}, "
                  f"achievement_ema={plasticity_ctrl.achievement_ema:.3f}")

    env.close()

    # Test frozen weights
    final_weights = agent.get_weights()
    frozen_lengths = test_frozen_weights(final_weights)

    return TrainingResult(
        name="achievement_gated",
        episode_lengths=episode_lengths,
        plasticity_history=plasticity_history,
        achievement_history=achievement_history,
        frozen_test_lengths=frozen_lengths,
        final_weights=final_weights,
    )


def train_with_integrated_plasticity(
    max_episodes: int = 200,
    seed: int = 42,
    min_plasticity: float = 0.2,
    success_threshold: float = 0.9,
    ema_alpha: float = 0.1,
) -> TrainingResult:
    """Train with episode-level plasticity gating and step-level boosts.

    Key insight: Plasticity reduction is based on EPISODE SUCCESS (long episodes),
    not per-step goal error. Per-step novelty/surprise still boost plasticity.
    """
    print(f"\nTraining with EPISODE-GATED plasticity (with novelty boosts)...")
    print(f"  min_plasticity={min_plasticity}, success_threshold={success_threshold}")

    agent = create_agent()
    base_action_lr = agent.config.action_learning_rate
    base_pred_lr = agent.config.prediction_learning_rate

    plasticity_ctrl = AchievementGatedPlasticity(
        target_length=500,
        success_threshold=success_threshold,
        ema_alpha=ema_alpha,
        min_plasticity=min_plasticity,
        surprise_threshold=0.5,
        surprise_boost=0.3,
        novelty_threshold=2.5,
        novelty_boost=0.2,
        min_episodes_before_reduction=5,
    )

    env = gym.make("CartPole-v1")

    episode_lengths = []
    plasticity_history = []
    achievement_history = []

    for ep in range(max_episodes):
        state, _ = env.reset(seed=seed + ep)
        agent.reset()
        plasticity_ctrl.reset()
        steps = 0
        ep_plasticities = []

        # Use episode-level base plasticity
        ep_base_plasticity = plasticity_ctrl.plasticity

        for step in range(500):
            # Scale learning rates BEFORE step (use episode-level base plasticity)
            agent.config.action_learning_rate = base_action_lr * ep_base_plasticity
            agent.config.prediction_learning_rate = base_pred_lr * ep_base_plasticity

            # Now do the learning step
            action, goal_error, pred_error = agent.step(state)

            # Track step-level plasticity (with novelty/surprise boosts) for metrics
            step_plasticity = plasticity_ctrl.update(state, goal_error, pred_error)
            ep_plasticities.append(step_plasticity)

            # Execute action
            next_state, _, term, trunc, _ = env.step(action)
            state = next_state
            steps += 1

            if term or trunc:
                break

        # Restore base learning rates
        agent.config.action_learning_rate = base_action_lr
        agent.config.prediction_learning_rate = base_pred_lr

        # Update episode-level plasticity (PRIMARY driver of reduction)
        plasticity_ctrl.end_episode(steps)

        episode_lengths.append(steps)
        plasticity_history.append(plasticity_ctrl.plasticity)  # Episode-level plasticity
        achievement_history.append(plasticity_ctrl.achievement_ema)

        if ep % 50 == 0 or steps >= 450:
            print(f"  Episode {ep}: {steps} steps, plasticity={plasticity_ctrl.plasticity:.3f}, "
                  f"achievement_ema={plasticity_ctrl.achievement_ema:.3f}")

    env.close()

    # Test frozen weights
    final_weights = agent.get_weights()
    frozen_lengths = test_frozen_weights(final_weights)

    return TrainingResult(
        name="episode_gated",
        episode_lengths=episode_lengths,
        plasticity_history=plasticity_history,
        achievement_history=achievement_history,
        frozen_test_lengths=frozen_lengths,
        final_weights=final_weights,
    )


def train_with_performance_gating(
    max_episodes: int = 200,
    seed: int = 42,
) -> TrainingResult:
    """Train with performance-gated plasticity (episode-level)."""
    print(f"\nTraining with PERFORMANCE-GATED plasticity...")

    agent = create_agent()
    base_action_lr = agent.config.action_learning_rate
    base_pred_lr = agent.config.prediction_learning_rate

    plasticity_ctrl = PerformanceGatedPlasticity(
        target_length=500,
        window_size=10,
        min_plasticity=0.1,
        threshold_ratio=0.8,
    )

    env = gym.make("CartPole-v1")

    episode_lengths = []
    plasticity_history = []
    achievement_history = []

    for ep in range(max_episodes):
        # Set learning rate based on previous episodes' performance
        plasticity = plasticity_ctrl.plasticity
        agent.config.action_learning_rate = base_action_lr * plasticity
        agent.config.prediction_learning_rate = base_pred_lr * plasticity

        state, _ = env.reset(seed=seed + ep)
        agent.reset()
        steps = 0

        for _ in range(500):
            action, goal_error, pred_error = agent.step(state)
            state, _, term, trunc, _ = env.step(action)
            steps += 1
            if term or trunc:
                break

        episode_lengths.append(steps)

        # Update plasticity based on this episode
        plasticity_ctrl.end_episode(steps)
        plasticity_history.append(plasticity_ctrl.plasticity)
        achievement_history.append(steps / 500.0)

        if ep % 50 == 0 or steps >= 450:
            print(f"  Episode {ep}: {steps} steps, plasticity={plasticity_ctrl.plasticity:.3f}")

    env.close()

    # Restore base learning rates
    agent.config.action_learning_rate = base_action_lr
    agent.config.prediction_learning_rate = base_pred_lr

    # Test frozen weights
    final_weights = agent.get_weights()
    frozen_lengths = test_frozen_weights(final_weights)

    return TrainingResult(
        name="performance_gated",
        episode_lengths=episode_lengths,
        plasticity_history=plasticity_history,
        achievement_history=achievement_history,
        frozen_test_lengths=frozen_lengths,
        final_weights=final_weights,
    )


def plot_results(results: List[TrainingResult], output_path: Path):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Achievement-Gated Plasticity Comparison", fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # 1. Episode lengths over training
    ax = axes[0, 0]
    for i, r in enumerate(results):
        # Smooth with rolling average
        window = 10
        smoothed = np.convolve(r.episode_lengths, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=r.name, color=colors[i], alpha=0.8)
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='max')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length (smoothed)")
    ax.set_title("Training Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Plasticity over training
    ax = axes[0, 1]
    for i, r in enumerate(results):
        ax.plot(r.plasticity_history, label=r.name, color=colors[i], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Plasticity")
    ax.set_title("Plasticity Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Frozen test performance (box plot)
    ax = axes[1, 0]
    data = [r.frozen_test_lengths for r in results]
    names = [r.name for r in results]
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("Episode Length")
    ax.set_title("Frozen Policy Test (20 episodes)")
    ax.grid(True, alpha=0.3)

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary = "RESULTS SUMMARY\n" + "="*50 + "\n\n"
    for r in results:
        train_avg = np.mean(r.episode_lengths[-20:])
        test_avg = np.mean(r.frozen_test_lengths)
        test_perfect = sum(1 for l in r.frozen_test_lengths if l == 500)
        final_plasticity = r.plasticity_history[-1] if r.plasticity_history else 1.0

        summary += f"{r.name}:\n"
        summary += f"  Training avg (last 20): {train_avg:.1f}\n"
        summary += f"  Frozen test avg: {test_avg:.1f}\n"
        summary += f"  Perfect (500): {test_perfect}/20\n"
        summary += f"  Final plasticity: {final_plasticity:.3f}\n\n"

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


def main():
    print("="*70)
    print("ACHIEVEMENT-GATED PLASTICITY EXPERIMENT")
    print("="*70)

    output_dir = Path("output/plasticity_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    max_episodes = 150

    results = []

    # 1. Baseline (constant plasticity)
    results.append(train_baseline(max_episodes, seed))

    # 2. Performance-gated (episode-level)
    results.append(train_with_performance_gating(max_episodes, seed))

    # 3. Episode-gated (with novelty/surprise boosts)
    results.append(train_with_integrated_plasticity(
        max_episodes, seed,
        min_plasticity=0.2,
        success_threshold=0.9,
        ema_alpha=0.1,
    ))

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for r in results:
        train_avg = np.mean(r.episode_lengths[-20:])
        test_avg = np.mean(r.frozen_test_lengths)
        test_std = np.std(r.frozen_test_lengths)
        test_perfect = sum(1 for l in r.frozen_test_lengths if l == 500)

        print(f"\n{r.name}:")
        print(f"  Training avg (last 20): {train_avg:.1f}")
        print(f"  Frozen test: {test_avg:.1f} ± {test_std:.1f}")
        print(f"  Perfect (500): {test_perfect}/20")

    # Create plots
    plot_results(results, output_dir / "plasticity_comparison.png")


if __name__ == "__main__":
    main()
