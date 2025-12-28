"""Achievement-gated plasticity for goal state agents.

The key insight: the goal state architecture already contains signals
about when learning should slow down:
- Goal error: "Have I achieved the goal?"
- Prediction error: "Do I understand this situation?"
- State novelty: "Have I been here before?"

When all three signal confidence, reduce plasticity.
When any signals uncertainty, increase plasticity.
"""

import numpy as np
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.goal_state_agent import GoalStateAgent
from dataclasses import dataclass


@dataclass
class PlasticityMetrics:
    """Metrics tracked by plasticity controller."""
    plasticity: float
    achievement_ema: float
    is_surprised: bool
    is_novel: bool
    goal_error: float
    prediction_error: float


class AchievementGatedPlasticity:
    """Modulates learning rate based on EPISODE-LEVEL achievement and step-level novelty.

    Key insight: Per-step goal error is misleading - a failing 150-step episode
    still has many low-error steps before the pole falls. We need to measure
    actual episode success, not instantaneous goal proximity.

    Core principles:
    1. Episode success (long episodes) → gradually reduce plasticity
    2. Episode failure (short episodes) → restore plasticity
    3. Surprising predictions → boost plasticity (world changed)
    4. Novel states → boost plasticity (new territory)

    The plasticity only decreases after SUSTAINED episode-level success.
    """

    def __init__(
        self,
        # Episode achievement settings
        target_length: int = 500,  # Target episode length for "success"
        success_threshold: float = 0.9,  # Length ratio above this = success
        ema_alpha: float = 0.1,  # How fast episode success EMA adapts (slower than before)

        # For backwards compatibility (ignored in new logic)
        goal_threshold: float = 0.05,

        # Plasticity bounds
        min_plasticity: float = 0.2,  # Higher minimum - never freeze too much
        max_plasticity: float = 1.0,

        # Surprise detection (prediction module uncertainty)
        surprise_threshold: float = 0.5,  # Pred error above this = surprised
        surprise_boost: float = 0.3,  # How much surprise increases plasticity

        # Novelty detection (unfamiliar states)
        novelty_threshold: float = 2.0,  # Z-score above this = novel
        novelty_boost: float = 0.2,  # How much novelty increases plasticity

        # State statistics for novelty
        state_ema_alpha: float = 0.01,  # For running mean/std
        warmup_steps: int = 100,  # Steps before novelty detection activates

        # Episode tracking
        min_episodes_before_reduction: int = 5,  # Wait for this many episodes
    ):
        self.target_length = target_length
        self.success_threshold = success_threshold
        self.goal_threshold = goal_threshold
        self.ema_alpha = ema_alpha
        self.min_plasticity = min_plasticity
        self.max_plasticity = max_plasticity
        self.surprise_threshold = surprise_threshold
        self.surprise_boost = surprise_boost
        self.novelty_threshold = novelty_threshold
        self.novelty_boost = novelty_boost
        self.state_ema_alpha = state_ema_alpha
        self.warmup_steps = warmup_steps
        self.min_episodes_before_reduction = min_episodes_before_reduction

        # State tracking
        self.achievement_ema = 0.0  # Now represents episode success rate
        self.plasticity = 1.0
        self.steps = 0
        self.episodes = 0
        self.current_episode_length = 0

        # Running statistics for state normalization
        self.state_mean: Optional[np.ndarray] = None
        self.state_var: Optional[np.ndarray] = None

        # History for debugging
        self.history: List[PlasticityMetrics] = []
        self.max_history = 1000

    def reset(self):
        """Reset for new episode (keeps learned statistics)."""
        # Don't reset achievement_ema - it should persist across episodes
        # Don't reset state statistics - they should accumulate
        self.current_episode_length = 0

    def hard_reset(self):
        """Full reset (for new training run)."""
        self.achievement_ema = 0.0
        self.plasticity = 1.0
        self.steps = 0
        self.episodes = 0
        self.current_episode_length = 0
        self.state_mean = None
        self.state_var = None
        self.history = []

    def end_episode(self, episode_length: int) -> float:
        """Update plasticity based on episode outcome.

        This is the PRIMARY driver of plasticity reduction.
        Call at the end of each episode.

        Args:
            episode_length: How long the episode lasted.

        Returns:
            Updated plasticity value.
        """
        self.episodes += 1
        self.current_episode_length = 0

        # Compute success ratio
        success_ratio = episode_length / self.target_length
        is_success = success_ratio >= self.success_threshold

        # Update achievement EMA based on episode success
        self.achievement_ema = (
            (1 - self.ema_alpha) * self.achievement_ema +
            self.ema_alpha * float(is_success)
        )

        # Only reduce plasticity after minimum episodes
        if self.episodes >= self.min_episodes_before_reduction:
            # Base plasticity from achievement (inverse relationship)
            # achievement_ema near 1.0 → plasticity approaches min
            self.plasticity = (
                self.min_plasticity +
                (1.0 - self.achievement_ema) * (self.max_plasticity - self.min_plasticity)
            )
        else:
            self.plasticity = self.max_plasticity

        return self.plasticity

    def update(
        self,
        state: np.ndarray,
        goal_error: float,
        prediction_error: float,
    ) -> float:
        """Update plasticity based on step-level signals (novelty, surprise).

        This provides TEMPORARY boosts to plasticity, not reductions.
        The base plasticity is set by end_episode().

        Args:
            state: Current state observation.
            goal_error: Current goal state error (MSE from goal).
            prediction_error: Current prediction module error.

        Returns:
            Current plasticity value [min_plasticity, max_plasticity].
        """
        state = np.atleast_1d(state).flatten()
        self.steps += 1
        self.current_episode_length += 1

        # 1. Update state statistics
        self._update_state_stats(state)

        # 2. Check for surprise (prediction module wrong)
        is_surprised = prediction_error > self.surprise_threshold

        # 3. Check for novelty (unfamiliar state)
        is_novel = self._is_novel(state)

        # 4. Start with base plasticity (set by end_episode)
        step_plasticity = self.plasticity

        # 5. Add boosts for surprise and novelty (temporary increases)
        if is_surprised:
            step_plasticity += self.surprise_boost
        if is_novel:
            step_plasticity += self.novelty_boost

        # 6. Clamp to bounds
        step_plasticity = np.clip(step_plasticity, self.min_plasticity, self.max_plasticity)

        # 7. Record metrics
        metrics = PlasticityMetrics(
            plasticity=step_plasticity,
            achievement_ema=self.achievement_ema,
            is_surprised=is_surprised,
            is_novel=is_novel,
            goal_error=goal_error,
            prediction_error=prediction_error,
        )
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return step_plasticity

    def _update_state_stats(self, state: np.ndarray):
        """Update running mean and variance of states."""
        if self.state_mean is None:
            self.state_mean = state.copy()
            self.state_var = np.ones_like(state)
        else:
            # Welford's online algorithm for running mean/variance
            delta = state - self.state_mean
            self.state_mean = self.state_mean + self.state_ema_alpha * delta
            self.state_var = (
                (1 - self.state_ema_alpha) * self.state_var +
                self.state_ema_alpha * delta * (state - self.state_mean)
            )
            # Ensure variance doesn't go to zero
            self.state_var = np.maximum(self.state_var, 1e-6)

    def _is_novel(self, state: np.ndarray) -> bool:
        """Check if state is novel (far from running statistics)."""
        if self.steps < self.warmup_steps:
            return False  # Don't detect novelty during warmup

        if self.state_mean is None:
            return True

        # Compute z-score for each dimension
        z_scores = np.abs(state - self.state_mean) / np.sqrt(self.state_var + 1e-6)

        # Novel if any dimension is far from mean
        return np.max(z_scores) > self.novelty_threshold

    def get_effective_lr(self, base_lr: float) -> float:
        """Get learning rate scaled by current plasticity."""
        return base_lr * self.plasticity

    def get_stats(self) -> dict:
        """Get current statistics for debugging."""
        return {
            "plasticity": self.plasticity,
            "achievement_ema": self.achievement_ema,
            "steps": self.steps,
            "state_mean": self.state_mean.tolist() if self.state_mean is not None else None,
            "state_std": np.sqrt(self.state_var).tolist() if self.state_var is not None else None,
        }


class BestCheckpoint:
    """Keeps track of the best-performing weights during training.

    Unlike plasticity gating which tries to prevent forgetting by reducing
    learning, this approach simply saves checkpoints and keeps the best one.
    Much more robust in practice.
    """

    def __init__(
        self,
        check_interval: int = 10,  # Episodes between checkpoints
        test_episodes: int = 5,  # Episodes to test each checkpoint
        test_seed_offset: int = 1000,  # Seed offset for test episodes
    ):
        self.check_interval = check_interval
        self.test_episodes = test_episodes
        self.test_seed_offset = test_seed_offset

        self.best_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.best_score: float = 0.0
        self.checkpoint_history: List[Tuple[int, float]] = []  # (episode, score)
        self.episode_count: int = 0

    def reset(self):
        """Reset for new training run."""
        self.best_weights = None
        self.best_score = 0.0
        self.checkpoint_history = []
        self.episode_count = 0

    def maybe_checkpoint(
        self,
        agent,
        test_env,
    ) -> Optional[float]:
        """Check if it's time to evaluate and possibly save checkpoint.

        Call this at the end of each episode.

        Args:
            agent: The agent to checkpoint.
            test_env: A gymnasium environment for testing.

        Returns:
            Test score if checkpoint was taken, None otherwise.
        """
        self.episode_count += 1

        if self.episode_count % self.check_interval != 0:
            return None

        # Evaluate current weights
        current_weights = agent.get_weights()
        test_lengths = []

        for t in range(self.test_episodes):
            state, _ = test_env.reset(seed=self.test_seed_offset + t)
            steps = 0
            for _ in range(500):
                action = agent.get_action(state, training=False)
                state, _, term, trunc, _ = test_env.step(action)
                steps += 1
                if term or trunc:
                    break
            test_lengths.append(steps)

        avg_score = float(np.mean(test_lengths))
        self.checkpoint_history.append((self.episode_count, avg_score))

        # Save if best so far
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_weights = (current_weights[0].copy(), current_weights[1].copy())

        return avg_score

    def get_best_weights(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the best checkpoint weights."""
        return self.best_weights

    def get_stats(self) -> dict:
        """Get checkpoint statistics."""
        return {
            "best_score": self.best_score,
            "episode_count": self.episode_count,
            "checkpoints_taken": len(self.checkpoint_history),
            "checkpoint_history": self.checkpoint_history,
        }


class PerformanceGatedPlasticity:
    """Simpler version: gate plasticity based on recent episode performance.

    This is less sophisticated but more robust - it looks at actual
    episode outcomes rather than per-step goal errors.
    """

    def __init__(
        self,
        target_length: int = 500,  # Target episode length
        window_size: int = 10,  # Episodes to consider
        min_plasticity: float = 0.1,
        threshold_ratio: float = 0.9,  # Reduce plasticity when avg > this * target
    ):
        self.target_length = target_length
        self.window_size = window_size
        self.min_plasticity = min_plasticity
        self.threshold_ratio = threshold_ratio

        self.episode_lengths: List[int] = []
        self.plasticity = 1.0

    def end_episode(self, episode_length: int) -> float:
        """Update plasticity at end of episode.

        Args:
            episode_length: Length of completed episode.

        Returns:
            Updated plasticity for next episode.
        """
        self.episode_lengths.append(episode_length)
        if len(self.episode_lengths) > self.window_size:
            self.episode_lengths.pop(0)

        if len(self.episode_lengths) >= self.window_size:
            avg_length = np.mean(self.episode_lengths)
            ratio = avg_length / self.target_length

            if ratio >= self.threshold_ratio:
                # Performing well - reduce plasticity
                # Plasticity decreases as ratio increases above threshold
                excess = (ratio - self.threshold_ratio) / (1.0 - self.threshold_ratio)
                self.plasticity = max(
                    self.min_plasticity,
                    1.0 - excess * (1.0 - self.min_plasticity)
                )
            else:
                # Not performing well enough - full plasticity
                self.plasticity = 1.0

        return self.plasticity

    def get_effective_lr(self, base_lr: float) -> float:
        """Get learning rate scaled by current plasticity."""
        return base_lr * self.plasticity

    def reset(self):
        """Reset for new training run."""
        self.episode_lengths = []
        self.plasticity = 1.0
