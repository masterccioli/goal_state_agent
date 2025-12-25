"""Training loop and utilities for goal state agent."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from .agents.goal_state_agent import GoalStateAgent, AgentConfig
from .configs.config import TrainingConfig
from .utils.optimizers import SGD, Adam


@dataclass
class TrainingMetrics:
    """Container for training metrics.

    Attributes:
        episode_lengths: Steps per episode.
        goal_errors: Average goal error per episode.
        prediction_errors: Average prediction error per episode.
        episode_times: Wall-clock time per episode.
    """
    episode_lengths: List[int] = field(default_factory=list)
    goal_errors: List[float] = field(default_factory=list)
    prediction_errors: List[float] = field(default_factory=list)
    episode_times: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "episode_lengths": self.episode_lengths,
            "goal_errors": self.goal_errors,
            "prediction_errors": self.prediction_errors,
            "episode_times": self.episode_times,
        }

    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class Trainer:
    """Training loop for goal state agent.

    Handles the full training process including:
    - Environment interaction
    - Agent updates
    - Metrics tracking
    - Checkpointing
    - Convergence detection
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        agent: Optional[GoalStateAgent] = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration. Uses defaults if None.
            agent: Pre-configured agent. Creates new one if None.
        """
        self.config = config or TrainingConfig()

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # Create environment
        self.env = gym.make(self.config.env_name)

        # Create agent
        if agent is not None:
            self.agent = agent
        else:
            self.agent = self._create_agent()

        # Metrics
        self.metrics = TrainingMetrics()

        # Training state
        self.current_episode = 0
        self.converged = False
        self.convergence_episode = None

    def _create_agent(self) -> GoalStateAgent:
        """Create agent from config."""
        agent_config = AgentConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            goal_indices=self.config.goal_indices,
            goal_values=np.array(self.config.goal_values),
            action_learning_rate=self.config.action_learning_rate,
            prediction_learning_rate=self.config.prediction_learning_rate,
            action_update_steps=self.config.action_update_steps,
            prediction_update_steps=self.config.prediction_update_steps,
            use_adaptive_lr=self.config.use_adaptive_lr,
            action_threshold=self.config.action_threshold,
            # Replay buffer settings
            use_replay_buffer=self.config.use_replay_buffer,
            replay_buffer_capacity=self.config.replay_buffer_capacity,
            replay_batch_size=self.config.replay_batch_size,
            min_buffer_size=self.config.min_buffer_size,
            updates_per_step=self.config.updates_per_step,
            # Gradient clipping
            gradient_clip_norm=self.config.gradient_clip_norm,
        )

        # Create optimizers if specified
        action_optimizer = None
        prediction_optimizer = None

        if self.config.optimizer_type == "adam":
            action_optimizer = Adam(
                learning_rate=self.config.action_learning_rate,
                beta1=self.config.adam_beta1,
                beta2=self.config.adam_beta2,
                epsilon=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
            prediction_optimizer = Adam(
                learning_rate=self.config.prediction_learning_rate,
                beta1=self.config.adam_beta1,
                beta2=self.config.adam_beta2,
                epsilon=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "sgd":
            action_optimizer = SGD(
                learning_rate=self.config.action_learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
            prediction_optimizer = SGD(
                learning_rate=self.config.prediction_learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        return GoalStateAgent(
            config=agent_config,
            action_optimizer=action_optimizer,
            prediction_optimizer=prediction_optimizer,
        )

    def _check_convergence(self) -> bool:
        """Check if training has converged.

        Convergence is defined as achieving the threshold cumulative
        steps over the convergence window.
        """
        if len(self.metrics.episode_lengths) < self.config.convergence_window:
            return False

        recent = self.metrics.episode_lengths[-self.config.convergence_window:]
        return sum(recent) >= self.config.convergence_threshold

    def run_episode(self) -> Tuple[int, float, float]:
        """Run a single training episode.

        Returns:
            Tuple of (steps, avg_goal_error, avg_prediction_error).
        """
        obs, info = self.env.reset()
        self.agent.reset()

        episode_goal_errors = []
        episode_pred_errors = []
        step_count = 0
        done = False

        while not done and step_count < self.config.max_steps_per_episode:
            # Agent step: update networks and get action
            action, goal_error, pred_error = self.agent.step(obs)

            # Store errors
            episode_goal_errors.append(goal_error)
            if pred_error > 0:
                episode_pred_errors.append(pred_error)

            # Environment step
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step_count += 1

        avg_goal_error = np.mean(episode_goal_errors) if episode_goal_errors else 0
        avg_pred_error = np.mean(episode_pred_errors) if episode_pred_errors else 0

        return step_count, avg_goal_error, avg_pred_error

    def train(self, verbose: bool = True) -> TrainingMetrics:
        """Run full training loop.

        Args:
            verbose: Whether to print progress.

        Returns:
            Training metrics.
        """
        start_time = time.time()

        for episode in range(self.config.max_episodes):
            self.current_episode = episode
            episode_start = time.time()

            # Run episode
            steps, goal_error, pred_error = self.run_episode()

            # Record metrics
            episode_time = time.time() - episode_start
            self.metrics.episode_lengths.append(steps)
            self.metrics.goal_errors.append(goal_error)
            self.metrics.prediction_errors.append(pred_error)
            self.metrics.episode_times.append(episode_time)

            # Check convergence
            if self._check_convergence():
                self.converged = True
                self.convergence_episode = episode
                if verbose:
                    print(f"Converged at episode {episode}")
                break

            # Logging
            if verbose and episode % self.config.log_interval == 0:
                avg_steps = np.mean(self.metrics.episode_lengths[-10:])
                print(
                    f"Episode {episode}: "
                    f"steps={steps}, "
                    f"avg_steps={avg_steps:.1f}, "
                    f"goal_err={goal_error:.4f}, "
                    f"pred_err={pred_error:.4f}"
                )

            # Checkpointing
            if self.config.save_interval > 0 and episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)

        total_time = time.time() - start_time

        if verbose:
            print(f"\nTraining complete in {total_time:.1f}s")
            print(f"Episodes: {self.current_episode + 1}")
            print(f"Converged: {self.converged}")
            if self.converged:
                print(f"Convergence episode: {self.convergence_episode}")

        return self.metrics

    def save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint.

        Args:
            episode: Current episode number.
        """
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        action_W, pred_W = self.agent.get_weights()
        np.save(save_dir / f"action_weights_{episode}.npy", action_W)
        np.save(save_dir / f"prediction_weights_{episode}.npy", pred_W)

        # Save metrics
        self.metrics.save(str(save_dir / f"metrics_{episode}.json"))

    def load_checkpoint(self, episode: int) -> None:
        """Load training checkpoint.

        Args:
            episode: Episode number to load.
        """
        save_dir = Path(self.config.save_dir)

        action_W = np.load(save_dir / f"action_weights_{episode}.npy")
        pred_W = np.load(save_dir / f"prediction_weights_{episode}.npy")

        self.agent.set_weights(action_W, pred_W)

    def evaluate(
        self,
        num_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, float]:
        """Evaluate trained agent.

        Args:
            num_episodes: Number of evaluation episodes.
            render: Whether to render the environment.

        Returns:
            Dictionary of evaluation metrics.
        """
        if render:
            eval_env = gym.make(self.config.env_name, render_mode="human")
        else:
            eval_env = gym.make(self.config.env_name)

        episode_lengths = []

        for _ in range(num_episodes):
            obs, info = eval_env.reset()
            self.agent.reset()
            done = False
            steps = 0

            while not done and steps < self.config.max_steps_per_episode:
                # Get action without training updates
                action = self.agent.get_action(obs, training=False)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                steps += 1

            episode_lengths.append(steps)

        eval_env.close()

        return {
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "min_episode_length": np.min(episode_lengths),
            "max_episode_length": np.max(episode_lengths),
        }


def train_agent(
    config: Optional[TrainingConfig] = None,
    verbose: bool = True,
) -> Tuple[GoalStateAgent, TrainingMetrics]:
    """Convenience function to train an agent.

    Args:
        config: Training configuration.
        verbose: Whether to print progress.

    Returns:
        Tuple of (trained_agent, metrics).
    """
    trainer = Trainer(config=config)
    metrics = trainer.train(verbose=verbose)
    return trainer.agent, metrics
