"""Configuration system for training hyperparameters."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union
import json
import numpy as np


@dataclass
class TrainingConfig:
    """Complete training configuration.

    This dataclass contains all hyperparameters for training a goal state agent.
    It can be saved to and loaded from JSON for reproducibility.

    Attributes:
        # Environment settings
        env_name: Gymnasium environment name.
        max_episodes: Maximum number of training episodes.
        max_steps_per_episode: Maximum steps per episode.

        # Agent architecture
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        goal_indices: State indices that define the goal.
        goal_values: Target values for goal dimensions.

        # Learning rates
        action_learning_rate: Learning rate for action policy.
        prediction_learning_rate: Learning rate for prediction module.
        use_adaptive_lr: Scale learning rate by gradient magnitude.

        # Update steps (inner loop iterations)
        action_update_steps: Gradient steps per action update.
        prediction_update_steps: Gradient steps per prediction update.

        # Action settings
        action_threshold: Threshold for discrete action selection.

        # Optimizer settings
        optimizer_type: Type of optimizer ('sgd', 'adam', 'builtin').
        adam_beta1: Adam beta1 parameter.
        adam_beta2: Adam beta2 parameter.
        adam_epsilon: Adam epsilon parameter.
        momentum: SGD momentum parameter.
        weight_decay: L2 regularization coefficient.

        # Convergence settings
        convergence_window: Number of episodes to check for convergence.
        convergence_threshold: Required cumulative steps for convergence.

        # Logging
        log_interval: Episodes between progress logs.
        save_interval: Episodes between checkpoint saves.
        save_dir: Directory for saving checkpoints.

        # Random seed
        seed: Random seed for reproducibility.
    """
    # Environment settings
    env_name: str = "CartPole-v1"
    max_episodes: int = 1000
    max_steps_per_episode: int = 500

    # Agent architecture
    state_dim: int = 4
    action_dim: int = 1
    goal_indices: List[int] = field(default_factory=lambda: [2, 3])
    goal_values: List[float] = field(default_factory=lambda: [0.0, 0.0])

    # Learning rates
    action_learning_rate: float = 0.5
    prediction_learning_rate: float = 0.5
    use_adaptive_lr: bool = True

    # Update steps
    action_update_steps: int = 5
    prediction_update_steps: int = 5

    # Action settings
    action_threshold: float = 0.5

    # Optimizer settings
    optimizer_type: str = "builtin"  # 'sgd', 'adam', 'builtin'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    momentum: float = 0.0
    weight_decay: float = 0.0

    # Convergence settings
    convergence_window: int = 5
    convergence_threshold: int = 2500  # 5 * 500 = 2500 for CartPole-v1

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    save_dir: str = "checkpoints"

    # Random seed
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "TrainingConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))


def save_config(config: TrainingConfig, path: Union[str, Path]) -> None:
    """Save configuration to JSON file.

    Args:
        config: Configuration to save.
        path: File path for saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(config.to_json())


def load_config(path: Union[str, Path]) -> TrainingConfig:
    """Load configuration from JSON file.

    Args:
        path: File path to load from.

    Returns:
        Loaded configuration.
    """
    with open(path, "r") as f:
        return TrainingConfig.from_json(f.read())


# Preset configurations for common use cases
CARTPOLE_DEFAULT = TrainingConfig()

CARTPOLE_FAST = TrainingConfig(
    action_learning_rate=0.5,
    prediction_learning_rate=0.5,
    action_update_steps=5,
    prediction_update_steps=5,
    use_adaptive_lr=True,
)

CARTPOLE_STABLE = TrainingConfig(
    action_learning_rate=0.01,
    prediction_learning_rate=0.01,
    action_update_steps=1,
    prediction_update_steps=1,
    use_adaptive_lr=False,
    optimizer_type="adam",
)

CARTPOLE_ADAM = TrainingConfig(
    optimizer_type="adam",
    action_learning_rate=0.001,
    prediction_learning_rate=0.001,
    action_update_steps=5,
    prediction_update_steps=5,
    use_adaptive_lr=False,
)
