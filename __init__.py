"""Goal State Agent Library.

A clean implementation of goal state agents for model-based reinforcement learning.
The agent learns to take actions that drive predicted next states toward a goal.

Architecture:
    - Action Policy: Maps state -> action
    - Prediction Module: Maps (state, action) -> next_state
    - Goal State: Target state to achieve

The key mechanism is that gradients from goal state error flow through the
prediction module to update the action policy.

Example:
    from goal_state_agent import GoalStateAgent, Trainer, TrainingConfig

    # Train with default config
    config = TrainingConfig(max_episodes=500)
    trainer = Trainer(config=config)
    metrics = trainer.train()

    # Evaluate
    results = trainer.evaluate(num_episodes=10)
    print(f"Average episode length: {results['mean_episode_length']}")

For more control:
    from goal_state_agent.agents import GoalStateAgent, AgentConfig
    from goal_state_agent.configs import TrainingConfig, CARTPOLE_FAST
    from goal_state_agent.utils import Adam

    agent = GoalStateAgent(
        config=AgentConfig(goal_values=np.array([0.0, 0.0])),
        action_optimizer=Adam(learning_rate=0.001),
    )
"""

__version__ = "0.1.0"

from .agents.goal_state_agent import GoalStateAgent, AgentConfig
from .configs.config import (
    TrainingConfig,
    load_config,
    save_config,
    CARTPOLE_DEFAULT,
    CARTPOLE_FAST,
    CARTPOLE_STABLE,
    CARTPOLE_ADAM,
    CARTPOLE_REPLAY,
    CARTPOLE_CLIPPED,
    CARTPOLE_LOCAL_TARGETS,
    CARTPOLE_LR_DECAY,
    CARTPOLE_GOLDILOCKS,
    CARTPOLE_SURPRISE,
    CARTPOLE_OPTIMAL,
)
from .training import Trainer, TrainingMetrics, train_agent
from .core.layers import LinearLayer, TanhActivation, Network
from .core.losses import MSELoss, LogLoss, CrossEntropyLoss
from .utils.optimizers import (
    SGD, Adam, AdaptiveLR,
    LRScheduler, ExponentialDecay, StepDecay, LinearDecay,
    SurpriseBasedLR, InverseSurpriseLR,
)
from .utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# Visualization imports (optional dependency)
try:
    from .visualization import (
        plot_training_metrics,
        plot_episode_lengths,
        plot_errors,
        record_episode,
        record_learning_progression,
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

__all__ = [
    # Main classes
    "GoalStateAgent",
    "AgentConfig",
    "Trainer",
    "TrainingMetrics",
    # Configuration
    "TrainingConfig",
    "load_config",
    "save_config",
    "CARTPOLE_DEFAULT",
    "CARTPOLE_FAST",
    "CARTPOLE_STABLE",
    "CARTPOLE_ADAM",
    "CARTPOLE_REPLAY",
    "CARTPOLE_CLIPPED",
    "CARTPOLE_LOCAL_TARGETS",
    "CARTPOLE_LR_DECAY",
    "CARTPOLE_GOLDILOCKS",
    "CARTPOLE_SURPRISE",
    "CARTPOLE_OPTIMAL",
    # Convenience functions
    "train_agent",
    # Core components
    "LinearLayer",
    "TanhActivation",
    "Network",
    "MSELoss",
    "LogLoss",
    "CrossEntropyLoss",
    # Replay buffer
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    # Optimizers
    "SGD",
    "Adam",
    "AdaptiveLR",
    # Learning rate schedulers
    "LRScheduler",
    "ExponentialDecay",
    "StepDecay",
    "LinearDecay",
    "SurpriseBasedLR",
    "InverseSurpriseLR",
    # Visualization (optional)
    "plot_training_metrics",
    "plot_episode_lengths",
    "plot_errors",
    "record_episode",
    "record_learning_progression",
]
