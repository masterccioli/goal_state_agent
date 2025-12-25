"""Goal State Agent implementation.

This module implements the core goal state agent architecture where:
1. An action policy network produces actions given current state
2. A prediction module predicts the next state given (state, action)
3. The predicted next state is compared to the goal state
4. Gradients flow backward through the prediction module to update the action policy

This is a form of model-based learning where the learned prediction module
serves as a differentiable simulator for action policy optimization.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np

from ..core.layers import LinearLayer
from ..core.losses import MSELoss, Loss
from ..utils.optimizers import Optimizer, Adam, AdaptiveLR
from ..utils.replay_buffer import TransitionBuffer, GoalTransitionBuffer


@dataclass
class AgentConfig:
    """Configuration for the goal state agent.

    Attributes:
        state_dim: Dimension of the state space.
        action_dim: Dimension of the action space (typically 1 for discrete).
        goal_indices: Indices of state dimensions that define the goal.
            For CartPole, [2, 3] corresponds to angle and angular velocity.
        goal_values: Target values for the goal state dimensions.
        action_learning_rate: Learning rate for action policy updates.
        prediction_learning_rate: Learning rate for prediction module updates.
        action_update_steps: Number of gradient steps per action update.
        prediction_update_steps: Number of gradient steps per prediction update.
        use_adaptive_lr: Whether to scale learning rate by gradient magnitude.
        action_threshold: Threshold for converting continuous action to discrete.

        # Two-buffer replay system:
        # - TransitionBuffer for prediction module (world model)
        # - GoalTransitionBuffer for action policy (uses local targets)
        use_replay_buffer: Whether to use the two-buffer replay system.
        replay_buffer_capacity: Maximum transitions to store.
        replay_batch_size: Samples per training batch.
        min_buffer_size: Minimum samples before using buffer.
        use_local_targets: Use achieved next_states as targets instead of global goal.
        local_target_percentile: Use transitions in this percentile of goal distance.

        gradient_clip_norm: Maximum gradient norm (None to disable).
    """
    state_dim: int = 4
    action_dim: int = 1
    goal_indices: List[int] = field(default_factory=lambda: [2, 3])
    goal_values: np.ndarray = field(default_factory=lambda: np.zeros(2))
    action_learning_rate: float = 0.5
    prediction_learning_rate: float = 0.5
    action_update_steps: int = 5
    prediction_update_steps: int = 5
    use_adaptive_lr: bool = True
    action_threshold: float = 0.5

    # Two-buffer replay settings
    use_replay_buffer: bool = False
    replay_buffer_capacity: int = 10000
    replay_batch_size: int = 32
    min_buffer_size: int = 100
    use_local_targets: bool = False  # Use achieved next_states as targets (experimental)
    local_target_percentile: float = 25.0  # Sample from best % of transitions
    use_buffer_states: bool = False  # Sample states from buffer (with global goal)

    # Gradient clipping
    gradient_clip_norm: Optional[float] = None


class GoalStateAgent:
    """Goal State Agent with coupled action policy and prediction module.

    This agent learns to take actions that drive the predicted next state
    toward a specified goal state. The key architectural feature is that
    gradients from the goal state error flow through the prediction module
    to update the action policy.

    Architecture:
        Action Policy: state -> action (LinearLayer: state_dim -> action_dim)
        Prediction Module: [state, action] -> next_state (LinearLayer: state_dim + action_dim -> state_dim)

    The agent can be extended with:
        - Deeper networks (multiple layers)
        - Different activation functions
        - Multiple goal states (goal-conditioned policy)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        action_optimizer: Optional[Optimizer] = None,
        prediction_optimizer: Optional[Optimizer] = None,
    ):
        """Initialize the goal state agent.

        Args:
            config: Agent configuration. Uses defaults if None.
            action_optimizer: Optimizer for action policy. Uses Adam if None.
            prediction_optimizer: Optimizer for prediction module. Uses Adam if None.
        """
        self.config = config or AgentConfig()

        # Initialize networks
        self.action_policy = LinearLayer(
            self.config.state_dim,
            self.config.action_dim,
        )
        self.prediction_module = LinearLayer(
            self.config.state_dim + self.config.action_dim,
            self.config.state_dim,
        )

        # Loss function for goal state comparison
        self.loss_fn = MSELoss()

        # Optimizers (optional, can use built-in gradient descent)
        self.action_optimizer = action_optimizer
        self.prediction_optimizer = prediction_optimizer

        # Adaptive learning rate helper
        self.adaptive_lr = AdaptiveLR(base_learning_rate=1.0)

        # Memory for previous state/action (needed for prediction module training)
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[np.ndarray] = None

        # Initialize two-buffer system if enabled
        self.transition_buffer: Optional[TransitionBuffer] = None
        self.goal_buffer: Optional[GoalTransitionBuffer] = None
        if self.config.use_replay_buffer:
            # Buffer for prediction module: (state, action, next_state)
            self.transition_buffer = TransitionBuffer(
                capacity=self.config.replay_buffer_capacity
            )
            # Buffer for action policy: (state, next_state) with goal distance
            self.goal_buffer = GoalTransitionBuffer(
                capacity=self.config.replay_buffer_capacity,
                goal_indices=self.config.goal_indices,
                goal_values=self.config.goal_values,
            )

        # Metrics tracking
        self.metrics = {
            "goal_errors": [],
            "prediction_errors": [],
            "actions": [],
        }

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get discrete action from the action policy.

        Args:
            state: Current state observation.
            training: Whether to update the action policy.

        Returns:
            Discrete action (0 or 1 for CartPole).
        """
        state = np.atleast_2d(state)

        # Get continuous action from policy
        action_raw = self.action_policy.forward(state)

        # Convert to discrete action
        action = 1 if action_raw.item() > self.config.action_threshold else 0

        return action

    def get_action_raw(self, state: np.ndarray) -> np.ndarray:
        """Get raw continuous action output.

        Args:
            state: Current state observation.

        Returns:
            Continuous action value.
        """
        state = np.atleast_2d(state)
        return self.action_policy.forward(state)

    def _clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradient by norm if clipping is enabled.

        Args:
            gradient: Gradient array to clip.

        Returns:
            Clipped gradient (or original if clipping disabled).
        """
        if self.config.gradient_clip_norm is None:
            return gradient

        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.config.gradient_clip_norm:
            return gradient * (self.config.gradient_clip_norm / grad_norm)
        return gradient

    def update_prediction_module(
        self,
        current_state: np.ndarray,
    ) -> float:
        """Update the prediction module using the previous state transition.

        Trains the prediction module to predict the current state from
        the previous (state, action) pair. Uses TransitionBuffer for
        diverse training data when enabled.

        Args:
            current_state: The current observed state.

        Returns:
            Prediction error (MSE loss).
        """
        if self._prev_state is None or self._prev_action is None:
            return 0.0

        current_state = np.atleast_2d(current_state)
        prev_state = np.atleast_2d(self._prev_state)
        prev_action = np.atleast_2d(self._prev_action)

        # Add transition to buffers if enabled
        if self.transition_buffer is not None:
            self.transition_buffer.add(
                self._prev_state.flatten(),
                self._prev_action.flatten(),
                current_state.flatten(),
            )
        if self.goal_buffer is not None:
            self.goal_buffer.add(
                self._prev_state.flatten(),
                current_state.flatten(),
            )

        total_error = 0.0

        # CRITICAL INSIGHT: The prediction module must stay tuned to CURRENT dynamics
        # for the action policy gradients to be useful. Using buffer here breaks
        # the coupling between prediction accuracy and action gradient quality.
        # Always use online learning for the prediction module.

        for _ in range(self.config.prediction_update_steps):
            # Always use current transition (online learning)
            # This keeps the prediction module tuned to current state distribution
            pred_input = np.concatenate([prev_state, prev_action], axis=1)
            target_state = current_state

            # Forward pass
            predicted_state = self.prediction_module.forward(pred_input)

            # Compute loss and gradient
            error = self.loss_fn.compute(predicted_state, target_state)
            gradient = self.loss_fn.gradient(predicted_state, target_state)

            # Apply gradient clipping
            gradient = self._clip_gradient(gradient)

            # Compute and apply gradients
            self.prediction_module.compute_parameter_gradients(pred_input)

            if self.prediction_optimizer is not None:
                # Use external optimizer
                self.prediction_module.apply_gradients([], gradient, learning_rate=1.0)
                new_W = self.prediction_optimizer.step(
                    self.prediction_module.W,
                    self.prediction_module.combined_gradient,
                )
                self.prediction_module.set_weights(new_W)
            else:
                # Use built-in gradient descent with adaptive LR
                lr = self.config.prediction_learning_rate
                if self.config.use_adaptive_lr:
                    lr *= self.adaptive_lr.get_lr(gradient)
                self.prediction_module.apply_gradients([], gradient, learning_rate=lr)

            total_error = error

        return total_error

    def update_action_policy(
        self,
        current_state: np.ndarray,
    ) -> float:
        """Update the action policy to minimize goal state error.

        This is the key mechanism: gradients from the goal state error
        flow through the prediction module's weights to update the
        action policy.

        Two training modes:
        1. Global goal (default): Train to reach the specified goal state
        2. Local targets (with buffer): Train to reach achieved next_states
           that were close to goal. These are "achievable" targets.

        Args:
            current_state: The current observed state.

        Returns:
            Goal state error (MSE loss).
        """
        current_state = np.atleast_2d(current_state)
        goal_values = np.atleast_2d(self.config.goal_values)
        goal_indices = self.config.goal_indices

        total_error = 0.0

        # Check if we should use buffer for action policy training
        use_local_targets = (
            self.config.use_local_targets
            and self.goal_buffer is not None
            and self.goal_buffer.is_ready(self.config.min_buffer_size)
        )
        use_buffer_states = (
            self.config.use_buffer_states
            and self.goal_buffer is not None
            and self.goal_buffer.is_ready(self.config.min_buffer_size)
        )

        for _ in range(self.config.action_update_steps):
            if use_local_targets:
                # Sample good (state, next_state) pairs from buffer
                # Train toward achieved next_states (experimental)
                batch_states, batch_next_states, _ = self.goal_buffer.sample_good_transitions(
                    batch_size=self.config.replay_batch_size,
                    percentile=self.config.local_target_percentile,
                )

                if batch_states is None:
                    training_states = current_state
                    target_values = goal_values
                else:
                    training_states = batch_states
                    target_values = batch_next_states[:, goal_indices]
            elif use_buffer_states:
                # Sample states from buffer, but use GLOBAL goal as target
                # This provides state diversity without changing the objective
                batch_states, _, _ = self.goal_buffer.sample_good_transitions(
                    batch_size=self.config.replay_batch_size,
                    percentile=50.0,  # More diverse sampling
                )

                if batch_states is None:
                    training_states = current_state
                else:
                    training_states = batch_states
                # Always use global goal
                target_values = np.tile(goal_values, (training_states.shape[0], 1))
            else:
                # Use current state with global goal (original behavior)
                training_states = current_state
                target_values = goal_values

            # Get current action from policy
            action_raw = self.action_policy.forward(training_states)

            # Concatenate state and action for prediction
            pred_input = np.concatenate([training_states, action_raw], axis=1)

            # Predict next state
            predicted_next_state = self.prediction_module.forward(pred_input)

            # Extract goal-relevant dimensions
            predicted_goal_dims = predicted_next_state[:, goal_indices]

            # Compute loss and gradient against target (local or global)
            error = self.loss_fn.compute(predicted_goal_dims, target_values)
            goal_gradient = self.loss_fn.gradient(predicted_goal_dims, target_values)

            # Apply gradient clipping
            goal_gradient = self._clip_gradient(goal_gradient)

            # Compute parameter gradients for action policy
            self.action_policy.compute_parameter_gradients(training_states)

            # Get prediction module's pass-through gradient
            # This is the weight matrix showing how action affects predicted state
            pred_pass_through = self.prediction_module.get_pass_through_gradient()

            # Extract the relevant portion: how action affects goal dimensions
            # Rows: action input index (last row(s) of prediction input)
            # Cols: goal state indices
            action_input_idx = self.config.state_dim  # Action starts after state
            action_to_goal_gradient = pred_pass_through[
                action_input_idx : action_input_idx + self.config.action_dim,
                goal_indices,
            ]

            # Apply gradients with pass-through from prediction module
            if self.action_optimizer is not None:
                self.action_policy.apply_gradients(
                    [action_to_goal_gradient],
                    goal_gradient,
                    learning_rate=1.0,
                )
                new_W = self.action_optimizer.step(
                    self.action_policy.W,
                    self.action_policy.combined_gradient,
                )
                self.action_policy.set_weights(new_W)
            else:
                lr = self.config.action_learning_rate
                if self.config.use_adaptive_lr:
                    lr *= self.adaptive_lr.get_lr(goal_gradient)
                self.action_policy.apply_gradients(
                    [action_to_goal_gradient],
                    goal_gradient,
                    learning_rate=lr,
                )

            total_error = error

        return total_error

    def step(
        self,
        current_state: np.ndarray,
    ) -> Tuple[int, float, float]:
        """Perform a full agent step: update networks and get action.

        Args:
            current_state: Current state observation.

        Returns:
            Tuple of (action, goal_error, prediction_error).
        """
        current_state = np.atleast_2d(current_state)

        # Update prediction module using previous transition
        pred_error = self.update_prediction_module(current_state)

        # Update action policy using goal state error
        goal_error = self.update_action_policy(current_state)

        # Get action
        action_raw = self.get_action_raw(current_state)
        action = 1 if action_raw.item() > self.config.action_threshold else 0

        # Store for next step
        self._prev_state = current_state.copy()
        self._prev_action = action_raw.copy()

        # Track metrics
        self.metrics["goal_errors"].append(goal_error)
        self.metrics["prediction_errors"].append(pred_error)
        self.metrics["actions"].append(action)

        return action, goal_error, pred_error

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self._prev_state = None
        self._prev_action = None

    def reset_metrics(self) -> None:
        """Clear metrics tracking."""
        self.metrics = {
            "goal_errors": [],
            "prediction_errors": [],
            "actions": [],
        }

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current network weights.

        Returns:
            Tuple of (action_policy_weights, prediction_module_weights).
        """
        return self.action_policy.W.copy(), self.prediction_module.W.copy()

    def set_weights(
        self,
        action_weights: np.ndarray,
        prediction_weights: np.ndarray,
    ) -> None:
        """Set network weights.

        Args:
            action_weights: Weights for action policy.
            prediction_weights: Weights for prediction module.
        """
        self.action_policy.W = action_weights.copy()
        self.prediction_module.W = prediction_weights.copy()
