"""Experience replay buffer for stable online learning."""

from collections import deque
from typing import List, Tuple, Optional
import random
import numpy as np


class TransitionBuffer:
    """Buffer for prediction module: stores full (state, action, next_state) transitions.

    This buffer is used to train the prediction module (world model) on diverse
    transitions from the environment.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        """Add a transition."""
        self.buffer.append((
            np.asarray(state).flatten(),
            np.asarray(action).flatten(),
            np.asarray(next_state).flatten(),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions."""
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        next_states = np.array([t[2] for t in batch])
        return states, actions, next_states

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size


class GoalTransitionBuffer:
    """Buffer for action policy: stores (state, next_state, goal_distance) tuples.

    This buffer stores transitions and their goal distance, allowing the action
    policy to learn from states that led to good outcomes. The key insight is
    that actual next_states are achievable targets (they happened!), unlike
    the global goal which may be far away.

    Training approach:
    - Sample (state, next_state) pairs where next_state was close to goal
    - Train action policy so P(state, π(state)) ≈ next_state
    - This uses achievable local targets instead of global goal
    """

    def __init__(
        self,
        capacity: int = 10000,
        goal_indices: List[int] = None,
        goal_values: np.ndarray = None,
    ):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.goal_indices = goal_indices or [2, 3]
        self.goal_values = goal_values if goal_values is not None else np.zeros(len(self.goal_indices))

    def add(self, state: np.ndarray, next_state: np.ndarray) -> None:
        """Add a (state, next_state) pair with computed goal distance."""
        state = np.asarray(state).flatten()
        next_state = np.asarray(next_state).flatten()

        # Compute goal distance for next_state
        goal_dims = next_state[self.goal_indices]
        goal_distance = np.sum((goal_dims - self.goal_values) ** 2)

        self.buffer.append((state, next_state, goal_distance))

    def sample_good_transitions(
        self,
        batch_size: int,
        threshold: float = None,
        percentile: float = 25.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample transitions where next_state was close to goal.

        Args:
            batch_size: Number of samples.
            threshold: Maximum goal distance. If None, uses percentile.
            percentile: Use this percentile of goal distances as threshold.

        Returns:
            (states, next_states, goal_distances) arrays.
        """
        if len(self.buffer) == 0:
            return None, None, None

        # Compute threshold from percentile if not specified
        if threshold is None:
            all_distances = [t[2] for t in self.buffer]
            threshold = np.percentile(all_distances, percentile)

        # Filter for good transitions
        good_transitions = [t for t in self.buffer if t[2] <= threshold]

        if len(good_transitions) == 0:
            # Fall back to best available
            sorted_buffer = sorted(self.buffer, key=lambda t: t[2])
            good_transitions = sorted_buffer[:max(1, len(sorted_buffer) // 4)]

        batch_size = min(batch_size, len(good_transitions))
        batch = random.sample(good_transitions, batch_size)

        states = np.array([t[0] for t in batch])
        next_states = np.array([t[1] for t in batch])
        goal_distances = np.array([t[2] for t in batch])

        return states, next_states, goal_distances

    def sample_weighted(
        self,
        batch_size: int,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample transitions weighted by inverse goal distance.

        Lower goal distance = higher probability of being sampled.

        Args:
            batch_size: Number of samples.
            temperature: Controls sharpness (lower = more focused on best).

        Returns:
            (states, next_states, goal_distances) arrays.
        """
        if len(self.buffer) == 0:
            return None, None, None

        batch_size = min(batch_size, len(self.buffer))

        # Compute sampling weights (inverse distance with temperature)
        distances = np.array([t[2] for t in self.buffer])
        weights = 1.0 / (distances + 1e-6)  # Avoid division by zero
        weights = weights ** (1.0 / temperature)  # Temperature scaling
        weights = weights / weights.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=weights, replace=False)

        states = np.array([self.buffer[i][0] for i in indices])
        next_states = np.array([self.buffer[i][1] for i in indices])
        goal_distances = np.array([self.buffer[i][2] for i in indices])

        return states, next_states, goal_distances

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {"count": 0}

        distances = [t[2] for t in self.buffer]
        return {
            "count": len(self.buffer),
            "mean_goal_dist": np.mean(distances),
            "min_goal_dist": np.min(distances),
            "max_goal_dist": np.max(distances),
            "p25_goal_dist": np.percentile(distances, 25),
            "p50_goal_dist": np.percentile(distances, 50),
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions.

    Stores (state, action, next_state) transitions and provides random
    sampling for mini-batch training. This breaks the temporal correlation
    in sequential data and prevents catastrophic forgetting.

    Attributes:
        capacity: Maximum number of transitions to store.
        buffer: Internal storage deque.
    """

    def __init__(self, capacity: int = 10000):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store. When full,
                oldest transitions are discarded.
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: State observation (flattened).
            action: Action taken (can be raw continuous value).
            next_state: Resulting next state.
        """
        # Ensure arrays are flattened for consistent storage
        state = np.asarray(state).flatten()
        action = np.asarray(action).flatten()
        next_state = np.asarray(next_state).flatten()

        self.buffer.append((state, action, next_state))

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, next_states) arrays, each with
            shape (batch_size, dim).
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        next_states = np.array([t[2] for t in batch])

        return states, actions, next_states

    def sample_states(self, batch_size: int) -> np.ndarray:
        """Sample only states (for action policy training).

        Args:
            batch_size: Number of states to sample.

        Returns:
            Array of states with shape (batch_size, state_dim).
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)
        return np.array([t[0] for t in batch])

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training.

        Args:
            min_size: Minimum required samples.

        Returns:
            True if buffer has at least min_size samples.
        """
        return len(self.buffer) >= min_size

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized sampling (optional enhancement).

    Transitions with higher TD error or loss are sampled more frequently.
    This can accelerate learning by focusing on "surprising" transitions.

    Note: This is a simplified implementation. Full prioritized experience
    replay (PER) uses a sum-tree for O(log n) sampling.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size.
            alpha: Priority exponent (0 = uniform, 1 = full prioritization).
            beta: Importance sampling exponent (annealed to 1).
            beta_increment: Beta increment per sample call.
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities: deque = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        priority: Optional[float] = None,
    ) -> None:
        """Add transition with priority."""
        super().add(state, action, next_state)
        self.priorities.append(priority if priority else self.max_priority)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Sample batch with priorities.

        Returns:
            Tuple of (states, actions, next_states, weights, indices).
            Weights are importance sampling weights for unbiased gradients.
        """
        batch_size = min(batch_size, len(self.buffer))

        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Get transitions
        states = np.array([self.buffer[i][0] for i in indices])
        actions = np.array([self.buffer[i][1] for i in indices])
        next_states = np.array([self.buffer[i][2] for i in indices])

        # Compute importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        return states, actions, next_states, weights, list(indices)

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero
            self.max_priority = max(self.max_priority, priority)
