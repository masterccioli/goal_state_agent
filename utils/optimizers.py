"""Optimizer implementations for goal state agent training."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Update parameters given gradients.

        Args:
            params: Current parameter values.
            grads: Gradient of loss w.r.t. parameters.

        Returns:
            Updated parameter values.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset optimizer state (e.g., for new training run)."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Supports momentum and weight decay.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Initialize SGD optimizer.

        Args:
            learning_rate: Step size for updates.
            momentum: Momentum coefficient (0 = no momentum).
            weight_decay: L2 regularization coefficient.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: Optional[np.ndarray] = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform SGD update step."""
        # Apply weight decay
        if self.weight_decay > 0:
            grads = grads + self.weight_decay * params

        # Apply momentum
        if self.momentum > 0:
            if self._velocity is None:
                self._velocity = np.zeros_like(grads)
            self._velocity = self.momentum * self._velocity + grads
            update = self._velocity
        else:
            update = grads

        return params - self.learning_rate * update

    def reset(self) -> None:
        """Reset momentum state."""
        self._velocity = None


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation).

    Combines momentum with adaptive learning rates per parameter.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """Initialize Adam optimizer.

        Args:
            learning_rate: Step size for updates.
            beta1: Exponential decay rate for first moment estimates.
            beta2: Exponential decay rate for second moment estimates.
            epsilon: Small constant for numerical stability.
            weight_decay: L2 regularization coefficient (AdamW-style).
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self._m: Optional[np.ndarray] = None  # First moment
        self._v: Optional[np.ndarray] = None  # Second moment
        self._t: int = 0  # Time step

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform Adam update step."""
        self._t += 1

        # Initialize moments if needed
        if self._m is None:
            self._m = np.zeros_like(grads)
            self._v = np.zeros_like(grads)

        # Apply weight decay (AdamW style)
        if self.weight_decay > 0:
            params = params - self.learning_rate * self.weight_decay * params

        # Update biased first moment estimate
        self._m = self.beta1 * self._m + (1 - self.beta1) * grads

        # Update biased second moment estimate
        self._v = self.beta2 * self._v + (1 - self.beta2) * (grads ** 2)

        # Compute bias-corrected estimates
        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)

        # Update parameters
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self) -> None:
        """Reset optimizer state."""
        self._m = None
        self._v = None
        self._t = 0


class AdaptiveLR:
    """Adaptive learning rate wrapper that scales by gradient magnitude.

    This implements the gradient normalization used in the original notebooks:
    lr_effective = lr * (1 / ||gradient||)
    """

    def __init__(
        self,
        base_learning_rate: float = 0.5,
        min_lr: float = 1e-6,
        max_lr: float = 10.0,
    ):
        """Initialize adaptive learning rate.

        Args:
            base_learning_rate: Base learning rate to scale.
            min_lr: Minimum learning rate (prevents explosion with small gradients).
            max_lr: Maximum learning rate (prevents tiny updates with large gradients).
        """
        self.base_lr = base_learning_rate
        self.min_lr = min_lr
        self.max_lr = max_lr

    def get_lr(self, gradient: np.ndarray) -> float:
        """Compute adaptive learning rate based on gradient magnitude.

        Args:
            gradient: Gradient array.

        Returns:
            Scaled learning rate.
        """
        grad_magnitude = np.linalg.norm(gradient)

        if grad_magnitude < 1e-10:
            return self.max_lr

        lr = self.base_lr / grad_magnitude
        return np.clip(lr, self.min_lr, self.max_lr)
