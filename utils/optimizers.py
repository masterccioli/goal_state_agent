"""Optimizer implementations for goal state agent training."""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np


# =============================================================================
# Learning Rate Schedulers
# =============================================================================


class LRScheduler(ABC):
    """Abstract base class for learning rate schedulers."""

    @abstractmethod
    def get_lr(self, base_lr: float, step: int) -> float:
        """Compute learning rate for current step.

        Args:
            base_lr: Base learning rate.
            step: Current training step.

        Returns:
            Adjusted learning rate.
        """
        pass

    def step(self) -> None:
        """Called after each training step (optional hook)."""
        pass

    def reset(self) -> None:
        """Reset scheduler state."""
        pass


class ExponentialDecay(LRScheduler):
    """Exponential learning rate decay.

    lr = base_lr * decay_rate^(step / decay_steps)
    """

    def __init__(self, decay_rate: float = 0.99, decay_steps: int = 100):
        """Initialize exponential decay scheduler.

        Args:
            decay_rate: Factor to multiply learning rate by.
            decay_steps: Steps between decay applications.
        """
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def get_lr(self, base_lr: float, step: int) -> float:
        return base_lr * (self.decay_rate ** (step / self.decay_steps))


class StepDecay(LRScheduler):
    """Step-based learning rate decay.

    lr drops by decay_factor every step_size steps.
    """

    def __init__(self, step_size: int = 500, decay_factor: float = 0.5):
        """Initialize step decay scheduler.

        Args:
            step_size: Steps between learning rate drops.
            decay_factor: Factor to multiply learning rate by at each drop.
        """
        self.step_size = step_size
        self.decay_factor = decay_factor

    def get_lr(self, base_lr: float, step: int) -> float:
        num_decays = step // self.step_size
        return base_lr * (self.decay_factor ** num_decays)


class LinearDecay(LRScheduler):
    """Linear learning rate decay.

    lr decreases linearly from base_lr to min_lr over total_steps.
    """

    def __init__(self, total_steps: int = 10000, min_lr: float = 0.01):
        """Initialize linear decay scheduler.

        Args:
            total_steps: Steps over which to decay.
            min_lr: Minimum learning rate (final value).
        """
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self, base_lr: float, step: int) -> float:
        if step >= self.total_steps:
            return self.min_lr
        progress = step / self.total_steps
        return base_lr + (self.min_lr - base_lr) * progress


class SurpriseBasedLR(LRScheduler):
    """Goldilocks effect: moderate surprise yields highest learning rate.

    Based on developmental psychology research where moderate novelty
    (not too boring, not too surprising) produces optimal learning.

    Uses a bell curve centered on target_surprise:
    - Low surprise (accurate predictions) → lower learning rate
    - Moderate surprise (some error) → higher learning rate
    - High surprise (large errors) → lower learning rate

    lr = base_lr * exp(-((surprise - target) / width)^2)
    """

    def __init__(
        self,
        target_surprise: float = 0.1,
        width: float = 0.1,
        min_multiplier: float = 0.1,
    ):
        """Initialize Goldilocks learning rate.

        Args:
            target_surprise: The optimal prediction error (MSE) for learning.
            width: Width of the bell curve (controls sensitivity).
            min_multiplier: Minimum learning rate multiplier (prevents zero LR).
        """
        self.target_surprise = target_surprise
        self.width = width
        self.min_multiplier = min_multiplier
        self._current_surprise = 0.0

    def set_surprise(self, prediction_error: float) -> None:
        """Update the current surprise level.

        Args:
            prediction_error: Current prediction module error (MSE).
        """
        self._current_surprise = prediction_error

    def get_lr(self, base_lr: float, step: int) -> float:
        # Bell curve centered on target_surprise
        deviation = (self._current_surprise - self.target_surprise) / self.width
        multiplier = np.exp(-(deviation ** 2))
        # Ensure minimum learning rate
        multiplier = max(multiplier, self.min_multiplier)
        return base_lr * multiplier

    def reset(self) -> None:
        self._current_surprise = 0.0


class InverseSurpriseLR(LRScheduler):
    """High surprise → high learning rate.

    The opposite of Goldilocks: surprising outcomes (large prediction errors)
    yield greater learning, since these are the situations where the model
    needs to update most.

    Uses exponential scaling with saturation:
    lr = base_lr * (1 + scale * tanh(surprise / saturation))

    This gives higher LR when surprised, but saturates to prevent instability.
    """

    def __init__(
        self,
        scale: float = 1.0,
        saturation: float = 0.5,
        baseline: float = 0.5,
    ):
        """Initialize inverse surprise learning rate.

        Args:
            scale: Maximum additional learning rate multiplier.
            saturation: Surprise level at which effect begins to saturate.
            baseline: Minimum learning rate multiplier (at zero surprise).
        """
        self.scale = scale
        self.saturation = saturation
        self.baseline = baseline
        self._current_surprise = 0.0

    def set_surprise(self, prediction_error: float) -> None:
        """Update the current surprise level.

        Args:
            prediction_error: Current prediction module error (MSE).
        """
        self._current_surprise = prediction_error

    def get_lr(self, base_lr: float, step: int) -> float:
        # Higher surprise → higher multiplier, with saturation
        surprise_factor = np.tanh(self._current_surprise / self.saturation)
        multiplier = self.baseline + self.scale * surprise_factor
        return base_lr * multiplier

    def reset(self) -> None:
        self._current_surprise = 0.0


# =============================================================================
# Optimizers
# =============================================================================


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
