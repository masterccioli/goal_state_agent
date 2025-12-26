"""Loss function implementations for goal state agent training."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute the loss value.

        Args:
            y_pred: Predicted values, shape (batch_size, output_dim).
            y_true: Target values, shape (batch_size, output_dim) or broadcastable.

        Returns:
            Scalar loss value.
        """
        pass

    @abstractmethod
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions.

        Args:
            y_pred: Predicted values, shape (batch_size, output_dim).
            y_true: Target values, shape (batch_size, output_dim) or broadcastable.

        Returns:
            Gradient array of same shape as y_pred.
        """
        pass

    def __call__(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute both loss and gradient.

        Args:
            y_pred: Predicted values.
            y_true: Target values.

        Returns:
            Tuple of (loss_value, gradient).
        """
        return self.compute(y_pred, y_true), self.gradient(y_pred, y_true)


class MSELoss(Loss):
    """Mean Squared Error loss.

    L = 0.5 * mean((y_pred - y_true)^2)
    dL/dy_pred = (y_pred - y_true) / batch_size

    Uses numerical stability measures to prevent overflow with large differences.
    """

    def __init__(self, reduction: str = "mean", max_diff: float = 1e6):
        """Initialize MSE loss.

        Args:
            reduction: How to reduce the loss. Options: 'mean', 'sum', 'none'.
            max_diff: Maximum difference value (clips to prevent overflow).
        """
        self.reduction = reduction
        self.max_diff = max_diff

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute MSE loss with overflow protection."""
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)

        diff = y_pred - y_true
        # Clip difference to prevent overflow in squaring
        diff = np.clip(diff, -self.max_diff, self.max_diff)
        squared = 0.5 * np.power(diff, 2)

        if self.reduction == "mean":
            return squared.sum() / (squared.shape[0] * squared.shape[1])
        elif self.reduction == "sum":
            return squared.sum()
        else:  # 'none'
            return squared

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of MSE loss with overflow protection."""
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)

        diff = y_pred - y_true
        # Clip difference to prevent exploding gradients
        diff = np.clip(diff, -self.max_diff, self.max_diff)

        if self.reduction == "mean":
            return diff / diff.shape[0]
        elif self.reduction == "sum":
            return diff
        else:
            return diff


class LogLoss(Loss):
    """Binary Cross-Entropy (Log Loss).

    L = -[y * log(y_pred) + (1-y) * log(1-y_pred)]
    """

    def __init__(self, eps: float = 1e-15):
        """Initialize log loss.

        Args:
            eps: Small value for numerical stability.
        """
        self.eps = eps

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute log loss."""
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)

        # Clip for numerical stability
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)

        loss = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        return loss.sum()

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of log loss."""
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)

        # Simplified gradient: (y_pred - y_true) for sigmoid output
        return y_pred - y_true


class CrossEntropyLoss(Loss):
    """Cross-Entropy loss for multi-class classification.

    L = -sum(y_true * log(y_pred))

    Assumes y_pred is already softmax output.
    """

    def __init__(self, eps: float = 1e-15):
        """Initialize cross-entropy loss.

        Args:
            eps: Small value for numerical stability.
        """
        self.eps = eps

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)

        # Clip for numerical stability
        y_pred = np.clip(y_pred, self.eps, 1.0)

        return -(y_true * np.log(y_pred)).sum()

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of cross-entropy loss."""
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)

        # Clip for numerical stability
        y_pred = np.clip(y_pred, self.eps, 1.0)

        return -y_true / y_pred
