"""Neural network layer implementations.

This module provides differentiable layers with explicit gradient computation,
following the original architecture where gradients flow through a prediction
module to update an action policy.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np


class Layer(ABC):
    """Abstract base class for all layers."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        pass

    @abstractmethod
    def get_pass_through_gradient(self) -> np.ndarray:
        """Get gradient of output w.r.t. input (for backprop through layers)."""
        pass

    @abstractmethod
    def compute_parameter_gradients(self, x: np.ndarray) -> None:
        """Compute gradients of output w.r.t. parameters."""
        pass

    @abstractmethod
    def apply_gradients(
        self,
        pass_through_gradients: List[np.ndarray],
        cost_gradient: np.ndarray,
        learning_rate: float,
    ) -> None:
        """Combine gradients and update parameters."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[np.ndarray]:
        """Return list of trainable parameters."""
        pass


class LinearLayer(Layer):
    """Linear transformation layer: y = Wx.

    This layer implements a simple linear transformation without bias.
    Gradients are computed explicitly for use in the goal state agent
    architecture where gradients flow through a prediction module.

    Attributes:
        input_size: Dimension of input features.
        output_size: Dimension of output features.
        W: Weight matrix of shape (input_size, output_size).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        init_scale: Optional[float] = None,
    ):
        """Initialize the linear layer.

        Args:
            input_size: Number of input features.
            output_size: Number of output features.
            init_scale: Scale for weight initialization. If None, uses
                1/sqrt(output_size) (Xavier-like initialization).
        """
        self.input_size = input_size
        self.output_size = output_size

        # Xavier-like initialization
        if init_scale is None:
            init_scale = 1.0 / np.sqrt(output_size)

        self.W = np.random.normal(0, init_scale, size=(input_size, output_size))

        # Gradient storage
        self._pass_through_gradient: Optional[np.ndarray] = None
        self._parameter_gradient: Optional[np.ndarray] = None
        self._combined_weight_gradient: Optional[np.ndarray] = None
        self._last_input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply linear transformation.

        Args:
            x: Input array of shape (batch_size, input_size) or (input_size,).

        Returns:
            Output array of shape (batch_size, output_size) or (output_size,).
        """
        self._last_input = x
        return np.dot(x, self.W)

    def get_pass_through_gradient(self) -> np.ndarray:
        """Get the weight matrix as the pass-through gradient.

        For y = Wx, dy/dx = W.

        Returns:
            Weight matrix W of shape (input_size, output_size).
        """
        self._pass_through_gradient = self.W
        return self._pass_through_gradient

    def compute_parameter_gradients(self, x: np.ndarray) -> None:
        """Compute gradient of output w.r.t. weights.

        For y = Wx, dy/dW[i,j] = x[i] for output dimension j.

        Args:
            x: Input array, can be 1D (input_size,) or 2D (batch, input_size).
        """
        x = np.atleast_2d(x)
        batch_size = x.shape[0]

        # Create derivative matrix: shape (input_size, output_size, output_size)
        # deriv_matrix[i, j, k] = 1 if j == k else 0
        # This represents: dW affects output[j] at position [i, j]
        deriv_matrix = np.zeros((self.input_size, self.output_size, self.output_size))
        for i in range(self.input_size):
            for j in range(self.output_size):
                deriv_matrix[i, j, j] = 1

        # Contract with input: shape (batch, input_size, output_size, output_size)
        self._parameter_gradient = np.einsum("abc, da -> dabc", deriv_matrix, x)

    def apply_gradients(
        self,
        pass_through_gradients: List[np.ndarray],
        cost_gradient: np.ndarray,
        learning_rate: float,
    ) -> None:
        """Combine gradients via chain rule and update weights.

        This implements the key mechanism where gradients from the cost function
        flow back through subsequent layers (via pass_through_gradients) to
        update this layer's parameters.

        Args:
            pass_through_gradients: List of gradients from subsequent layers.
                Each is the gradient of that layer's output w.r.t. its input.
            cost_gradient: Gradient of cost w.r.t. the network output.
            learning_rate: Step size for gradient descent.
        """
        if self._parameter_gradient is None:
            raise RuntimeError("Must call compute_parameter_gradients before apply_gradients")

        if len(pass_through_gradients) > 0:
            # Chain pass-through gradients together
            out = pass_through_gradients[0]
            for grad in pass_through_gradients[1:]:
                out = np.einsum("ab, bc -> ac", out, grad)

            # Multiply with parameter gradient
            out = np.einsum("abcd, de -> abce", self._parameter_gradient, out)
        else:
            out = self._parameter_gradient

        # Contract with cost gradient
        self._combined_weight_gradient = np.einsum("abcd, ad -> bc", out, cost_gradient)

        # Update weights
        new_W = self.W - self._combined_weight_gradient * learning_rate

        # Safety check for numerical stability
        if not (np.isnan(new_W).any() or np.isinf(new_W).any()):
            self.W = new_W

    def set_weights(self, W: np.ndarray) -> None:
        """Directly set weights (for use with external optimizers)."""
        self.W = W

    @property
    def parameters(self) -> List[np.ndarray]:
        """Return list of trainable parameters."""
        return [self.W]

    @property
    def combined_gradient(self) -> Optional[np.ndarray]:
        """Return the most recently computed combined gradient."""
        return self._combined_weight_gradient


class TanhActivation(Layer):
    """Tanh activation function layer.

    Applies element-wise tanh: y = tanh(x).
    """

    def __init__(self):
        self._last_input: Optional[np.ndarray] = None
        self._pass_through_gradient: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply tanh activation."""
        self._last_input = x
        return np.tanh(x)

    def get_pass_through_gradient(self) -> np.ndarray:
        """Compute d(tanh(x))/dx = 1 - tanh(x)^2 = sech(x)^2."""
        if self._last_input is None:
            raise RuntimeError("Must call forward before get_pass_through_gradient")
        self._pass_through_gradient = 1.0 / np.power(np.cosh(self._last_input), 2)
        return self._pass_through_gradient

    def compute_parameter_gradients(self, x: np.ndarray) -> None:
        """Tanh has no trainable parameters."""
        pass

    def apply_gradients(
        self,
        pass_through_gradients: List[np.ndarray],
        cost_gradient: np.ndarray,
        learning_rate: float,
    ) -> None:
        """Tanh has no trainable parameters to update."""
        pass

    @property
    def parameters(self) -> List[np.ndarray]:
        """Tanh has no trainable parameters."""
        return []


class Network:
    """A sequential container for layers.

    This provides a convenient wrapper for building multi-layer networks
    while still allowing access to individual layers for the goal state
    agent's gradient flow mechanism.
    """

    def __init__(self, layers: List[Layer]):
        """Initialize with a list of layers."""
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward."""
        return self.forward(x)

    def get_layer(self, index: int) -> Layer:
        """Get a specific layer by index."""
        return self.layers[index]

    @property
    def parameters(self) -> List[np.ndarray]:
        """Return all trainable parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params
