from ffnn.types import ActivationFunction, WeightsSetup
from ffnn.activation import Activation
from ffnn.weight import WeightInitialization
import numpy as np


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction,
        weights_setup: WeightsSetup,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = Activation.get_activation_from_type(activation)
        self.weight_initializer = WeightInitialization.get_weight_initializer_from_type(
            weights_setup
        )
        self.weights = self.weight_initializer.initialize(input_size + 1, output_size)
        self.output_value = None
        self.input_value = None

        print(f"Layer initialized: {input_size} -> {output_size}")
        print(f"Activation: {activation}")
        print(f"Weights setup: {weights_setup}")
        print(f"Weight initializer: {self.weight_initializer.get_initializer_type()}")
        print()

    def update_weight(self, weight_grad, bias_grad, learning_rate):
        # Update weights (excluding bias)
        self.weights[:-1] -= learning_rate * weight_grad
        # Update bias (last row of weights)
        self.weights[-1] -= learning_rate * bias_grad

    def get_gradient(
        self,
        output_layer: bool,
        loss_over_output=None,
        older_layer_grad=None,
        older_layer_weights=None,
    ):
        if loss_over_output is None:
            loss_over_output = np.array([])
        if older_layer_grad is None:
            older_layer_grad = np.array([])
        if older_layer_weights is None:
            older_layer_weights = np.array([])

        # calculate dloss/dnet
        node_delta = self.calculate_node_delta(
            output_layer, loss_over_output, older_layer_grad, older_layer_weights
        )

        # calculate dloss/dw
        if node_delta.ndim == 1:
            gradient = np.outer(self.input_value, node_delta)
        else:
            gradient = np.dot(
                self.input_value.T,
                node_delta.reshape(1, -1),
            )

        bias_grad = node_delta

        return gradient[:-1].T, bias_grad

    def calculate_node_delta(
        self,
        output_layer: bool,
        loss_over_output=None,
        older_layer_grad=None,
        older_layer_weights=None,
    ):
        if loss_over_output is None:
            loss_over_output = np.array([])
        if older_layer_grad is None:
            older_layer_grad = np.array([])
        if older_layer_weights is None:
            older_layer_weights = np.array([])

        node_delta = np.zeros(self.output_size)

        if output_layer:
            # For output layer: δ = ∂L/∂o * ∂o/∂z
            node_delta = loss_over_output * self.activation.derivative(
                self.output_value.flatten()
            )
        else:
            # For hidden layers: δ = (∑ w * δ_next) * ∂o/∂z
            if older_layer_grad.size > 0:
                older_layer_grad = older_layer_grad / (
                    self.activation.activate(self.output_value) + 1e-10
                )
                node_delta = (
                    np.dot(older_layer_weights[:-1], older_layer_grad)
                    * self.activation.derivative(self.output_value).flatten()
                )[:, -1].flatten()
        return node_delta.flatten()
