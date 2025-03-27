from ffnn.layer import Layer
from ffnn.types import ActivationFunction, LossFunction, WeightsSetup, WeightInitializer
from ffnn.loss import Loss
from random import randint
import numpy as np
import pickle


class FFNN:
    def __init__(
        self,
        layer_sizes: list[int],
        activation_functions: list[ActivationFunction] = None,
        loss_function: LossFunction = LossFunction.MEAN_SQUARED_ERROR,
        weights_setup: list[WeightsSetup] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 100,
        verbose: bool = False,
        random_state: int = None,
    ):
        if activation_functions is None:
            activation_functions = [ActivationFunction.RELU] * (len(layer_sizes - 1))

        if weights_setup is None:
            weights_setup = [WeightsSetup(WeightInitializer.ZERO)] * (
                len(layer_sizes) - 1
            )

        if random_state is None:
            random_state = randint(0, 1000)

        assert (
            len(layer_sizes) == len(activation_functions) + 1
        ), "Number of activation functions should be one less than number of layers"

        assert (
            len(layer_sizes) == len(weights_setup) + 1
        ), "Number of weights setup should be one less than number of layers"

        assert all(
            activation_function
            in [
                ActivationFunction.LINEAR,
                ActivationFunction.RELU,
                ActivationFunction.SIGMOID,
                ActivationFunction.TANH,
                ActivationFunction.SOFTMAX,
            ]
            for activation_function in activation_functions
        ), "Invalid activation function"

        for i in range(len(weights_setup)):
            weights_setup[i].seed = random_state

        self.layers: list[Layer] = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(
                Layer(
                    layer_sizes[i - 1],
                    layer_sizes[i],
                    activation_functions[i - 1],
                    weights_setup[i - 1],
                )
            )

        self.loss_function = Loss.get_loss_from_type(loss_function)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state

        print("FFNN initialized")
        print("Layer sizes:", layer_sizes)
        print("Activation functions:", activation_functions)
        print("Loss function:", loss_function)
        print("Weights setup:", weights_setup)
        print("Learning rate:", learning_rate)
        print("Batch size:", batch_size)
        print("Epochs:", epochs)
        print("Verbose:", verbose)
        print("Random state:", random_state)

    # TODO: Implemen semua method di bawah ini
    def show_graph(self):
        pass

    def plot_weights(self, layers: list[int]):
        pass

    def plot_gradients(self, layers: list[int]):
        pass

    def save_model(self, path: str):
        pickle.dump(self, open(path, "wb"))

    def load_model(self, path: str):
        return pickle.load(open(path, "rb"))

    def forward(self, X):
        current_input = X
        for layer in self.layers:
            # Add bias term (1) to input
            if len(current_input.shape) == 1:
                current_input = np.append(current_input, 1)
            else:
                current_input = np.hstack(
                    [current_input, np.ones((current_input.shape[0], 1))]
                )

            # Store input to layer
            layer.input_value = current_input

            # Calculate output
            z = np.dot(current_input, layer.weights)

            # Store net from layer
            layer.output_value = z

            current_input = layer.activation.activate(z)

        return current_input

    def fit(self, X, Y):
        for i in range(self.epochs):
            print(f"Epoch {i+1}/{self.epochs}")
            self.backward(X, Y)
            print(f"Loss: {self.loss_function.calculate(Y, self.predict(X))}")

    def print_weights(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1} weights:")
            print(layer.weights)
            print()

    def predict(self, X):
        return np.array([self.forward(x) for x in X])

    def set_weights(self, weights: list[np.ndarray]):
        for i, layer in enumerate(self.layers):
            layer.weights = weights[i]

    def backward(self, X, Y):
        layer_grad = [
            np.zeros((layer.output_size, layer.input_size)) for layer in self.layers
        ]
        bias_grad = [np.zeros(layer.output_size) for layer in self.layers]

        for x, y in zip(X, Y):
            # Forward pass
            output = (
                self.forward(x.reshape(1, -1)) if len(x.shape) == 1 else self.forward(x)
            )

            # Calculate dloss/do
            loss_over_outputs = self.loss_function.derivative(y, output.flatten())

            # Backward pass
            current_layer_grad = []
            current_bias_grad = []

            # Output layer
            grad, bias = self.layers[-1].get_gradient(True, loss_over_outputs)
            current_layer_grad.append(grad)
            current_bias_grad.append(bias)

            # Hidden layers
            for i in range(len(self.layers) - 2, -1, -1):
                grad, bias = self.layers[i].get_gradient(
                    False,
                    older_layer_grad=current_layer_grad[0],
                    older_layer_weights=self.layers[i + 1].weights,
                )
                current_layer_grad.insert(0, grad)
                current_bias_grad.insert(0, bias)

            # Accumulate gradients
            for i in range(len(self.layers)):
                layer_grad[i] += current_layer_grad[i]
                bias_grad[i] += current_bias_grad[i]

        # Average gradients
        n_samples = len(X)
        for i in range(len(self.layers)):
            layer_grad[i] /= n_samples
            bias_grad[i] /= n_samples

        # Update weights
        for i, layer in enumerate(self.layers):
            # Update weights (excluding bias)
            layer.weights[:-1] -= self.learning_rate * layer_grad[i].T
            # Update bias (last row of weights)
            layer.weights[-1] -= self.learning_rate * bias_grad[i]
