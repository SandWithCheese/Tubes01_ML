from ffnn.layer import Layer
from ffnn.types import ActivationFunction, LossFunction, WeightsSetup, WeightInitializer
from ffnn.activation import Activation
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
            activation_functions = [ActivationFunction.RELU] * (len(layer_sizes) - 1)

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

    def show_graph(self):
        # Placeholder for visualization logic
        pass

    def plot_weights(self, layers: list[int]):
        # Placeholder for weights plotting
        pass

    def plot_gradients(self, layers: list[int]):
        # Placeholder for gradients plotting
        pass

    def save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def fit(self, X, Y):
        np.random.seed(self.random_state)
        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(0, len(X), self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                Y_batch = Y_shuffled[i : i + self.batch_size]

                # Forward pass
                self.forward(X_batch)

                # Compute loss
                y_pred = self.layers[-1].output_value
                loss = self.loss_function.calculate(Y_batch, y_pred)

                # Backward pass
                self.backward(Y_batch)

                if self.verbose and (i // self.batch_size) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}, Batch {i//self.batch_size}, Loss: {loss:.4f}"
                    )

    def forward(self, X):
        a = X
        for layer in self.layers:
            layer.input_value = a
            z = np.dot(a, layer.weights) + layer.biases
            a = layer.activation.activate(z)
            layer.z = z
            layer.output_value = a
        return a

    def backward(self, Y):
        y_pred = self.layers[-1].output_value

        # Handle output layer gradient
        if isinstance(self.loss_function, Loss.CategoricalCrossEntropy) and isinstance(
            self.layers[-1].activation, Activation.Softmax
        ):
            delta = self.loss_function.derivative(Y, y_pred)
        else:
            loss_derivative = self.loss_function.derivative(Y, y_pred) / 2.0
            activation_derivative = self.layers[-1].activation.derivative(
                self.layers[-1].z
            )
            delta = loss_derivative * activation_derivative

        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = layer.input_value

            # Compute gradients
            grad_weights = np.dot(a_prev.T, delta)
            grad_biases = np.sum(delta, axis=0, keepdims=True)

            # Update parameters
            layer.old_weights = layer.weights.copy()
            layer.weights -= self.learning_rate * grad_weights
            layer.old_biases = layer.biases.copy()
            layer.biases -= self.learning_rate * grad_biases

            # Propagate delta to previous layer
            if i > 0:
                delta = np.dot(delta, layer.old_weights.T)
                prev_activation_derivative = self.layers[i - 1].activation.derivative(
                    self.layers[i - 1].z
                )
                delta *= prev_activation_derivative

    def predict(self, X):
        return self.forward(X)

    def set_weights(self, weights: list[np.ndarray]):
        assert len(weights) == len(
            self.layers
        ), "Number of weights should match number of layers"
        for i, weight in enumerate(weights):
            self.layers[i].weights = weight

    def set_biases(self, biases: list[np.ndarray]):
        assert len(biases) == len(
            self.layers
        ), "Number of biases should match number of layers"
        for i, bias in enumerate(biases):
            self.layers[i].biases = np.array([bias])
