from ffnn.layer import Layer
from ffnn.types import ActivationFunction, LossFunction, WeightsSetup, WeightInitializer
from ffnn.loss import Loss
from random import randint
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


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
        self.layers_gradient = []
        self.bias_gradient = []

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
        for layer_number in layers:
            data = self.layers[layer_number].weights[:, :-1].flatten()
            bias_data = self.layers[layer_number].weights[:, -1].flatten()

            plt.figure(figsize=(8, 6))
            sns.histplot(data, bins=10, kde=True)
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of weights in the {layer_number}-th layer")
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.histplot(bias_data, bins=10, kde=True, color="red")
            plt.xlabel("Bias value")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of biases in the {layer_number}-th layer")
            plt.show()

    def plot_gradients(self, layers: list[int]):
        for layer_number in layers:

            data = self.layers_gradient[layer_number].weights.flatten()
            plt.figure(figsize=(8, 6))
            sns.histplot(data, bins=10, kde=True)
            plt.xlabel("Weight gradient")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of weights gradients in the {layer_number}-th layer")
            plt.show()

            data = self.bias_gradient[layer_number].weights.flatten()
            plt.figure(figsize=(8, 6))
            sns.histplot(data, bins=10, kde=True, color="red")
            plt.xlabel("Bias gradient")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of bias gradients in the {layer_number}-th layer")
            plt.show()

    def save_model(self, path: str):
        pickle.dump(self, open(path, 'wb'))

    def load_model(self, path: str):
        return pickle.load(open(path, 'rb'))

    def forward(self, X):
        return X
    
    def fit(self, X, Y):
        for i in range(self.epochs):
            self.backward(X, Y)


    def backward(self, X, Y):
        layer_grad = []
        bias_grad = []

        for i, x in enumerate(X):
            current_layer_grad = [[] for i in range(len(self.layers))]
            current_bias_grad = [[] for i in range(len(self.layers))]

            output = self.forward(x)

            # calculate dloss/do
            loss_over_outputs = self.loss_function.derivative(Y[i], output)

            # get output layer grad
            current_layer_grad[len(self.layers)-1], current_bias_grad[len(self.layers)-1] = self.layers[len(self.layers)-1].get_gradient(True, loss_over_outputs)

            # update hidden layer grad
            for i in range(len(self.layers)-2,-1, -1):
                current_layer_grad[i], current_bias_grad[i] = self.layers[i].get_gradient(False, older_layer_grad=current_layer_grad[i+1])

            if i == 0:
                layer_grad = current_layer_grad
                bias_grad = current_bias_grad
                
            else:
                layer_grad = np.add(layer_grad, current_layer_grad)
                bias_grad = np.add(bias_grad, current_bias_grad)
        
        # update in the end of the batch
        self.layers_gradient = layer_grad = [[[element/len(X) for element in d1] for d1 in d2] for d2 in layer_grad]
        self.bias_gradient = bias_grad = [[[element/len(X) for element in d1] for d1 in d2] for d2 in bias_grad]
        for i, layer in enumerate (self.layers):
            layer.update_weight(layer_grad[i], bias_grad[i], self.learning_rate)