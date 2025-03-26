from ffnn.layer import Layer
from ffnn.types import ActivationFunction, LossFunction, WeightsSetup, WeightInitializer
from ffnn.loss import Loss
from random import randint


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
        pass

    def load_model(self, path: str):
        pass

    def forward(self, X): # Bro dokumentasi lah bro
        """
        Perform forward propagation through the neural network.
        
        Args:
            X (numpy.ndarray): Input data

        Returns:
            numpy.ndarray: Output of the final layer
        """
        # Current input is the input data
        current_input = X

        # Iterate through each layer and apply transformations
        for layer in self.layers:
            # Compute the pre-activation (z) values (atau kalo di slide disebut net)
            # z = weights * input + bias
            z = layer.weights @ current_input
            
            # Apply activation function to get layer's output
            current_input = layer.activation.activate(z)

        # Return the output of the final layer
        return current_input

    # backward propragation for single instance
    def backward(self, X, y):
        # calculate current result
        output = self.forward(X)

        # calculate dloss/do
        loss_over_outputs = self.loss_function.derivative(y, output)

        # update output layer
        current_layer_grad, bias_grad = self.layers[len(self.layers)-1].get_gradient(True, loss_over_outputs)

        # update hidden layer
        self.layers[len(self.layers)-1].update_layer(current_layer_grad, bias_grad, self.learning_rate)
        for i in range(len(self.layers)-2,0):
            current_layer_grad, bias_grad = self.layers[i].get_gradient(False, older_layer_grad=current_layer_grad)
            self.layers[i].update_layer(current_layer_grad, bias_grad, self.learning_rate)