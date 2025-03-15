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

    def forward(self, X):
        pass

    def backward(self, X, y):
        pass

    def update(self):
        pass
