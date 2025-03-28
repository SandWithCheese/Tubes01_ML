from ffnn.types import ActivationFunction, WeightsSetup
from ffnn.activation import Activation
from ffnn.weight import WeightInitialization


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
        self.weights = self.weight_initializer.initialize(input_size, output_size)
        self.biases = self.weight_initializer.initialize(1, output_size)
        self.old_weights = self.weights
        self.old_biases = self.biases
        self.output_value = []
        self.input_value = []
        self.z = []
