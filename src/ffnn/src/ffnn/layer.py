from ffnn.types import ActivationFunction, WeightsSetup
from ffnn.activation import Activation


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
        self.weights_setup = weights_setup

        print(f"Layer initialized: {input_size} -> {output_size}")
        print(f"Activation: {activation}")
        print(f"Weights setup: {weights_setup}")
        print()
