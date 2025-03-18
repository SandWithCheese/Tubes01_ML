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

        print(f"Layer initialized: {input_size} -> {output_size}")
        print(f"Activation: {activation}")
        print(f"Weights setup: {weights_setup}")
        print(f"Weight initializer: {self.weight_initializer.get_initializer_type()}")
        print()
