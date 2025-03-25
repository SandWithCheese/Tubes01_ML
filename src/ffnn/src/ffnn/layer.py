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
        self.output_value = []
        self.input_value = []

        print(f"Layer initialized: {input_size} -> {output_size}")
        print(f"Activation: {activation}")
        print(f"Weights setup: {weights_setup}")
        print(f"Weight initializer: {self.weight_initializer.get_initializer_type()}")
        print()

    def update_weight(self, weight_grad, bias_grad, learning_rate):
        for i, grads in enumerate(weight_grad):
            # update weights
            for j, grad in enumerate(grads):
               self.weights[i][j] += learning_rate * grad
            # update bias
            self.weights[i][self.input_size] += learning_rate * bias_grad[i]
    
    def get_gradient(self, output_layer: bool, loss_over_output = [], older_layer_grad = []):

        # calculate dloss/dnet
        node_delta = self.calculate_node_delta(output_layer, loss_over_output, older_layer_grad)
        bias_grad = [0 for i in range(self.output_size)]
        gradient = [0 for i in range(self.output_size) for j in range(self.input_size)]

        # calculate dloss/dw
        for i in range(self.output_size):
            for j in range(self.input_size):
                # gradient loss/w
                gradient[i][j] = self.input_value[j] * node_delta[i]
            # bias gradient
            bias_grad[i] = node_delta[i]
        return gradient, bias_grad                

    def calculate_node_delta(self, output_layer: bool, loss_over_output = [], older_layer_grad = []):

        node_delta = [0 for i in range(self.output_size)]
        # output layer
        if output_layer:
            # multiply loss/o with o/net
            for i in range(self.output_size):
                node_delta[i] = loss_over_output[i] * self.activation.derivative(self.output_value[i])
        # hidden layer
        else:
            for i in range(self.output_size):
                # get sum from all gradient of weight from other layer
                for j in range(len(older_layer_grad[i])):
                    node_delta[i] += older_layer_grad[i][j]
                node_delta[i] *= self.activation.derivative(self.output_value[i])

        return node_delta
        
        
