import numpy as np
from ffnn.types import ActivationFunction


class Activation:
    @staticmethod
    def get_activation_from_type(activation: ActivationFunction):
        if activation == ActivationFunction.LINEAR:
            return Activation.Linear()
        elif activation == ActivationFunction.RELU:
            return Activation.ReLU()
        elif activation == ActivationFunction.SIGMOID:
            return Activation.Sigmoid()
        elif activation == ActivationFunction.TANH:
            return Activation.TanH()
        elif activation == ActivationFunction.SOFTMAX:
            return Activation.Softmax()

    class Linear:
        def activate(self, z):
            return z

        def derivative(self, z):
            return 1

        def get_activation_type(self):
            return ActivationFunction.LINEAR

    class ReLU:
        def activate(self, z):
            return np.maximum(0, z)

        def derivative(self, z):
            return np.where(z > 0, 1, 0)

        def get_activation_type(self):
            return ActivationFunction.RELU

    class Sigmoid:
        def activate(self, z):
            return 1 / (1 + np.exp(-z))

        def derivative(self, z):
            a = 1 / (1 + np.exp(-z))
            return a * (1 - a)

        def get_activation_type(self):
            return ActivationFunction.SIGMOID

    class TanH:
        def activate(self, z):
            return np.tanh(z)

        def derivative(self, z):
            a = np.tanh(z)
            return 1 - np.square(a)

        def get_activation_type(self):
            return ActivationFunction.TANH

    class Softmax:
        def activate(self, z):
            exp_z = np.exp(z)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

        def derivative(self, z):
            exp_z = np.exp(z)
            softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            # jacobian = np.diag(softmax) - np.outer(softmax, softmax)
            # return jacobian.diagonal
            # TODO: Gatau bener atau ga lol
            return softmax * (1 - softmax)

        def get_activation_type(self):
            return ActivationFunction.SOFTMAX
