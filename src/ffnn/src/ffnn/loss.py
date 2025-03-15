import numpy as np
from ffnn.types import LossFunction


class Loss:
    @staticmethod
    def get_loss_from_type(loss: LossFunction):
        if loss == LossFunction.MEAN_SQUARED_ERROR:
            return Loss.MeanSquaredError()
        elif loss == LossFunction.BINARY_CROSS_ENTROPY:
            return Loss.BinaryCrossEntropy()
        elif loss == LossFunction.CATEGORICAL_CROSS_ENTROPY:
            return Loss.CategoricalCrossEntropy()

    class MeanSquaredError:
        def calculate(y_true, y_pred):
            return np.mean(np.square(y_true - y_pred))

        def derivative(y_true, y_pred):
            n = y_true.shape[0]
            return 2 * (y_pred - y_true) / n

        def get_loss_type():
            return LossFunction.MEAN_SQUARED_ERROR

    # TODO: Cek lagi kedua class di bawah
    class BinaryCrossEntropy:
        def calculate(y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        def derivative(y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            n = y_true.shape[0]
            return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n

        def get_loss_type():
            return LossFunction.BINARY_CROSS_ENTROPY

    class CategoricalCrossEntropy:
        def calculate(y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1.0)
            n_samples = y_pred.shape[0]
            return -np.sum(np.log(y_pred[np.arange(n_samples), y_true])) / n_samples

        def derivative(y_true, y_pred):
            n_samples = y_pred.shape[0]
            return y_pred - y_true / n_samples

        def get_loss_type():
            return LossFunction.CATEGORICAL_CROSS_ENTROPY
