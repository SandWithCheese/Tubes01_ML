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
        def calculate(self, y_true, y_pred):
            return np.mean(np.square(y_true - y_pred))

        def derivative(self, y_true, y_pred):
            n = y_true.shape[0]
            return 2 * (y_pred - y_true) / n

        def get_loss_type(self):
            return LossFunction.MEAN_SQUARED_ERROR

    # TODO: Cek lagi kedua class di bawah
    class BinaryCrossEntropy:
        def calculate(self, y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        def derivative(self, y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            n = y_true.shape[0]
            return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n

        def get_loss_type(self):
            return LossFunction.BINARY_CROSS_ENTROPY

    class CategoricalCrossEntropy:
        def calculate(self, y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1.0)
            n_samples = y_pred.shape[0]

            # Handle both one-hot encoded and class indices formats
            if y_true.ndim == 1:
                # Class indices format
                return -np.sum(np.log(y_pred[np.arange(n_samples), y_true])) / n_samples
            elif y_true.ndim == 2:
                # One-hot encoded format
                return -np.sum(y_true * np.log(y_pred)) / n_samples
            else:
                raise ValueError(
                    "y_true must be either 1D (class indices) or 2D (one-hot encoded)"
                )

        def derivative(self, y_true, y_pred):
            n_samples = y_pred.shape[0]
            if y_true.ndim == 1:
                # Convert class indices to one-hot for derivative calculation
                y_true_onehot = np.zeros_like(y_pred)
                y_true_onehot[np.arange(n_samples), y_true] = 1
                return (y_pred - y_true_onehot) / n_samples
            elif y_true.ndim == 2:
                return (y_pred - y_true) / n_samples
            else:
                raise ValueError(
                    "y_true must be either 1D (class indices) or 2D (one-hot encoded)"
                )

        def get_loss_type(self):
            return LossFunction.CATEGORICAL_CROSS_ENTROPY
