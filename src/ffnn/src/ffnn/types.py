from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ActivationFunction(Enum):
    LINEAR = 1
    RELU = 2
    SIGMOID = 3
    TANH = 4
    SOFTMAX = 5


class LossFunction(Enum):
    MEAN_SQUARED_ERROR = 1
    BINARY_CROSS_ENTROPY = 2
    CATEGORICAL_CROSS_ENTROPY = 3


class WeightInitializer(Enum):
    ZERO = 1
    UNIFORM = 2
    NORMAL = 3


@dataclass
class WeightsSetup:
    initializer: WeightInitializer
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    mean: Optional[float] = None
    variance: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.initializer == WeightInitializer.ZERO:
            if self.lower_bound is not None or self.upper_bound is not None:
                raise ValueError(
                    "lower_bound and upper_bound should be None for ZERO initializer"
                )
            if self.mean is not None or self.variance is not None:
                raise ValueError(
                    "mean and variance should be None for ZERO initializer"
                )
        elif self.initializer == WeightInitializer.UNIFORM:
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError(
                    "lower_bound and upper_bound are required for UNIFORM initializer"
                )
        elif self.initializer == WeightInitializer.NORMAL:
            if self.mean is None or self.variance is None:
                raise ValueError(
                    "mean and variance are required for NORMAL initializer"
                )
        else:
            raise ValueError(f"Unsupported initializer: {self.initializer}")
