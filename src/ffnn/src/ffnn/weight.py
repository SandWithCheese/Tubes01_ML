import numpy as np
from ffnn.types import WeightsSetup, WeightInitializer


class WeightInitialization:
    @staticmethod
    def get_weight_initializer_from_type(weight_setup: WeightsSetup):
        initializer = weight_setup.initializer

        if initializer == WeightInitializer.ZERO:
            return WeightInitialization.Zero()
        elif initializer == WeightInitializer.UNIFORM:
            return WeightInitialization.Uniform(
                weight_setup.lower_bound, weight_setup.upper_bound, weight_setup.seed
            )
        elif initializer == WeightInitializer.NORMAL:
            return WeightInitialization.Normal(
                weight_setup.mean, weight_setup.variance, weight_setup.seed
            )
        elif initializer == WeightInitializer.XAVIER:
            return WeightInitialization.Xavier(weight_setup.seed)
        elif initializer == WeightInitializer.HE:
            return WeightInitialization.He(weight_setup.seed)

    class Zero:
        def __init__(self):
            pass

        def initialize(
            self,
            n_inputs: int,
            n_outputs: int,
        ):
            return np.zeros((n_inputs, n_outputs))

        def get_initializer_type(self):
            return WeightInitializer.ZERO

    class Uniform:
        def __init__(
            self, lower_bound: int = -1, upper_bound: int = 1, seed: int | None = None
        ):
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.seed = seed

        def initialize(self, n_inputs: int, n_outputs: int):
            np.random.seed(self.seed)
            return np.random.uniform(
                self.lower_bound, self.upper_bound, (n_inputs, n_outputs)
            )

        def get_initializer_type(self):
            return WeightInitializer.UNIFORM

    class Normal:
        def __init__(self, mean: int = 0, variance: int = 1, seed: int | None = None):
            self.mean = mean
            self.variance = variance
            self.seed = seed

        def initialize(self, n_inputs: int, n_outputs: int):
            np.random.seed(self.seed)
            return np.random.normal(self.mean, self.variance, (n_inputs, n_outputs))

        def get_initializer_type(self):
            return WeightInitializer.NORMAL

    class Xavier:
        def __init__(self, seed: int | None = None):
            self.seed = seed

        def initialize(self, n_inputs: int, n_outputs: int):
            np.random.seed(self.seed)
            return np.random.normal(
                0, np.sqrt(2 / (n_inputs + n_outputs)), (n_inputs, n_outputs)
            )

        def get_initializer_type(self):
            return WeightInitializer.XAVIER

    class He:
        def __init__(self, seed: int | None = None):
            self.seed = seed

        def initialize(self, n_inputs: int, n_outputs: int):
            np.random.seed(self.seed)
            return np.random.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_outputs))

        def get_initializer_type(self):
            return WeightInitializer.HE
