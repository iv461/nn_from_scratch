
import numpy as np
import math
import nn_from_scratch.autograd as autograd
from nn_from_scratch.autograd import Tensor
from typing import List, Union, Dict


class Parameter(Tensor):
    """
    A type just to be able to detect what is a param.
    A Parameter is trainable

    TODO remove this class, detect parameters by requires_grad
    """

    def __init__(self, name, value: np.ndarray) -> None:
        super().__init__(value, name, requires_grad=True)

    def __str__(self) -> str:
        return f"Param {self.name}: {super().__str__()}"


class Module:
    """
    Base class for layers
    """

    random_gen = np.random.default_rng(seed=123456)

    def __init__(self, name):
        self.name = name

    def named_parameters(self) -> Dict[int, Parameter]:
        return {v.id: v for k, v in self.__dict__.items() if isinstance(v, Parameter)}

    def forward(self, x: Tensor) -> Tensor:
        ...

    def uniform_initializer(self, low_bound: float, upper_bound: float, shape: Union[int, tuple]):
        if isinstance(shape, tuple):
            number_of_samples = np.product(list(shape))
        else:
            number_of_samples = shape
        samples = self.random_gen.uniform(
            low_bound, upper_bound, number_of_samples)
        if isinstance(shape, tuple):
            samples = samples.reshape(*shape)
        return samples


class Perceptron(Module):
    """
    Simple perceptron to test the scalar case of the computation graph with autograd of the Tensor
    """

    def __init__(self, in_features):
        super().__init__("Perceptron")
        self.in_features = in_features

        k = 1. / in_features
        k_sqrt = math.sqrt(k)
        init_vals = self.uniform_initializer(
            -k_sqrt, k_sqrt, in_features+1)

        self.wheight = [Parameter(np.array(init_vals[i]),
                                  name=f"w{i}") for i in range(self.in_features)]

        self.bias = Tensor(np.array(init_vals[-1]), name="b")

    def forward(self, x):
        assert len(x) == self.in_features
        dot_prod = x[0] * self.wheight[0]
        for i, (x_i, w_i) in enumerate(zip(x, self.wheight)):
            if i == 0:
                continue
            dot_prod += x_i * w_i
        return dot_prod + self.bias


class Linear(Module):
    """
    Linear (also called Dense) layer, calculates wx + b
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Creates the layer and initializes the weights.

        Args:
            in_features (int): the number of input features
            out_features (int): the number of output features
        """
        super().__init__("Linear")

        self.in_features = in_features
        self.out_features = out_features
        # Initialization as per https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        k = 1. / in_features
        k_sqrt = math.sqrt(k)

        weights_tensor_shape = (out_features, in_features)
        self.weight = Parameter("w", self.uniform_initializer(
            -k_sqrt, k_sqrt, weights_tensor_shape))
        # We need to create a 2d array for the bias vector (a column vector) such that the broadcasting works correctly for batch-sized inputs
        self.bias = Parameter("b", self.uniform_initializer(-k_sqrt,
                                                            k_sqrt, (out_features, 1)))

    def forward(self, x: Tensor):
        return self.weight @ x + self.bias


class ReLU(Module):
    """
    The ReLU activation function.
    """
    def __init__(self):
        super().__init__("ReLu")

    def forward(self, x: Tensor):
        result = np.maximum(x.value, 0)
        return Tensor(result, None, True, [x, Tensor(np.array(0., dtype=x.value.dtype))], "max")


class Tanh(Module):
    """
    The Tanh activation functions.
    """
    def __init__(self):
        super().__init__("Tanh")

    def forward(self, x: Tensor):
        return autograd.tanh(x)


class Sequential(Module):
    """
    Module for sequential execution of multiple layers.
    """

    def __init__(self, layers: List[Module]):
        super().__init__("Sequential")
        self.layers = layers

    def named_parameters(self):
        return {param_name: param for layer in self.layers for param_name, param in layer.named_parameters().items()}

    def forward(self, x: Tensor):
        result = x
        for layer in self.layers:
            result = layer.forward(result)
        return result
