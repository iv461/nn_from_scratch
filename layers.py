import time
import numpy as np
import math

from typing import Union, Tuple, Dict

import autograd
from autograd import Tensor, draw_computation_graph


class Parameter(Tensor):
    """
    A type just to be able to detect what is a param.
    A Parameter is trainable
    """

    def __init__(self, name, value: np.ndarray) -> None:
        super().__init__(value, name)

    def __str__(self) -> str:
        return f"Param {self.name}: {super().__str__()}"


class Module:

    random_gen = np.random.default_rng(seed=123456)

    def __init__(self, name):
        self.name = name

    def get_parameters(self):
        return {v.id: v for k, v in self.__dict__.items() if isinstance(v, Parameter)}

    def forward(self, x: Tensor):
        ...

    def uniform_initializer(self, low_bound, upper_bound, shape: tuple):
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

        self.wheight = [Tensor(np.array(init_vals[i]),
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
    Linear (also called Dense) layer, calculates xA^T + b
    """

    def __init__(self, in_features: Union[tuple, int], out_features: Union[tuple, int]):
        super().__init__("Linear")

        self.in_features = in_features
        self.out_features = out_features
        # Initialization as per https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        k = 1. / in_features
        k_sqrt = math.sqrt(k)

        if out_features == 1:
            wheights_tensor_shape = (in_features,)
        elif in_features == 1:
            wheights_tensor_shape = (out_features,)
        else:
            wheights_tensor_shape = (out_features, in_features)
        bias_tensor_shape = out_features
        self.wheight = Parameter("w", self.uniform_initializer(
            -k_sqrt, k_sqrt, wheights_tensor_shape))
        self.bias = Parameter("b", self.uniform_initializer(-k_sqrt,
                                                            k_sqrt, bias_tensor_shape))

    def forward(self, x: Tensor):
        return self.wheight * x + self.bias


class ReLu(Module):

    def __init__(self):
        super().__init__("ReLu")

    def forward(self, x: Tensor):
        return autograd.max(x, 0.)


class Tanh(Module):

    def __init__(self):
        super().__init__("Tanh")

    def forward(self, x: Tensor):
        return autograd.tanh(x)


class Sequential(Module):

    def __init__(self, layers: list):
        super().__init__("Sequential")
        self.layers = layers

    def get_parameters(self):
        return {param_name: param for layer in self.layers for param_name, param in layer.get_parameters().items()}

    def forward(self, x: Tensor):
        result = x
        for layer in self.layers:
            result = layer.forward(result)
        return result


def test_perceptron():
    in_dim = 3
    p = Perceptron(in_features=in_dim)

    print(f"Perceptron params: {p.wheight}, {p.bias}")

    x = np.arange(in_dim)
    # We create a list of tensors to test the scalar case, normally we wouldn't do this
    x_t = [Tensor(np.array(x_i), f"x_{i}", is_variable=False)
           for i, x_i in enumerate(x)]
    res = p.forward(x_t)

    res.backward()

    draw_computation_graph(res, 2.)


def test_linear():

    in_dim = 3
    l = Linear(in_features=in_dim, out_features=2)

    print(f"Linear layer: w: {l.wheight}, b: {l.bias}")
    params = l.get_parameters()
    print(f"params: {params}")

    x_t = Tensor(np.arange(in_dim), "x", is_variable=False)

    res = l.forward(x_t)

    print(f"Result: {res}")
    gradients = res.backward()
    print(f"Gradients: {gradients}")
    draw_computation_graph(res, 2.)


def test_sequential_model():

    in_dim = 28*28
    out_dim = 10
    intemediate_dim = 512
    nn_model = Sequential([
        Linear(in_dim, intemediate_dim),
        ReLu(),
        Linear(intemediate_dim, intemediate_dim),
        ReLu(),
        Linear(intemediate_dim, out_dim)
    ])
    nn_params = nn_model.get_parameters()
    print(f"NN params: {nn_params}")

    x_t = Tensor(np.arange(in_dim), "x", is_variable=False)

    start_ = time.perf_counter()
    res = nn_model.forward(x_t)

    end_ = time.perf_counter()
    print(f"Forward took {(end_ - start_) * 1000.}ms")

    print(f"Result: {res.value}")

    start_ = time.perf_counter()
    res.backward()
    end_ = time.perf_counter()
    print(f"Backward took {(end_ - start_) * 1000.}ms")
    draw_computation_graph(res, 2.)


if __name__ == "__main__":
    # test_perceptron()
    # test_linear()
    test_sequential_model()
