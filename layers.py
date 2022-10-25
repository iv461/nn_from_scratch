import autograd.numpy as ag_np
import numpy as np
import math

from typing import Union, Tuple, Dict, overload


class Parameter:
    """
    A type just to be able to detect what is a param. 
    A Parameter is trainable 
    """

    def __init__(self, param) -> None:
        self.param = param


class Module:

    random_gen = np.random.default_rng(seed=123456)

    def __init__(self, name):
        self.name = name

    def get_parameters(self):
        return [v.param for k, v in self.__dict__.items() if isinstance(v, Parameter)]

    def forward(self):
        ...

    def uniform_initializer(self, low_bound, upper_bound, shape: tuple):
        print(f"shape in init {shape}")
        if isinstance(shape, tuple):
            number_of_samples = np.product(list(shape))
        else:
            number_of_samples = shape
        samples = self.random_gen.uniform(
            low_bound, upper_bound, number_of_samples)
        if isinstance(shape, tuple):
            samples = samples.reshape(*shape)
        return samples


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
        else:
            wheights_tensor_shape = (in_features, out_features)
        bias_tensor_shape = out_features
        self.wheight = Parameter(self.uniform_initializer(
            -k_sqrt, k_sqrt, wheights_tensor_shape))
        self.bias = Parameter(self.uniform_initializer(-k_sqrt,
                                                       k_sqrt, bias_tensor_shape))

    def forward(self, x):
        assert x.shape[0] == self.in_features
        w = self.wheight.param
        b = self.bias.param
        print(f"{w.shape}, {x.shape}")
        return np.tensordot(x, w, axes=1) + b


class ReLu(Module):

    def __init__(self):
        super().__init__("ReLu")

    def forward(self, x):
        return np.max(x, 0)


class Sequential(Module):

    def __init__(self, layers: list):
        super().__init__("Sequential")
        self.layers = layers

    def get_parameters(self):
        return [param for layer in self.layers for param in layer.get_parameters()]

    def forward(self, x):
        result = x
        for layer in self.layers:
            result = layer.forward(result)
        return result


def test_linear():

    in_dim = 9
    l = Linear(in_features=in_dim, out_features=2)

    print(f"Linear layer: w: {l.wheight}, b: {l.bias}")
    params = l.get_parameters()
    print(f"params: {params}")

    x = np.arange(in_dim)
    res = l.forward(x)

    print(f"res: {res}")

    nn_model = Sequential([
        Linear(28*28, 512),
        ReLu(),
        Linear(512, 512),
        ReLu(),
        Linear(512, 10)
    ])
    nn_params = nn_model.get_parameters()
    print(f"NN params: {nn_params}")


if __name__ == "__main__":
    test_linear()
