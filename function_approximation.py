import numpy as np
import math

from layers import Linear, ReLu, Sequential, Tensor
from losses import mse_loss

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")


def f(x):
    """
    The function to approximate
    """
    return np.sin(x) + .3 * np.exp(x)


class GradientDescent:
    """
    """

    def __init__(self, params: np.ndarray, lr: float):
        self.params = params
        self.lr = lr

    def step(self, gradient):
        # TODO get grad from self.params.grad after refactoring
        return np.subtract(self.params, gradient * self.lr, out=self.params)


def train():

    interval = [-5, 5.]
    x_vals = np.linspace(*interval, num=100)
    plt.plot(x_vals, np.vectorize(f)(x_vals))
    plt.xlim(tuple(interval))

    model = Sequential(Linear(in_features=1, out=5), ReLu(),
                       Linear(in_features=5, out_features=1))

    x_v = 4.
    steps = []
    opt = GradientDescent()
    loss = mse_loss
    for i in range(1000):

        steps.append(x_v)
        new_x = opt.update(x_v, f, .002)

        if abs(new_x - x_v) < 0.001:
            break
        x_v = new_x

    print(f"Minimum is at: {x_v}")
    plt.scatter(steps, vectorize(f)(steps), color="g")

    plt.show()
