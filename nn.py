import autograd.numpy as ag_np
from autograd import grad

import numpy as np
import math

from layers import Linear
from losses import mse_loss

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")


def f(x):
    return ag_np.sin(x) + .3 * ag_np.exp(x)


def vectorize(f):
    def vectorized(x_vals):
        output = []
        for x in x_vals:
            output.append(f(x))
        return output
    return vectorized


class GradientDescent:

    def update(self, x, f, lr):
        gradient = grad(f)(x)
        return x - gradient * lr


class Momentum:
    def __init__(self) -> None:
        self.moment = 0

    def update(self, x, f, lr):
        gradient = grad(f)(x)
        new_x = x - gradient * lr + .1 * self.moment
        self.moment += gradient * lr
        print(f"self.moment: {self.moment}")
        return new_x


def train():

    interval = [-5, 5.]
    x_vals = np.linspace(*interval, num=100)
    plt.plot(x_vals, vectorize(f)(x_vals))
    plt.xlim(tuple(interval))

    x_v = 4.
    steps = []
    opt = GradientDescent()
    for i in range(1000):

        steps.append(x_v)
        new_x = opt.update(x_v, f, .002)

        if abs(new_x - x_v) < 0.001:
            break
        x_v = new_x

    print(f"Minimum is at: {x_v}")
    plt.scatter(steps, vectorize(f)(steps), color="g")

    plt.show()


train()
