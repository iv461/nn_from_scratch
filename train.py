import numpy as np

from autograd import grad
from autograd.misc.flatten import flatten

import autograd.numpy as ag_np


class SGD:

    ...


def train(training_data, model, loss, optimizer):
    ...

    # for x, y_true in training_data:


def gradient_descent(x, model, lr: float):
    g = grad(model)(x)
    new_x = x - lr * g
    return new_x
