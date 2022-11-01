import numpy as np
import math

from layers import Linear, ReLu, Sequential, Tensor
from autograd import Node
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
        def update_params(param, grad):
            np.subtract(param, grad * self.lr, out=param)
        for name, param in self.params.items():
            update_params(param, gradient[name])


def train():

    interval = [-5, 5.]
    x_vals = np.linspace(*interval, num=100)
    y_vals = np.vectorize(f)(x_vals)
    plt.plot(x_vals, y_vals)
    plt.xlim(tuple(interval))

    intermediate_feat = 5
    model = Sequential([
        Linear(in_features=1, out_features=intermediate_feat),
        ReLu(),
        Linear(in_features=intermediate_feat, out_features=1)
    ])

    x_train = Tensor(x_vals, "x", is_variable=False)
    y_train = Tensor(y_vals, "y_true", is_variable=False)

    opt = GradientDescent(model.get_parameters(), lr=0.01)
    loss = mse_loss

    # TODO workaround, fix properly
    model.forward(x_train)
    id_count_Tensor_init = Tensor.id_cnt
    id_count_Node_init = Node.id_cnt

    steps = []
    for _ in range(1000):

        # TODO workaround, fix properly
        Node.id_cnt = id_count_Node_init
        Tensor.id_cnt = id_count_Tensor_init

        y_pred = model.forward(x_train)

        loss = mse_loss(y_train, y_pred)

        print(f"Loss is: {loss}")

        loss.backward()
        opt.step()

    plt.plot(x_vals, y_vals)
    y_pred = model.forward(x_vals)
    plt.plot(x_vals, y_pred)
    plt.xlim(tuple(interval))


train()
