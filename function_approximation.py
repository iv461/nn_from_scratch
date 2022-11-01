import numpy as np
import math

from layers import Linear, ReLu, Sequential, Tensor
from autograd import Node, draw_computation_graph, square
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

    def zero_grad(self):
        """
        Clear parameter gradients 
        """
        for name, param in self.params.items():
            param.grad = None

    def step(self):
        # TODO get grad from self.params.grad after refactoring
        for name, param in self.params.items():
            grad = param.grad
            print(f"Grad for param {name} is {grad}")
            np.subtract(param.value, grad * self.lr, out=param.value)


def train():

    interval = [-5, 5.]
    x_vals = np.linspace(*interval, num=10)
    y_vals = np.vectorize(f)(x_vals)

    intermediate_feat = 5
    model = Sequential([
        Linear(in_features=1, out_features=intermediate_feat),
        ReLu(),
        Linear(in_features=intermediate_feat, out_features=1)
    ])

    # Convert the train vector of from shape (N,) to (N, 1), this is the correct batch shape
    x_train = [Tensor(np.array(x_i).reshape(1), f"x_{i}",
                      is_variable=False) for i, x_i in enumerate(x_vals)]
    # reshape needed for check in local grad if both are scalar
    y_train = [Tensor(np.array(y_i).reshape(1), f"y_{i}",
                      is_variable=False) for i, y_i in enumerate(y_vals)]

    print(f"x_train: {x_train}, y_train: {y_train}")

    params = model.get_parameters()
    print(f"Params are: {params}")
    optimizer = GradientDescent(params, lr=0.001)
    loss = mse_loss

    # TODO workaround, fix properly
    id_counts = None

    def forward_x_train():
        y_pred = []
        for x_i, y_true in zip(x_train, y_train):
            y_pred.append(model.forward(x_i))
        return y_pred

    plt.plot(x_vals, y_vals)
    y_pred = forward_x_train()
    plt.plot(x_vals, [t.value for t in y_pred])
    plt.title("Before training, function")
    plt.legend()
    plt.xlim(tuple(interval))
    plt.show()

    for i in range(1000):
        # TODO workaround, fix properly
        if id_counts is not None:
            Node.id_cnt, Tensor.id_cnt = id_counts

        loss = None
        for x_i, y_true in zip(x_train, y_train):
            y_pred = model.forward(x_i)
            residual = (y_pred - y_true)
            residual = square(residual)
            # Create first a Tensor object
            if loss is None:
                loss = residual
            else:
                loss += residual

            draw_computation_graph(loss)
        if id_counts is None:
            id_counts = (Tensor.id_cnt, Node.id_cnt)
        #loss = mse_loss(y_train, y_pred)

        print(f"Loss is: {loss}")

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    plt.plot(x_vals, y_vals)
    y_pred = model.forward(x_vals)
    plt.plot(x_vals, y_pred)
    plt.xlim(tuple(interval))


train()
