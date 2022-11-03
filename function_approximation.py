import numpy as np
import math

from layers import Linear, ReLu, Sequential, Tensor, Tanh
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


class Optimizer:

    def __init__(self, params: np.ndarray) -> None:
        self.params = params

    def zero_grad(self):
        """
        Clear parameter gradients 
        """
        for name, param in self.params.items():
            param.grad = None


class GradientDescent(Optimizer):
    """
    """

    def __init__(self, params: np.ndarray, lr: float):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for name, param in self.params.items():
            grad = param.grad
            #print(f"Grad for param {name} is {grad}")
            np.subtract(param.value, grad * self.lr, out=param.value)


class Momentum(Optimizer):

    def __init__(self, params: np.ndarray, lr: float, momentum: float):

        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.b_t = None

    def step(self):
        """
        https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        """

        for name, param in self.params.items():
            grad = param.grad
            if self.b_t is None:
                self.b_t = grad
            else:
                self.momentum
            #print(f"Grad for param {name} is {grad}")
            np.subtract(param.value, grad * self.lr, out=param.value)


def train():

    interval = [-6, 4.5]

    batch_size = 10
    sample_size = 30 * batch_size
    x_vals_orig = np.linspace(*interval, num=sample_size)

    random_gen = np.random.default_rng(seed=123456)
    x_vals = x_vals_orig.copy()
    random_gen.shuffle(x_vals)
    f_v = np.vectorize(f)
    y_vals_orig = f_v(x_vals_orig)
    y_vals = f_v(x_vals)

    intermediate_feat = 6
    seq_model = Sequential([
        Linear(in_features=1, out_features=intermediate_feat),
        Tanh(),
        Linear(in_features=intermediate_feat, out_features=1),
    ])

    linear_model = Linear(in_features=1, out_features=1)
    lin_relu = Sequential([Linear(in_features=1, out_features=1), ReLu()])
    model = seq_model

    #print(f"x_train: {x_train}, y_train: {y_train}")

    params = model.get_parameters()
    print(f"Params are: {params}")
    optimizer = GradientDescent(params, lr=0.001)
    loss = mse_loss

    # TODO workaround, fix properly
    id_counts = None

    def forward_x_train(x_train):
        y_pred = []
        for x_i in x_train:
            y_pred.append(model.forward(x_i))
        return y_pred

    def plot_model_vs_function(x_t: list[Tensor], y_t: list[Tensor]):
        x_scalars = [float(t.value) for t in x_t]
        y_scalars = [float(t.value) for t in y_t]
        plt.plot(x_scalars, y_scalars)
        y_pred = forward_x_train(x_t)
        y_pred_scalar = [float(t.value) for t in y_pred]
        plt.plot(x_scalars, y_pred_scalar)
        plt.title("Function vs model")
        plt.legend()
        plt.xlim(tuple(interval))
        plt.show()

    def print_parameters():
        for param_id, param in params.items():
            print(f"Param {param.name}({param_id}) is: {param.value}")

    last_loss = None

    loss_vals = []

    x_orig_t = [Tensor(np.array(x_i).reshape(1), f"x_{i}",
                       is_variable=False) for i, x_i in enumerate(x_vals_orig)]
    # reshape needed for check in local grad if both are scalar
    y_orig_t = [Tensor(np.array(y_i).reshape(1), f"y_{i}",
                       is_variable=False) for i, y_i in enumerate(y_vals_orig)]

    plot_model_vs_function(x_orig_t, y_orig_t)
    lr_decay = .99
    for epoch_i in range(2000):
        for batch_i in range(sample_size // batch_size):
            # TODO workaround, fix properly
            if id_counts is not None:
                Node.id_cnt, Tensor.id_cnt = id_counts

            loss = None
            # Convert the train vector of from shape (N,) to (N, 1), this is the correct batch shape
            x_train = [Tensor(np.array(x_i).reshape(1), f"x_{i}",
                              is_variable=False) for i, x_i in enumerate(x_vals[batch_i*batch_size: (batch_i+1)*batch_size])]
            # reshape needed for check in local grad if both are scalar
            y_train = [Tensor(np.array(y_i).reshape(1), f"y_{i}",
                              is_variable=False) for i, y_i in enumerate(y_vals[batch_i*batch_size: (batch_i+1)*batch_size])]
            for x_i, y_true in zip(x_train, y_train):
                y_pred = model.forward(x_i)
                residual = (y_pred - y_true)
                residual = square(residual)
                # Create first a Tensor object
                if loss is None:
                    loss = residual
                else:
                    loss += residual

                """ optimizer.zero_grad()
                loss.backward()
                draw_computation_graph(loss) """

            if id_counts is None:
                id_counts = (Tensor.id_cnt, Node.id_cnt)
            #loss = mse_loss(y_train, y_pred)

            if (batch_i % 50) == 0:
                print(f"Batch #{batch_i}, epoch #{epoch_i} loss is: {loss}")

            if last_loss is not None:
                if np.abs(last_loss-loss.value) < 0.0001:
                    break

            last_loss = loss.value
            loss_vals.append(last_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print_parameters()

        optimizer.lr *= lr_decay
        if (epoch_i % 200) == 0:
            plot_model_vs_function(x_orig_t, y_orig_t)

    plot_model_vs_function(x_orig_t, y_orig_t)

    plt.plot(np.arange(len(loss_vals)), loss_vals)
    plt.title("Loss")
    plt.show()


train()
