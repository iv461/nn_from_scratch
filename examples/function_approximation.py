import os  # nopep8
import sys  # nopep8
import pathlib  # nopep8
sys.path.insert(0, pathlib.Path(__file__).parents)  # nopep8


print(f"Path is:\n{sys.path}")  # nopep8

from typing import List, Callable, Any, Tuple
import numpy as np
import random
from nn_from_scratch.layers import Linear, ReLu, Sequential, Tensor
from nn_from_scratch.autograd import Node, square
from nn_from_scratch.losses import mse_loss
from nn_from_scratch.optimizer import GradientDescent
import nn_from_scratch
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Qt5Agg")
# Raise error on numeric error like NaN, infinite etc.
np.seterr(all="raise")
random_gen = np.random.default_rng(seed=83754283547)
random.seed(3475346502095)


def f(x):
    """
    The function to approximate
    """
    return np.sin(x) + .3 * np.exp(x)


def create_training_data(function_to_approximate: Callable[[Any], Any], interval: Tuple[float, float], sample_size: int):
    x_values = np.linspace(*interval, num=sample_size)
    y_values = np.vectorize(function_to_approximate)(x_values)
    return x_values, y_values


def vectorize_model(model):
    def vectorized(x):
        y_pred = []
        for x_i in x:
            y_pred.append(model.forward(x_i))
        return y_pred
    return vectorized


def batcher(x_y_tuple: Tuple[List[Tensor], List[Tensor]], batch_size: int, shuffle=True):
    x_values, y_values = x_y_tuple
    for batch_i in range(len(x_y_tuple[0]) // batch_size):
        zipped = list(zip(x_values, y_values))
        if shuffle:
            random.shuffle(zipped)
        # Convert the train vector of from shape (N,) to (N, 1), this is the correct batch shape
        x_train = [Tensor(np.array(x_i).reshape(1), f"x_{i}",
                          is_variable=False) for i, (x_i, y_i) in enumerate(zipped[batch_i*batch_size: (batch_i+1)*batch_size])]
        # reshape needed for check in local grad if both are scalar
        y_train = [Tensor(np.array(y_i).reshape(1), f"y_{i}",
                          is_variable=False) for i, (x_i, y_i) in enumerate(zipped[batch_i*batch_size: (batch_i+1)*batch_size])]
        yield x_train, y_train


def plot_model_vs_function(vectorized_model, x_t: List[Tensor], y_t: List[Tensor], interval: Tuple[float, float]):
    x_scalars = [float(t.value) for t in x_t]
    y_scalars = [float(t.value) for t in y_t]
    plt.plot(x_scalars, y_scalars, label="function")
    y_pred = vectorized_model(x_t)
    y_pred_scalar = [float(t.value) for t in y_pred]
    plt.plot(x_scalars, y_pred_scalar, label="model")
    plt.title("Function vs model")
    plt.legend()
    plt.xlim(interval)
    plt.show()


def plot_loss(loss_values: List[float]):
    plt.plot(np.arange(len(loss_values)), loss_values)
    plt.title("Loss")
    plt.show()


def train():

    interval = (-6, 4.5)
    batch_size = 20
    sample_size = 5 * batch_size
    number_of_epochs = 2000

    x_values, y_values = create_training_data(
        f, interval=interval, sample_size=sample_size)

    number_of_intermediate_features = 30
    model = Sequential([
        Linear(in_features=1, out_features=number_of_intermediate_features),
        ReLu(),
        Linear(in_features=number_of_intermediate_features,
               out_features=number_of_intermediate_features),
        ReLu(),
        Linear(in_features=number_of_intermediate_features, out_features=1),
    ])

    params = model.get_parameters()
    params_str = ', '.join(map(str, params.values()))
    print(f"Params are:\n{params_str}")
    optimizer = GradientDescent(params, lr=1e-3)
    loss = mse_loss

    # TODO workaround, fix properly
    id_counts = None

    loss_values = []

    x_orig_t = [Tensor(np.array(x_i).reshape(1), f"x_{i}",
                       is_variable=False) for i, x_i in enumerate(x_values)]
    # reshape needed for check in local grad if both are scalar
    y_orig_t = [Tensor(np.array(y_i).reshape(1), f"y_{i}",
                       is_variable=False) for i, y_i in enumerate(y_values)]

    vectorized_model = vectorize_model(model)
    plot_model_vs_function(vectorized_model, x_orig_t, y_orig_t, interval)

    print(f"Starting training...")

    for epoch_i in range(number_of_epochs):
        for batch_i, (x_train, y_train) in enumerate(list(batcher((x_values, y_values), batch_size))):
            # TODO ID workaround, fix properly
            if id_counts is not None:
                Node.id_cnt, Tensor.id_cnt = id_counts

            # Manual adding up of the loss as the model is not vectorized
            # TODO vectorize model
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

            loss = loss * Tensor(np.array(1./batch_size), None, False, None)

            # TODO ID workaround, fix properly
            if id_counts is None:
                id_counts = (Tensor.id_cnt, Node.id_cnt)

            if (batch_i % 50) == 0:
                print(
                    f"Batch #{batch_i}, epoch #{epoch_i} loss is: {loss.value}")

            loss_values.append(loss.value)
            optimizer.zero_grad()
            loss.backward()
            # draw_computation_graph(loss)
            optimizer.step(trace=False)

        if (epoch_i % 300) == 0:
            plot_model_vs_function(
                vectorized_model, x_orig_t, y_orig_t, interval)

    plot_model_vs_function(vectorized_model, x_orig_t, y_orig_t, interval)
    plot_loss(loss_values)


train()
