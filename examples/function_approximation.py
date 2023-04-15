import time
import yappi

import numpy as np
import random
from nn_from_scratch.layers import Linear, ReLu, Sequential, Tensor
from nn_from_scratch.autograd import Node, square
from nn_from_scratch.losses import mse_loss
from nn_from_scratch.optimizer import GradientDescent
import nn_from_scratch
import matplotlib
import matplotlib.pyplot as plt
from training_common import create_training_data, plot_loss, plot_model_vs_function, vectorize_model, batcher


def f(x):
    """
    The function to approximate
    """
    return np.sin(x) + .3 * np.exp(x)


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
                       requires_grad=False) for i, x_i in enumerate(x_values)]
    # reshape needed for check in local grad if both are scalar
    y_orig_t = [Tensor(np.array(y_i).reshape(1), f"y_{i}",
                       requires_grad=False) for i, y_i in enumerate(y_values)]

    vectorized_model = vectorize_model(model)
    print(f"Initial model: Close window to continue training")
    plot_model_vs_function(vectorized_model, x_orig_t, y_orig_t, interval)

    print(f"Starting training...")

    for epoch_i in range(number_of_epochs):
        yappi.start()
        for batch_i, (x_train, y_train) in enumerate(list(batcher((x_values, y_values), batch_size))):
            # TODO ID workaround, fix properly
            if id_counts is not None:
                Node.id_cnt, Tensor.id_cnt = id_counts

            # Manual adding up of the loss as the model is not vectorized
            # TODO vectorize model

            loss = None
            start_fw = time.perf_counter()
            for x_i, y_true in zip(x_train, y_train):
                y_pred = model.forward(x_i)
                residual = (y_pred - y_true)
                residual = square(residual)
                # Create first a Tensor object
                if loss is None:
                    loss = residual
                else:
                    loss += residual

            end_fw = time.perf_counter()
            print(f"Fw time: {(end_fw-start_fw)*1000.}ms")

            loss = loss * Tensor(np.array(1./batch_size), None, False, None)

            # TODO ID workaround, fix properly
            if id_counts is None:
                id_counts = (Tensor.id_cnt, Node.id_cnt)

            optimizer.zero_grad()
            start_bw = time.perf_counter()
            loss.backward()
            end_bw = time.perf_counter()
            print(f"BW time: {(end_bw-start_bw)*1000.}ms")

            loss_values.append(loss.value)
            # draw_computation_graph(loss)
            optimizer.step(trace=False)

        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()
        input()
        print(f"Epoch #{epoch_i} loss is: {float(loss_values[-1]):.4f}")

        if epoch_i > 0 and (epoch_i % 300) == 0:
            print(f"Current model: Close window to continue training")
            plot_model_vs_function(
                vectorized_model, x_orig_t, y_orig_t, interval)

    print(f"Finished training, final model: Close window to show loss curves")
    plot_model_vs_function(vectorized_model, x_orig_t, y_orig_t, interval)
    plot_loss(loss_values)


train()
