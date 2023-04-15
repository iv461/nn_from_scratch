import time
from nn_from_scratch.autograd import Node, square
from nn_from_scratch.optimizer import GradientDescent
from nn_from_scratch.layers import Linear, ReLu, Sequential, Tensor
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable, Any, Tuple
import random

matplotlib.use("Qt5Agg")
# Raise error on numeric error like NaN, infinite etc.
np.seterr(all="raise")
random_gen = np.random.default_rng(seed=83754283547)
random.seed(3475346502095)


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
                          requires_grad=False) for i, (x_i, y_i) in enumerate(zipped[batch_i*batch_size: (batch_i+1)*batch_size])]
        # reshape needed for check in local grad if both are scalar
        y_train = [Tensor(np.array(y_i).reshape(1), f"y_{i}",
                          requires_grad=False) for i, (x_i, y_i) in enumerate(zipped[batch_i*batch_size: (batch_i+1)*batch_size])]
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
    plt.ylabel("MSE-loss")
    plt.xlabel("Iterations")
    plt.show()


class Trainer:

    def __init__(self, model, loss_function, lr) -> None:
        self.model = model
        self.loss_function = loss_function
        self.lr = lr

    def fit(self):
        params = self.model.get_parameters()
        params_str = ', '.join(map(str, params.values()))
        print(f"Params are:\n{params_str}")
        optimizer = GradientDescent(params, lr=self.lr)
        loss = self.loss_function

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
                    y_pred = self.model.forward(x_i)
                    residual = (y_pred - y_true)
                    residual = square(residual)
                    # Create first a Tensor object
                    if loss is None:
                        loss = residual
                    else:
                        loss += residual

                end_fw = time.perf_counter()
                print(f"Fw time: {(end_fw-start_fw)*1000.}ms")

                loss = loss * Tensor(np.array(1./batch_size),
                                     None, False, None)

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
