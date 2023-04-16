from typing import List, Callable, Any, Tuple, Optional
import time
from nn_from_scratch.autograd import Node
from nn_from_scratch.graph_drawing import build_and_draw_computation_graph
from nn_from_scratch.optimizer import GradientDescent
from nn_from_scratch.layers import Tensor, Module
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import yappi
from dataclasses import dataclass, field

matplotlib.use("Qt5Agg")
# Raise error on numeric error like NaN, infinite etc.
np.seterr(all="raise")
random_gen = np.random.default_rng(seed=83754283547)


def create_training_data(function_to_approximate: Callable[[Any], Any], interval: Tuple[float, float], sample_size: int):
    x_values = np.linspace(*interval, num=sample_size)
    y_values = np.vectorize(function_to_approximate)(x_values)
    return x_values.reshape(sample_size, 1, 1), y_values.reshape(sample_size, 1, 1)


def batcher(x: Tensor, y: Tensor, batch_size: int, shuffle=True):
    assert len(x.value) == len(y.value)
    for batch_i in range(len(x.value) // batch_size):
        x_batch = x.value[batch_i*batch_size: (batch_i+1)*batch_size]
        y_batch = y.value[batch_i*batch_size: (batch_i+1)*batch_size]

        if shuffle:
            random_indices = np.random.permutation(batch_size)
            x_batch = np.take(x_batch, random_indices, axis=0)
            y_batch = np.take(y_batch, random_indices, axis=0)
        x_train = Tensor(x_batch, f"x_b{batch_i}")
        y_train = Tensor(y_batch, f"x_b{batch_i}")
        yield x_train, y_train


def plot_model_vs_function(model, x_t: Tensor, y_t: Tensor, interval: Tuple[float, float]):
    x_sq, y_sq = np.squeeze(x_t.value), np.squeeze(y_t.value)
    plt.plot(x_sq, y_sq, label="function")
    y_pred = model.forward(x_t)
    y_pred_sq = np.squeeze(y_pred.value)
    plt.plot(x_sq, y_pred_sq, label="model")
    plt.title("Function vs model")
    plt.legend()
    plt.xlim(interval)
    plt.show()


def plot_loss(loss_values: List[float]):
    plt.plot(np.arange(len(loss_values)), loss_values)
    plt.title("Loss over iterations")
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.show()


@dataclass
class Trainer:
    dataset: Tuple[Tensor, Tensor]  # x and y values
    model: Module
    loss_function: Callable[[Tensor, Tensor], Tensor]
    lr: float
    batch_size: int
    number_of_epochs: int
    in_between_epochs: Optional[Callable[[], Any]]
    profile: bool = False
    loss_values: List[float] = field(default_factory=list)
    plot_computation_graph: bool = False

    def fit(self, profile=False):
        params = self.model.get_parameters()
        params_str = ', '.join(map(str, params.values()))
        print(f"Params are:\n{params_str}")
        optimizer = GradientDescent(params, lr=self.lr)

        # TODO workaround, fix properly
        id_counts = None
        print(f"Starting training...")

        for epoch_i in range(self.number_of_epochs):
            if self.profile:
                yappi.start()
            for batch_i, (x_train, y_train) in enumerate(list(batcher(*self.dataset, self.batch_size))):
                # TODO ID workaround, fix properly
                if id_counts is not None:
                    Node.id_cnt, Tensor.id_cnt = id_counts

                loss = None
                start_fw = time.perf_counter()

                y_pred = self.model.forward(x_train)
                loss = self.loss_function(y_pred, y_train)

                end_fw = time.perf_counter()
                if self.profile:
                    print(f"Forward pass time: {(end_fw-start_fw)*1000.}ms")

                # TODO ID workaround, fix properly
                if id_counts is None:
                    id_counts = (Tensor.id_cnt, Node.id_cnt)

                optimizer.zero_grad()
                start_bw = time.perf_counter()
                loss.backward()
                end_bw = time.perf_counter()
                if self.profile:
                    print(f"Backprop time: {(end_bw-start_bw)*1000.}ms")

                if self.plot_computation_graph:
                    build_and_draw_computation_graph(loss)

                self.loss_values.append(loss.value)
                optimizer.step(trace=False)

            if self.profile:
                yappi.get_func_stats().print_all()
                yappi.get_thread_stats().print_all()
                input()
            print(
                f"Epoch #{epoch_i} loss is: {float(self.loss_values[-1]):.4f}")

            if epoch_i > 0 and (epoch_i % 300) == 0 and self.in_between_epochs is not None:
                self.in_between_epochs()
