
from autograd import Tensor, draw_computation_graph
import autograd
import numpy as np


def mse_loss(y_true: Tensor, y_pred: Tensor):
    assert y_true.value.shape == y_pred.value.shape
    # assert that we have an array of at least scalars
    assert y_true.value.ndim >= 1
    return autograd.sum(autograd.mean(autograd.square(y_pred - y_true)))


def test_mse_loss():
    y_true1 = Tensor(
        np.array([[0., 1.], [0., 0.]]), "y_true1")
    y_pred1 = Tensor(
        np.array([[1., 1.], [1., 0.]]), "y_pred1")

    y_true2 = Tensor(
        np.array([[0., 10.], [0., 0.]]), "y_true2")
    y_pred2 = Tensor(
        np.array([[10., 10.], [10., 0.]]), "y_pred2")

    mse = mse_loss

    loss = mse(y_true1, y_pred1)

    loss.backward()
    draw_computation_graph(loss)

    print(
        f"mse between y_true1 and y_pred2  with own: {loss}")

    return
    print(
        f"mse between y_true1 and y_pred2  with own: {mse(y_true2, y_pred2)}")


if __name__ == "__main__":
    test_mse_loss()
