import numpy as np
from nn_from_scratch.autograd import Tensor
from nn_from_scratch.losses import mse_loss
import torch


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
    torch_mse_loss = torch.nn.MSELoss()

    print(
        f"mse between y_true1 and y_pred2  with own: {loss}")

    return
    print(
        f"mse between y_true1 and y_pred2  with own: {mse(y_true2, y_pred2)}")


test_mse_loss()
