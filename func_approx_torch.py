import numpy as np

from torch import nn, from_numpy as t_from_from_numpy
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

import matplotlib
import matplotlib.pyplot as plt

from typing import List

matplotlib.use("Qt5Agg")


class FunctionApproximationDataset(Dataset):
    """
    An implementation of a Torch dataset which can be used for function approximation.
    """

    def __init__(self, f, interval, sample_size: int):
        """
        f: A callable which maps from R -> R
        interval a list with lower and upper value
        sample_size: the number of samples 
        """
        self.x_vals = np.linspace(*interval, num=sample_size)
        self.y_vals = np.vectorize(f)(self.x_vals)

        """
        The examples have to be of N x 1 size such that the vectorization 
        of the layer works and also of float32 type as this
         is the default torch parameter type.
        """
        self.x_vals = t_from_from_numpy(
            self.x_vals.reshape(sample_size, 1).astype(np.float32))
        self.y_vals = t_from_from_numpy(
            self.y_vals.reshape(sample_size, 1).astype(np.float32))

    def __getitem__(self, index: int):
        return self.x_vals[index], self.y_vals[index]

    def __len__(self):
        return len(self.x_vals)


class NeuralNetwork(nn.Module):
    def __init__(self, intermediate_layers):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, intermediate_layers),
            nn.Tanh(),
            nn.Linear(intermediate_layers, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def f(x):
    """
    The function to approximate
    """
    return np.sin(x) + .3 * np.exp(x)


def plot_model_vs_function(model, x_t: Tensor, y_t: Tensor, x_lim):
    x_t_np = x_t.detach().numpy()
    y_t_np = y_t.detach().numpy()
    plt.plot(x_t_np, y_t_np)
    y_pred = model(x_t)
    y_pred_np = y_pred.detach().numpy()
    plt.plot(x_t_np, y_pred_np)
    plt.title("Function vs model")
    plt.legend()
    plt.xlim(tuple(x_lim))
    plt.show()


def train_loop(dataloader, model, loss_fn, optimizer):
    losses = []
    for batch_i, batch in enumerate(dataloader):

        # if torch.cuda.is_available():

        #batch = batch.cuda()

        X, y = batch
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def train():

    interval = [-6, 5.]

    intermediate_feat = 10
    model = NeuralNetwork(intermediate_layers=intermediate_feat)

    dataset = FunctionApproximationDataset(f, interval, sample_size=100)

    dataloader = DataLoader(dataset, batch_size=20,
                            shuffle=True, num_workers=0)

    params = model.parameters()

    optimizer = SGD(params, lr=.00007, momentum=0.5, dampening=.5)
    mse_loss = nn.MSELoss()

    # Move to cuda
    # if torch.cuda.is_available():
    #    pass
    #model = model.cuda()

    #plot_model_vs_function(model, dataset.x_vals, dataset.y_vals, interval)

    loss_vals = []
    epochs = 5000
    for epoch_i in range(epochs):
        loss_vals += train_loop(dataloader, model, mse_loss, optimizer)

        if (epoch_i % 1000) == 0:
            print(f"Params are: ")
            for param in list(params):
                print(f"Param: {param}")
            plot_model_vs_function(model, dataset.x_vals,
                                   dataset.y_vals, interval)

    plot_model_vs_function(model, dataset.x_vals, dataset.y_vals, interval)

    plt.plot(np.arange(len(loss_vals)), loss_vals)
    plt.title("Loss")
    plt.show()


train()
