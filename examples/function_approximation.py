
import numpy as np
from nn_from_scratch.layers import Linear, ReLU, Sequential, Tensor
from nn_from_scratch.losses import mse_loss
from training_common import create_training_data, plot_loss, plot_model_vs_function, Trainer


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
        ReLU(),
        Linear(in_features=number_of_intermediate_features,
               out_features=number_of_intermediate_features),
        ReLU(),
        Linear(in_features=number_of_intermediate_features, out_features=1),
    ])

    x_values = x_values.astype(np.float32)
    y_values = y_values.astype(np.float32)

    x_orig_t = Tensor(x_values, "x")
    y_orig_t = Tensor(y_values, "y")

    print(f"Initial model: Close window to continue training")
    plot_model_vs_function(model, x_orig_t, y_orig_t, interval)

    def plot_model():
        print(f"Current model: Close window to continue training")
        plot_model_vs_function(
            model, x_orig_t, y_orig_t, interval)

    trainer = Trainer((x_orig_t, y_orig_t), model, mse_loss,
                      1e-3, batch_size, number_of_epochs, plot_model, profile=False)

    trainer.fit()
    print(f"Finished training, final model: Close window to show loss curves")
    plot_model()
    plot_loss(trainer.loss_values)


train()
