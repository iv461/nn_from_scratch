
from nn_from_scratch.autograd import Tensor
from nn_from_scratch.autograd import sum, mean, square


def mse_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculates the mean-squared loss in the same way as pytorch

    Args:
        y_true (Tensor): 
        y_pred (Tensor): 

    Returns:
        Tensor: 
    """
    assert y_true.value.shape == y_pred.value.shape, "y_true an y_pred have to be of the same shape"
    assert y_true.value.ndim >= 1, "y has to be an array of at least scalars"
    return sum(mean(square(y_pred - y_true)))
