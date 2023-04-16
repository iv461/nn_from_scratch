
from nn_from_scratch.autograd import Tensor
from nn_from_scratch.autograd import sum, mean, square


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """
    Calculates the mean-squared loss in the same way as pytorch
    TODO document further.
    Args:
        input (Tensor): _description_
        target (Tensor): _description_

    Returns:
        Tensor: _description_
    """

    assert input.value.shape == target.value.shape, "input an target have to be of the same shape"
    assert input.value.ndim >= 1, "y has to be an array of at least scalars"
    return sum(mean(square(target - input)))
