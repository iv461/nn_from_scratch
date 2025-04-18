import numpy as np
from typing import Dict
from nn_from_scratch.autograd import Tensor


class Optimizer:
    """
    A base class for all optimizers.
    """
    def __init__(self, params: Dict[str, Tensor]) -> None:
        """
        Initializes the optimizer by storing the parameters.
        Args:
            params (Dict[str, Tensor]): a dict of named parameter tensors.
        """
        self.params = params

    def zero_grad(self):
        """
        Clear parameter gradients, sets all to None.
        """
        for name, param in self.params.items():
            param.grad = None


class GradientDescent(Optimizer):
    """
    The regular gradient descent optimization algorithm, called "SGD" in PyTorch.
    """
    def __init__(self, params: Dict[str, Tensor], lr: float, clip_grad=False):
        """
        Construct gradient descent
        Args:
            params (Dict[str, Tensor]): The parameters of the model to optimize
            lr (float): the learning rate
            clip_grad (bool): Whether to clip the gradients
        """
        super().__init__(params)
        self.lr = lr
        self.clip_grad = clip_grad

    def step(self, trace=False):
        """
        Step the optimizer. The gradients already have been computed.
        """
        for _, param in self.params.items():
            # Retrieve the gradients
            grad = param.grad
            if grad is None:
                raise Exception("You first have to call backward")
            if self.clip_grad:
                grad_clip_val = 20000
                # cannot use out= as return arrays must be of ArrayType
                grad = np.clip(grad, a_min=-grad_clip_val, a_max=grad_clip_val)
            if trace:
                print(f"[Optimizer] Grad is:\n{grad}")
                print(f"[Optimizer] Old Parameters are:\n{param.value}")

            # Gradient descent: param = param - grad * learning_rate
            np.subtract(param.value, grad * self.lr, out=param.value)
            if trace:
                print(f"[Optimizer] New Parameters are:\n{param.value}")
