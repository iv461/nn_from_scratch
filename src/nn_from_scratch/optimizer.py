import numpy as np
from typing import Dict
from nn_from_scratch.autograd import Tensor
# TODO take Tensor as argument ?


class Optimizer:

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
    Generic gradient descent, named in torch SGD
    """

    def __init__(self, params: Dict[str, Tensor], lr: float):
        """
        Construct gradient descent
        Args:
            params (Dict[str, Tensor]): parameters dict
            lr (float): the learning rate
        """
        super().__init__(params)
        self.lr = lr

    def step(self, trace=False):
        for _, param in self.params.items():
            grad = param.grad
            grad_clip_val = 20000
            # cannot use out= as return arrays must be of ArrayType
            grad = np.clip(grad, a_min=-grad_clip_val, a_max=grad_clip_val)
            if trace:
                print(f"[Optimizer] Grad is:\n{grad}")
                print(f"[Optimizer] Old Parameters are:\n{param.value}")

            # Subtracts in-place
            np.subtract(param.value, grad * self.lr, out=param.value)
            if trace:
                print(f"[Optimizer] New Parameters are:\n{param.value}")
