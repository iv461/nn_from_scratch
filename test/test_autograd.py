from typing import List
import numpy as np
import torch
from nn_from_scratch import autograd, layers

np.random.default_rng(seed=587346287)


def compare_ag_with_torch(test_data: List[List[np.ndarray]], f_torch, f_ag, check_grad=True, use_single_precision=True):
    """
    Compares the outputs of PyTorch and nn_from_scratch's for equality with an assert.

    Args:
        data (List[List[np.ndarray]]): A list of different inputs to the function to test for. We need a list in case the function is multivariate.
        f_torch (_type_): The nn_from_scratch function to call 
        f_ag (_type_): The PyTorch function to call
        check_grad (bool, optional): If true, the backward pass will be executed and the gradients checked for equality as well. Defaults to True.
        use_single_precision (bool, optional): If true, the passed numpy-arrays will be converted to single-precision float. Defaults to True.
    """
    inputs_ts = []
    outputs = []
    for inputs in test_data:
        if use_single_precision:
            inputs_conv = [x.astype(np.float32) for x in inputs]
        else:
            inputs_conv = inputs
        torch_x = [torch.tensor(
            x, device="cpu", requires_grad=check_grad)for x in inputs_conv]
        ag_x = [autograd.Tensor(x, requires_grad=check_grad)
                for x in inputs_conv]
        torch_y = f_torch(*torch_x)
        ag_y = f_ag(*ag_x)
        inputs_ts.append((torch_x, ag_x))
        outputs.append((torch_y, ag_y))
        if check_grad:
            torch_y.backward()
            ag_y.backward()
    assert all([np.allclose(p_ag.value, p_torch.detach().numpy())
               for p_torch, p_ag in outputs])
    assert all([np.allclose(input_torch.grad.numpy(), input_ag.grad)
                for inputs_torch, inputs_ag in inputs_ts for input_torch, input_ag in zip(inputs_torch, inputs_ag)])


def matrix_multiplication(A, x, b):
    r = A @ x
    r *= b
    return r


def two_layer_nn(w1, b1, w2, b2, w3, b3, x):
    res = w1 @ x + b1
    res = w2 @ res + b2
    res = w3 @ res + b3
    return res


def matrix_vector_multiplication_test():
    A = np.arange(12).reshape(3, 4)
    x = np.arange(4)
    m = 3 * np.arange(3)

    def f_torch(*args): return torch.sum(matrix_multiplication(*args))
    def f_ag(*args): return autograd.sum(matrix_multiplication(*args))

    compare_ag_with_torch([[A, x, m]], f_torch, f_ag, check_grad=True)


def matrix_multiplication_test():
    m = 2
    n = 3
    p = 4
    A = np.random.rand(m, n)
    B = np.random.rand(n, p)
    M = np.random.rand(m, p)

    print(f"A:\n{A}\nB:{B}\nM:\n{M}")

    def f_torch(*args): return torch.sum(matrix_multiplication(*args))
    def f_ag(*args): return autograd.sum(matrix_multiplication(*args))
    compare_ag_with_torch([[A, B, M]], f_torch, f_ag, check_grad=True)


def nn_test():
    intermediate_feat = 20
    x = np.random.rand(1).astype(np.float32)

    w1 = np.random.rand(intermediate_feat, 1)
    b1 = np.random.rand(intermediate_feat)
    w2 = np.random.rand(intermediate_feat,
                        intermediate_feat)
    b2 = np.random.rand(intermediate_feat)
    w3 = np.random.rand(1, intermediate_feat)
    b3 = np.random.rand(1)

    compare_ag_with_torch([[w1, b1, w2, b2, w3, b3, x]],
                          two_layer_nn, two_layer_nn, check_grad=True)


def relu_test():
    x = np.random.rand(1)
    x2 = np.random.rand(10, 1)
    relu = layers.ReLu()
    relu_torch = torch.nn.ReLU()
    # Sum is needed to have scalar-output for backward
    def f_ag(x): return autograd.sum(relu.forward(x))
    def f_torch(x): return torch.sum(relu_torch.forward(x))
    compare_ag_with_torch([[x], [x2]], f_torch, f_ag, check_grad=True)


matrix_vector_multiplication_test()
matrix_multiplication_test()
nn_test()
relu_test()
