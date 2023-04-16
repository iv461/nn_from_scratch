from nn_from_scratch.graph_drawing import build_and_draw_computation_graph
from typing import List, Union
import numpy as np
import torch
from nn_from_scratch import autograd, layers, losses

np.random.default_rng(seed=587346287)


def assert_arrays_equal(x1, x2):
    if not np.allclose(x1, x2):
        raise Exception(f"Arrays were not equal:\n{x1}\n{x2}")


def compare_ag_with_torch(test_data: List[Union[List[np.ndarray], np.ndarray]], f_torch, f_ag, check_grad=True, use_single_precision=True, sum_result=True, plot_graph=False):
    """
    Compares the outputs of PyTorch and nn_from_scratch's for equality with an assert.

    Args:
        data (List[Union[List[np.ndarray], np.ndarray]]): A list of different inputs to the function to test for. We need a list in case the function is multivariate.
        f_torch (_type_): The nn_from_scratch function to call 
        f_ag (_type_): The PyTorch function to call
        check_grad (bool, optional): If true, the backward pass will be executed and the gradients checked for equality as well. Defaults to True.
        use_single_precision (bool, optional): If true, the passed numpy-arrays will be converted to single-precision float. Defaults to True.
        sum_result (bool, optional): If true, the result will be summed to obtain a scalar before performing the backward-pass This is useful in combination with gradient checking. Defaults to True.
    """
    inputs_ts = []
    outputs = []
    for inputs in test_data:
        if not isinstance(inputs, list):
            inputs = list(inputs)
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
        if sum_result:
            torch_y = torch.sum(torch_y)
            ag_y = autograd.sum(ag_y)
        if check_grad:
            torch_y.backward()
            if plot_graph:
                build_and_draw_computation_graph(ag_y)
            ag_y.backward()

    for p_torch, p_ag in outputs:
        assert_arrays_equal(p_ag.value, p_torch.detach().numpy())
    if check_grad:
        for inputs_torch, inputs_ag in inputs_ts:
            for input_torch, input_ag in zip(inputs_torch, inputs_ag):
                assert_arrays_equal(input_torch.grad.numpy(), input_ag.grad)


def matrix_multiplication(A, x, b):
    r = A @ x
    r *= b
    return r


def three_layer_nn(x, w1, b1, w2, b2, w3, b3):
    res = w1 @ x + b1
    res = w2 @ res + b2
    res = w3 @ res + b3
    return res


def matrix_vector_multiplication_test():
    A = np.arange(12).reshape(3, 4)
    x = np.arange(4)
    m = 3 * np.arange(3)

    compare_ag_with_torch([[A, x, m]], matrix_multiplication,
                          matrix_multiplication, check_grad=True)


def matrix_multiplication_test():
    m = 2
    n = 3
    p = 4
    A = np.random.rand(m, n)
    B = np.random.rand(n, p)
    M = np.random.rand(m, p)

    print(f"A:\n{A}\nB:{B}\nM:\n{M}")

    compare_ag_with_torch([[A, B, M]], matrix_multiplication,
                          matrix_multiplication, check_grad=True)


def nn_test():
    intermediate_feat = 20
    x = np.random.rand(1, 1)

    w1 = np.random.rand(intermediate_feat, 1)
    b1 = np.random.rand(intermediate_feat, 1)
    w2 = np.random.rand(intermediate_feat,
                        intermediate_feat)
    b2 = np.random.rand(intermediate_feat, 1)
    w3 = np.random.rand(1, intermediate_feat)
    b3 = np.random.rand(1)

    compare_ag_with_torch([[x, w1, b1, w2, b2, w3, b3]],
                          three_layer_nn, three_layer_nn, check_grad=True)


def batched_nn_test():
    intermediate_feat = 20
    batch_size = 30
    x = np.random.rand(batch_size, 10, 1)

    w1 = np.random.rand(intermediate_feat, 10)
    b1 = np.random.rand(intermediate_feat, 1)
    w2 = np.random.rand(intermediate_feat,
                        intermediate_feat)
    b2 = np.random.rand(intermediate_feat, 1)
    w3 = np.random.rand(1, intermediate_feat)
    b3 = np.random.rand(1)

    # We let also calculate the derivatives wrt. to the input to check whether we can calculate the gradient here correctly as well
    # Otherwise, we can apply partially x
    compare_ag_with_torch([[x, w1, b1, w2, b2, w3, b3]],
                          three_layer_nn, three_layer_nn, check_grad=True)


def relu_test():
    x = np.random.rand(1)
    x2 = np.random.rand(10, 1)
    relu = layers.ReLU()
    relu_torch = torch.nn.ReLU()

    compare_ag_with_torch([[x], [x2]], relu_torch.forward,
                          relu.forward, check_grad=True)


def test_shapes_coercing():
    shape1 = (1, 1, 20, 1)
    shape2 = (1,)
    axes_to_sum_over = autograd._find_axes_over_which_to_sum(shape1, shape2)
    assert axes_to_sum_over == [0, 1, 2]

    shape1 = (1, 1, 20, 1)
    shape2 = (20, 1)
    axes_to_sum_over = autograd._find_axes_over_which_to_sum(shape1, shape2)
    assert axes_to_sum_over == [0, 1]

    shape1 = (1, 20)
    shape2 = (20, 1)
    axes_to_sum_over = autograd._find_axes_over_which_to_sum(shape1, shape2)
    assert axes_to_sum_over == [0, 1]


def test_mse_loss():
    test_data = [
        [np.random.rand(1), np.random.rand(1)],
        [np.random.rand(10, 1), np.random.rand(10, 1)],
        [np.random.rand(1, 10), np.random.rand(1, 10)],
        [np.random.rand(100, 1, 1), np.random.rand(100, 1, 1)],
        [np.random.rand(32, 20, 20), np.random.rand(32, 20, 20)]
    ]
    torch_mse_loss = torch.nn.MSELoss()
    compare_ag_with_torch(test_data, torch_mse_loss,
                          losses.mse_loss, check_grad=True)


matrix_vector_multiplication_test()
matrix_multiplication_test()
nn_test()
relu_test()
batched_nn_test()
test_mse_loss()
