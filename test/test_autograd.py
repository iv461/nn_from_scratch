from typing import List
import numpy as np
import torch
from nn_from_scratch import autograd, layers

np.random.default_rng(seed=587346287)


def matrix_multiplication(A, x, b):
    r = A @ x
    r *= b
    return r


def two_layer_nn(x, w1, b1, w2, b2, w3, b3):
    res = w1 @ x + b1
    res = w2 @ res + b2
    res = w3 @ res + b3
    return res


def matrix_vector_multiplication_test():
    A = np.arange(12).reshape(3, 4).astype(np.float32)
    x = np.arange(4).astype(np.float32)
    m = 3 * np.arange(3).astype(np.float32)

    print(f"A: {A}\nx:{x}")

    A_t = torch.tensor(A, device="cpu", requires_grad=True)
    x_t = torch.tensor(x, device="cpu", requires_grad=True)
    m_t = torch.tensor(m, device="cpu", requires_grad=True)
    r = matrix_multiplication(A_t, x_t, m_t)
    r = torch.sum(r)
    r.backward()

    print(f"Torch A grad: {A_t.grad}")
    print(f"Torch x grad: {x_t.grad}")
    print(f"Torch x grad: {m_t.grad}")

    A_t2 = autograd.Tensor(A, "A", requires_grad=True)
    x_t2 = autograd.Tensor(x, "x", requires_grad=True)
    m_t2 = autograd.Tensor(m, "m", requires_grad=True)
    r2 = matrix_multiplication(A_t2, x_t2, m_t2)
    r2 = autograd.sum(r2)
    r2.backward()

    print(f"AG A grad: {A_t2.grad}")
    print(f"AG x grad: {x_t2.grad}")
    print(f"AG m grad: {m_t2.grad}")
    assert np.allclose(r.detach().numpy(), r2.value)
    assert np.allclose(A_t.grad.numpy(), A_t2.grad)
    assert np.allclose(x_t.grad.numpy(), x_t2.grad)
    assert np.allclose(m_t.grad.numpy(), m_t2.grad)


def matrix_multiplication_test():
    m = 2
    n = 3
    p = 4
    A = np.random.rand(m, n).astype(np.float32)
    B = np.random.rand(n, p).astype(np.float32)
    M = np.random.rand(m, p).astype(np.float32)

    print(f"A:\n{A}\nB:{B}\nM:\n{M}")

    A_t = torch.tensor(A, device="cpu", requires_grad=True)
    B_t = torch.tensor(B, device="cpu", requires_grad=True)
    M_t = torch.tensor(M, device="cpu", requires_grad=True)
    r = matrix_multiplication(A_t, B_t, M_t)
    r = torch.sum(r)
    r.backward()

    print(f"Torch A grad: {A_t.grad}")
    print(f"Torch B grad: {B_t.grad}")
    print(f"Torch M grad: {M_t.grad}")

    A_t2 = autograd.Tensor(A, "A", requires_grad=True)
    B_t2 = autograd.Tensor(B, "B", requires_grad=True)
    M_t2 = autograd.Tensor(M, "C", requires_grad=True)
    r2 = matrix_multiplication(A_t2, B_t2, M_t2)
    r2 = autograd.sum(r2)
    r2.backward()

    print(f"AG A grad: {A_t2.grad}")
    print(f"AG B grad: {B_t2.grad}")
    print(f"AG M grad: {M_t2.grad}")
    assert np.allclose(r.detach().numpy(), r2.value)
    assert np.allclose(A_t.grad.numpy(), A_t2.grad)
    assert np.allclose(B_t.grad.numpy(), B_t2.grad)
    assert np.allclose(M_t.grad.numpy(), M_t2.grad)


def nn_test():
    intermediate_feat = 20
    x = np.random.rand(1).astype(np.float32)

    w1 = np.random.rand(intermediate_feat, 1).astype(np.float32)
    b1 = np.random.rand(intermediate_feat).astype(np.float32)
    w2 = np.random.rand(intermediate_feat,
                        intermediate_feat).astype(np.float32)
    b2 = np.random.rand(intermediate_feat).astype(np.float32)
    w3 = np.random.rand(1, intermediate_feat).astype(np.float32)
    b3 = np.random.rand(1).astype(np.float32)

    params = [w1, b1, w2, b2, w3, b3]
    params_torch = map(lambda p: torch.tensor(
        p, device="cpu", requires_grad=True), params)
    params_ag = map(lambda p: autograd.Tensor(
        p, "p", requires_grad=True), params)

    x_t = torch.tensor(x, device="cpu", requires_grad=True)
    y = two_layer_nn(x_t, *params_torch)
    y.backward()

    x_t2 = autograd.Tensor(x, "x", requires_grad=True)
    y2 = two_layer_nn(x_t2, *params_ag)
    y2.backward()

    assert np.allclose(y2.value, y.detach().numpy())

    assert all([np.allclose(p_torch.grad.numpy(), p_ag.grad)
               for p_torch, p_ag in zip(params_torch, params_ag)])


def compare_ag_with_torch(data: List[List[np.ndarray]], f_torch, f_ag, check_grad=True):
    torch_outputs = []
    ag_outputs = []
    for inputs in data:
        torch_x = [torch.tensor(
            x, device="cpu", requires_grad=check_grad)for x in inputs]
        ag_x = [autograd.Tensor(x, requires_grad=check_grad) for x in inputs]
        torch_y = f_torch(*torch_x)
        ag_y = f_ag(*ag_x)
        torch_outputs.append(torch_y)
        ag_outputs.append(ag_y)
        if check_grad:
            torch_y.backward()
            ag_y.backward()

    assert all([np.allclose(p_ag.value, p_torch.detach().numpy())
               for p_torch, p_ag in zip(torch_outputs, ag_outputs)])
    assert all([np.allclose(p_torch.grad.numpy(), p_ag.grad)
               for p_torch, p_ag in zip(torch_outputs, ag_outputs)])


def relu_test():
    x = np.random.rand(1).astype(np.float32)
    x2 = np.random.rand(10, 1).astype(np.float32)
    relu = layers.ReLu()
    relu_torch = torch.nn.ReLU()
    # Sum is needed to have scalar-output for backward
    def f_ag(x): return autograd.sum(relu.forward(x))
    def f_torch(x): return torch.sum(relu_torch.forward(x))
    compare_ag_with_torch([[x], [x2]], f_torch, f_ag, check_grad=True)


matrix_vector_multiplication_test()
matrix_multiplication_test()
nn_test()
# relu_test()
