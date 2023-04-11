import numpy as np

from torch import nn, from_numpy as t_from_from_numpy
import torch
from torch import functional as F


import autograd

np.random.default_rng(seed=123456)


def matrix_vector_multiplication():
    A = np.arange(12).reshape(3, 4).astype(np.float32)
    x = np.arange(4).astype(np.float32)
    m = 3 * np.arange(3).astype(np.float32)

    print(f"A: {A}\nx:{x}")

    A_t = torch.tensor(A, device="cpu", requires_grad=True)
    x_t = torch.tensor(x, device="cpu", requires_grad=True)
    m_t = torch.tensor(m, device="cpu", requires_grad=True)
    b = A_t @ x_t
    print(f"b is: {b.shape}")
    b *= m_t
    b = torch.sum(b)
    b.backward()

    print(f"Torch A grad: {A_t.grad}")
    print(f"Torch x grad: {x_t.grad}")
    print(f"Torch x grad: {m_t.grad}")

    A_t2 = autograd.Tensor(A, "A")
    x_t2 = autograd.Tensor(x, "x")
    m_t2 = autograd.Tensor(m, "m")
    b2 = A_t2 @ x_t2
    b2 = b2 * m_t2
    b2 = autograd.sum(b2)
    b2.backward()

    print(f"AG A grad: {A_t2.grad}")
    print(f"AG x grad: {x_t2.grad}")
    print(f"AG m grad: {m_t2.grad}")
    assert np.allclose(A_t.grad.numpy(), A_t2.grad)
    assert np.allclose(x_t.grad.numpy(), x_t2.grad)
    assert np.allclose(m_t.grad.numpy(), m_t2.grad)


def matrix_multiplication():
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
    C = A_t @ B_t
    print(f"A @ B = C =\n{C.shape}")
    C = C * M_t
    b = torch.sum(C)
    b.backward()

    print(f"Torch A grad: {A_t.grad}")
    print(f"Torch B grad: {B_t.grad}")
    print(f"Torch M grad: {M_t.grad}")

    A_t2 = autograd.Tensor(A, "A")
    B_t2 = autograd.Tensor(B, "B")
    M_t2 = autograd.Tensor(M, "C")
    C2 = A_t2 @ B_t2
    C2 = C2 * M_t2
    b2 = autograd.sum(C2)
    b2.backward()

    print(f"AG A grad: {A_t2.grad}")
    print(f"AG B grad: {B_t2.grad}")
    print(f"AG M grad: {M_t2.grad}")
    assert np.allclose(A_t.grad.numpy(), A_t2.grad)
    assert np.allclose(B_t.grad.numpy(), B_t2.grad)
    assert np.allclose(M_t.grad.numpy(), M_t2.grad)


matrix_vector_multiplication()
matrix_multiplication()
