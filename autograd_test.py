import numpy as np

from torch import nn, from_numpy as t_from_from_numpy
import torch
from torch import functional as F


import autograd

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

A_t2 = autograd.Tensor(A, "A")
x_t2 = autograd.Tensor(x, "x")
m_t2 = autograd.Tensor(m, "m")
b2 = A_t2 * x_t2
b2 = b2 * m_t2
b2 = autograd.sum(b2)
b2.backward()

print(f"AG A grad: {A_t2.grad}")
print(f"AG x grad: {x_t2.grad}")
