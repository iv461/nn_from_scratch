from __future__ import annotations

import numpy as np
from typing import Union


class Node:
    """
    Node in computation graph
    """
    id_cnt = 0

    def __init__(self, parents, child, name: str) -> None:
        if parents is not None:
            if isinstance(parents, list):
                if not all(map(lambda parent: isinstance(parent, Node), parents)):
                    raise Exception("All parents have to be other nodes")
        self.parents = parents
        self.child = child
        self.name = name
        self.id = Node.id_cnt
        Node.id_cnt += 1


class Tensor(Node):
    cnt = 0

    def __init__(self, value: np.ndarray, name: Union[str, None], is_variable=True, parents=None, op=None, is_batched=None) -> None:
        if not name:
            # TODO fix
            prefix = "f" if is_variable else "c"
            name = f"{prefix}_{Tensor.cnt}"
            Tensor.cnt += 1

        super().__init__(parents, None, name)
        """
        The value of the variable or contant,
         or the result of an operation if this node is an operation,
          in this case it is calculated and stored during the forward pass
        """
        self.value = value
        # The gradient vector
        self.grad = None
        # String expression of the gradient, for debug
        self.grad_exp = None
        # Wheter it is a variable and thus a gradient needs to be calculated w.r.t to it
        self.is_variable = is_variable
        # name of operation which produced this tensor, can be none if the node is a variable or constant
        self.operation = op
        # The local gradient vector, it is a dict from the Ids of the parent nodes as key
        # and the local partial derivatives to these parent nodes.
        # It can be calculated when the operation and the number of operands including their dimensions is known.
        self.local_grad = None
        # To not have to pass the is_batched flag to every tensor, we check if the value is_batched was passed, if not we determine
        # if this tensor is batched from its parents. Batched tensors always have to be of shape (batch_size, sample_size, ...)
        if is_batched is None:
            if parents is not None:
                self.is_batched = any(
                    [parent.is_batched for parent in parents])
            else:
                self.is_batched = False
        else:
            self.is_batched = is_batched

    def calc_local_grad_binary_ops(self, grad_tensor: Tensor):
        """
        Calculates the "local" partial derivatives of this function, for common binary operations like +, -, * etc.
        Sets self.local_grad and returns the local gradient multiplied with grad_tensor (chain-rule applied)
        """
        assert len(self.parents) == 2

        op1, op2 = self.parents[0], self.parents[1]
        both_scalar = (op1.value.ndim == 0 or op1.value.shape == (1, )) and (
            op2.value.ndim == 0 or op2.value.shape == (1, ))
        both_scalar_or_same_shape = both_scalar or (
            op1.value.shape == op2.value.shape)

        if self.operation == "+":
            assert both_scalar_or_same_shape
            self.local_grad = {
                parent_node.id: np.ones(op1.value.shape) for parent_node in self.parents}

        elif self.operation == "-":
            assert both_scalar_or_same_shape
            self.local_grad = {
                self.parents[0].id: np.ones(op1.value.shape),
                self.parents[1].id: -np.ones(op1.value.shape)}

        elif self.operation == "*":
            # Assert either both scalar, both vectors of same shape, or matrix-vector multiplication (Ax where A \in R^(n x m) and x \in R^m )
            # TODO matrix-matrix multiplication is not handled here !
            assert both_scalar_or_same_shape or (
                op1.value.ndim == 2 and op2.value.ndim == 1 and op1.value.shape[1] == op2.value.shape[0]) or (op1.value.shape == (1,) or op1.value.shape == () or op2.value.shape == (1,) or op2.value.shape == ())
            if both_scalar_or_same_shape:
                # scalar and scalar product between vectors case
                self.local_grad = {op1.id: op2.value, op2.id: op1.value}
            elif op1.value.shape == (1,) or op1.value.shape == () or op2.value.shape == (1,) or op2.value.shape == ():
                # Scalar-vector case
                if op1.value.shape == (1,) or op1.value.shape == ():
                    c = op1
                    v = op2
                else:
                    c = op2
                    v = op1
                d_f_dv = np.broadcast_to(c.value, (len(v.value), ))
                d_f_dc = v.value
                self.local_grad = {v.id: d_f_dv, c.id: d_f_dc}

            else:
                raise Exception("Invalid multiplication")
        elif self.operation == "@":
            if op1.value.ndim == 2 and op2.value.ndim == 1:
                # Matrix-vector case, Ax, op1 is A and op2 is x
                # Partial derivative w.r.t to the Matrix is the vector repeated as rows of the matrix
                A = op1.value
                x = op2.value
                # In linear algebra, this would be an ordinary matrix-multiplication as the line below,
                # but numpy throws an error when you try to matrix-multiply two vectors. So we have to explicitly specify the outer product.
                # Another option would be np.outer.
                # In numpy essentially you can't  calculate (A @ x) @ x.T given that x is a vector and A @ x is a valid matrix-multiplication.
                # It seems PEP 465 does not consider outer product.
                d_f_dA = grad_tensor[:, np.newaxis] @ x[np.newaxis, :]
                d_f_dx = grad_tensor @ A

                self.local_grad = {
                    self.parents[0].id: x,
                    self.parents[1].id: A}

                return {op1.id: d_f_dA,
                        op2.id: d_f_dx}
            elif op1.value.ndim == 2 and op2.value.ndim == 2:
                # matrix-matrix case
                M = grad_tensor
                A = op1.value
                B = op2.value
                df_dA = M @ B.T
                df_dB = A.T @ M
                self.local_grad = {
                    op1.id: B,
                    op2.id: A}

                return {op1.id: df_dA,
                        op2.id: df_dB}

        elif self.operation == "/":
            assert both_scalar_or_same_shape
            self.local_grad = {
                op1.id: 1. / op2.value,
                op2.id: (-op1.value / op2.value ** 2)}
        elif self.operation == "max":
            assert both_scalar_or_same_shape
            """
                Does element-wise comparison with numpy,
                if True, then it is converted to 1., otherwise to 0. .
                The partial derivatives of the max(a, b)
                are 1. for a when a is larger than b, otherwise 0 and vice-versa for b.
            """
            # TODO cast to same dtype, do not cast always to float, which implies double-precision
            self.local_grad = {
                op1.id: (op1.value > op2.value).astype(float),
                op2.id: (op2.value > op1.value).astype(float)
            }
        else:
            raise Exception(f"Binary op {self.operation} not implemented")

        return {op1.id: self.local_grad[op1.id]*grad_tensor,
                op2.id: self.local_grad[op2.id]*grad_tensor}

    def calc_local_grad_unary_ops(self, grad_tensor: Tensor):
        """
        Calculates the "local" partial derivatives of this function, 
        for common unary operations like unary sum (including single operand case), mean and square
        Sets self.local_grad and returns the local gradient multiplied with grad_tensor (chain-rule applied)
        """
        implemented_ops = ["mean", "sum", "square", "tanh"]
        if self.operation not in implemented_ops:
            raise Exception(f"Unary op {self.operation} not implemented")

        op1 = self.parents[0]
        if self.operation == "square":
            self.local_grad = {
                op1.id: 2. * op1.value
            }
        elif self.operation == "tanh":
            dx = np.ones(op1.value.shape) - np.square(np.tanh(op1.value))
            self.local_grad = {
                op1.id: dx
            }
        elif self.operation == "mean" or self.operation == "sum":
            normalizer = 1. if self.operation == "sum" else 1. / op1.value.size
            self.local_grad = {
                op1.id: normalizer * np.ones(op1.value.shape)
            }

        return {op1.id: self.local_grad[op1.id]*grad_tensor}

    def calc_local_gradient_and_mul(self, grad_tensor: Tensor):
        """
        Get the gradient of the output of this opnode w.r.t to all the parents.
        This calculates the local gradient and multiplies it with the already multplied gradiend 
        along the path of the computational graph. grad_tensor is needed as for non-scalar output ops, 
        the grad_tensor may need expansion ops for multiplication.
        Sets self.local_grad and returns the local gradient multiplied with grad_tensor (chain-rule applied)
        """
        if len(self.parents) == 1:
            return self.calc_local_grad_unary_ops(grad_tensor)
        elif len(self.parents) == 2:
            return self.calc_local_grad_binary_ops(grad_tensor)

    def backward(self, trace=False, profile=False):
        """
        Runs backprop, returns dict of the Tensor names as dict
        """
        gradient = {}
        # TODO vector-valued output, multiply parent with child

        def dfs(node: Tensor):
            multiplied_grad = node.calc_local_gradient_and_mul(node.grad)
            for parent in node.parents:
                # Skip calculation of gradients for nodes for which we should not calculate the gradient
                if not parent.is_variable:
                    continue

                grad_elem = multiplied_grad[parent.id]

                if trace:
                    print(
                        f"Grad product of for d{node.name}/{parent.name} is:\n{grad_elem}")

                if parent.grad is None:
                    parent.grad = 0.
                parent.grad += grad_elem
                gradient[parent.name] = grad_elem
                if parent.operation:

                    dfs(parent)

        self.grad = np.array(1.)
        dfs(self)
        return gradient

    def op_checks_vector(self, other_operand):
        if not isinstance(other_operand, Tensor):
            raise Exception(
                f"Operation with {other_operand} of type {type(other_operand)} not supported")
        return
        # Determine which is operand is a vector of training examples, i.e. the batch
        # Now check if both shapes are compatible for element-wise operation
        if len(self.value.shape) > len(other_operand.value.shape):
            shapes_compat = self.value.shape[1:] == other_operand.value.shape
        elif len(self.value.shape) == len(other_operand.value.shape):
            shapes_compat = self.value.shape == other_operand.value.shape
        else:
            shapes_compat = self.value.shape == other_operand.value.shape[1:]
        if not shapes_compat:
            raise Exception(
                f"Operation with {other_operand} of with shapes {self.value.shape} and {other_operand.value.shape} not supported")

    def __add__(self, other):
        """
        Works also over batch size because of automatic broadcasting of numpy, 
        e.g. addition works with x = np.ones((2, 3)) and w = np.arange(3).
        """
        self.op_checks_vector(other)
        return Tensor(self.value + other.value, None, True, parents=[self, other], op="+")

    def __radd__(self, other):
        return type(self).__add__(self, other)

    def __sub__(self, other):
        self.op_checks_vector(other)
        return Tensor(self.value - other.value, None, True, parents=[self, other], op="-")

    def __rsub__(self, other):
        return type(self).__sub__(self, other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise Exception(f"Mul with {type(other)} not supported")
        result = self.value * other.value
        return Tensor(result, None, True, parents=[self, other], op="*")

    def __matmul__(self, other):
        result = np.matmul(self.value, other.value)
        return Tensor(result, None, True, parents=[self, other], op="@")

    def __rmul__(self, other):
        return type(self).__mul__(self, other)

    def __str__(self) -> str:
        return f"{self.value}, grad: {self.grad}"


# common functions
def _max(op1: Tensor, op2: Union[Tensor, float]):
    if not (isinstance(op2, float) or isinstance(op2, Tensor)):
        raise Exception("")
    if isinstance(op2, float):
        # TODO name, add ability to make unique names
        # Broadcast to be able to do max(np.array(...), 0)
        if not isinstance(op1.value, float):
            op2 = np.broadcast_to(op2, op1.value.shape)
        op2_t = Tensor(value=op2, name=None, is_variable=False)
    else:
        op2_t = op2
    val = np.maximum(op1.value, op2_t.value)
    return Tensor(val, None, True, [op1, op2_t], "max")


def max(op1, op2):
    if not (isinstance(op1, Tensor) or isinstance(op2, Tensor)):
        return max(op1, op2)
    else:
        if isinstance(op1, Tensor):
            return _max(op1, op2)
        else:
            return _max(op2, op1)


def mean(op1: Tensor):
    return Tensor(np.mean(op1.value), None, True, [op1], "mean")


def sum(op1: Tensor):
    return Tensor(np.sum(op1.value), None, True, [op1], "sum")


def square(op1: Tensor):
    return Tensor(np.square(op1.value), None, True, [op1], "square")


def tanh(op1: Tensor):
    return Tensor(np.tanh(op1.value), None, True, [op1], "tanh")
