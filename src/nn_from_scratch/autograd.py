from __future__ import annotations

import numpy as np
from typing import Union, List, Optional


def _find_axes_over_which_to_sum(bigger: tuple, smaller: tuple):
    # This will throw an error if both shapes are not broadcastable
    np.broadcast_shapes(bigger, smaller)
    # Is shape size is not equal, broadcasting rule says we fill with ones. Thus the axes are all axes which do not exist one of the shapes
    len_diff = len(bigger) - len(smaller)
    assert len_diff >= 0, "First shape has to be bigger or equal the second shape"
    # broadcast the smaller shape, insert 1 size axes
    smaller = ([1] * len_diff) + list(smaller)
    axes_to_sum_over = []
    for i, (shape1_axis, shape2_axis) in enumerate(zip(bigger, smaller)):
        if shape1_axis != shape2_axis:
            axes_to_sum_over.append(i)
    return axes_to_sum_over


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

    def __init__(self, value: np.ndarray, name: Optional[str] = None,
                 requires_grad=False, parents: Optional[List[Tensor]] = None,
                 op: Optional[str] = None, is_batched=None):
        if name is None:
            # TODO fix
            prefix = "f" if requires_grad else "c"
            name = f"{prefix}_{Tensor.cnt}"
            Tensor.cnt += 1

        super().__init__(parents, None, name)
        """
        The value of the variable or constant,
         or the result of an operation if this node is an operation,
          in this case it is calculated and stored during the forward pass
        """
        self.value = value
        # The gradient vector
        self.grad = None
        # String expression of the gradient, for debug
        self.grad_exp = None
        # Whether a gradient needs to be calculated w.r.t to it or not.
        self.requires_grad = requires_grad
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

    def numpy(self): return self.value

    def backward(self, trace=False, profile=False):
        """
        Runs backpropagation algoritm and sets all the gradients for tensors requiring gradients.
        It assumes that this tensor is scalar-valued.
        """
        def dfs(node: Tensor):
            multiplied_grad = node._backprop(node.grad)
            for parent in node.parents:
                # Skip calculation of gradients for nodes for which we should not calculate the gradient
                if not parent.requires_grad:
                    continue
                grad_elem = multiplied_grad[parent.id]

                if trace:
                    print(
                        f"Grad product of for d{node.name}/{parent.name} is:\n{grad_elem}")
                if parent.grad is None:
                    parent.grad = 0.
                parent.grad += grad_elem
                if parent.operation:
                    dfs(parent)

        self.grad = np.array(1.)
        dfs(self)

    def __add__(self, other):
        """
        Works also over batch size because of automatic broadcasting of numpy, 
        e.g. addition works with x = np.ones((2, 3)) and w = np.arange(3).
        """
        self._op_checks_vector(other)
        return Tensor(self.value + other.value, None, True, parents=[self, other], op="+")

    def __radd__(self, other): return type(self).__add__(self, other)

    def __sub__(self, other):
        self._op_checks_vector(other)
        return Tensor(self.value - other.value, None, True, parents=[self, other], op="-")

    def __rsub__(self, other): return type(self).__sub__(self, other)

    def __mul__(self, other: Tensor): return Tensor(
        self.value * other.value, None, True, parents=[self, other], op="*")

    def __matmul__(self, other: Tensor): return Tensor(
        self.value @ other.value, None, True, parents=[self, other], op="@")

    def __rmul__(self, other: Tensor): return type(self).__mul__(self, other)

    def __str__(self) -> str: return f"{self.value}, grad: {self.grad}"

    def _sum_over_batch_axis_if_needed(self, parents, accumulated_gradient):
        """
        For processing batches, we have to sum the gradients over the batch-dimension.

        Args:
            parents (_type_): _description_
            accumulated_gradient (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Hint we use here < instead of comparing if we have one dimension more to catch the case where we have (30, 1, 1) and (1,) for example.
        # For this exact case we reshape also, to remove unnesecary 1 dim axes. Squeeze is not enough, a sometimes, the bias in the linear layer e.g. is (1, 20), in this case we do not want to squeeze this axis, we actually want the grad shape to match exactly the value shape
        out_dict = {}
        for op_id, acc_grad in accumulated_gradient.items():
            # TODO hack which saves compute but is actually here as we have not generalized summing over all broadcast axes yet: Don't do anything if we will not need the grad anyways
            if not parents[op_id].requires_grad:
                new_acc_grad = acc_grad
            # TODO hacky check, assumes that without the batch dimension we have only a matrix
            elif acc_grad.ndim == 3 and parents[op_id].value.ndim < 3:
                new_acc_grad = np.reshape(
                    np.sum(acc_grad, axis=0), parents[op_id].value.shape)
            else:
                new_acc_grad = acc_grad

            out_dict[op_id] = new_acc_grad
        return out_dict

    def _backprop_binary_ops(self, grad_output: np.ndarray):
        """
        Calculates the "local" partial derivatives of this function, for common binary operations like +, -, * etc.
        Sets self.local_grad and returns the local gradient multiplied with grad_output (chain-rule applied)
        """
        assert len(self.parents) == 2

        op1, op2 = self.parents[0], self.parents[1]
        # TODO fix this, this is needed for sum batch function. Also, actually we do not need it as the dict is ordered in CPython
        parents_dict = {op1.id: op1, op2.id: op2}
        both_scalar = (op1.value.ndim == 0 or op1.value.shape == (1, )) and (
            op2.value.ndim == 0 or op2.value.shape == (1, ))
        both_scalar_or_same_shape = both_scalar or (
            op1.value.shape == op2.value.shape)

        if self.operation == "+":
            self.local_grad = {
                parent_node.id: np.ones(parent_node.value.shape) for parent_node in self.parents}

        elif self.operation == "-":
            self.local_grad = {
                op1.id: np.ones(op1.value.shape),
                op2.id: -np.ones(op2.value.shape)}

        elif self.operation == "*":
            # Assert either both scalar, both vectors of same shape, or matrix-vector multiplication (Ax where A \in R^(n x m) and x \in R^m )
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
            op1_dim = op1.value.ndim if op1.value.ndim < 3 else op1.value.ndim - 1
            op2_dim = op2.value.ndim if op2.value.ndim < 3 else op2.value.ndim - 1
            if (op1_dim - 1) == op2_dim:
                # Matrix-vector case, Ax, op1 is A and op2 is x
                # Partial derivative w.r.t to the Matrix is the vector repeated as rows of the matrix
                A = op1.value
                x = op2.value
                # In linear algebra, this would be an ordinary matrix-multiplication as the line below,
                # but numpy throws an error when you try to matrix-multiply two vectors. So we have to explicitly specify the outer product.
                # Another option would be np.outer.
                # In numpy essentially you can't  calculate (A @ x) @ x.T given that x is a vector and A @ x is a valid matrix-multiplication.
                # It seems PEP 465 does not consider outer product.
                d_f_dA = grad_output[:, np.newaxis] @ x[np.newaxis, :]
                d_f_dx = grad_output @ A

                self.local_grad = {
                    self.parents[0].id: x,
                    self.parents[1].id: A}

                return self._sum_over_batch_axis_if_needed(parents_dict, {op1.id: d_f_dA,
                                                                          op2.id: d_f_dx})
            elif op1_dim == op2_dim:
                # matrix-matrix case
                A = op1.value
                B = op2.value
                # We use np.swapaxes(B, 0, 1) instead of B.T as .T is ambigous for multi-dim arrays, and here needed for batch-sized inputs, see also:

                df_dA = grad_output @ np.swapaxes(B, -1, -2)
                df_dB = np.swapaxes(A, -1, -2) @ grad_output
                self.local_grad = {
                    op1.id: B,
                    op2.id: A}

                return self._sum_over_batch_axis_if_needed(parents_dict, {op1.id: df_dA,
                                                                          op2.id: df_dB})
            else:
                raise Exception("Unknown matrix backward")

        elif self.operation == "/":
            assert both_scalar_or_same_shape
            self.local_grad = {
                op1.id: 1. / op2.value,
                op2.id: (-op1.value / op2.value ** 2)}
        elif self.operation == "max":
            #assert both_scalar_or_same_shape
            """
                Does element-wise comparison with numpy,
                if True, then it is converted to 1., otherwise to 0. .
                The partial derivatives of the max(a, b)
                are 1. for a when a is larger than b, otherwise 0 and vice-versa for b.
            """
            self.local_grad = {
                op1.id: (op1.value > op2.value).astype(op1.value.dtype),
                op2.id: (op2.value > op1.value).astype(op1.value.dtype)
            }
        else:
            raise Exception(f"Binary op {self.operation} not implemented")

        acc_grad = {op1.id: self.local_grad[op1.id]*grad_output,
                    op2.id: self.local_grad[op2.id]*grad_output}
        acc_grad = self._sum_over_batch_axis_if_needed(parents_dict, acc_grad)
        return acc_grad

    def _backprop_unary_ops(self, grad_tensor: np.ndarray):
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

        # TODO fix this, this is needed for sum batch function. Also, actually we do not need it as the dict is ordered in CPython
        parents_dict = {op1.id: op1}
        acc_grad = {op1.id: self.local_grad[op1.id]*grad_tensor}
        acc_grad = self._sum_over_batch_axis_if_needed(parents_dict, acc_grad)
        return acc_grad

    def _backprop(self, grad_tensor: np.ndarray):
        """
        Calculates the gradient of the output of this op-node w.r.t to all the parents.
        This calculates the local gradient (partial derivatives) and multiplies it with the already multiplied gradient
        along the path of the computational graph. grad_tensor is needed as for non-scalar output ops, 
        the grad_tensor may need expansion ops for multiplication.
        Sets self.local_grad and returns the local gradient multiplied with grad_tensor, thus applies the chain-rule.
        """
        if len(self.parents) == 1:
            return self._backprop_unary_ops(grad_tensor)
        return self._backprop_binary_ops(grad_tensor)

    def _op_checks_vector(self, other_operand):
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

# common functions


def _max(op1: Tensor, op2: Union[Tensor, float]):
    if not (isinstance(op2, float) or isinstance(op2, Tensor)):
        raise Exception("")
    if isinstance(op2, float):
        # TODO name, add ability to make unique names
        # Broadcast to be able to do max(np.array(...), 0)
        if not isinstance(op1.value, float):
            op2 = np.broadcast_to(op2, op1.value.shape)
        op2_t = Tensor(value=op2, name=None, requires_grad=False)
    else:
        op2_t = op2
    result = np.maximum(op1.value, op2_t.value)
    return Tensor(result, None, True, [op1, op2_t], "max")


def max(op1, op2):
    if isinstance(op1, Tensor):
        return _max(op1, op2)
    else:
        return _max(op2, op1)


def mean(op1: Tensor): return Tensor(
    np.mean(op1.value), None, True, [op1], "mean")


def sum(op1: Tensor): return Tensor(
    np.sum(op1.value), None, True, [op1], "sum")


def square(op1: Tensor): return Tensor(
    np.square(op1.value), None, True, [op1], "square")


def tanh(op1: Tensor): return Tensor(
    np.tanh(op1.value), None, True, [op1], "tanh")
