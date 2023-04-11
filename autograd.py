from __future__ import annotations
import time
import numpy as np


from typing import Union

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt


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
                # First, broadcast the vector row-wise to create a matrix.
                x_broadcasted = np.broadcast_to(x, (A.shape[0], len(x)))
                # Then broadcast the vector over the columns and multiply element-wise
                d_f_dA = x_broadcasted * grad_tensor[:, np.newaxis]
                # Multiply every column with the same vec element wise
                A_mult = A * grad_tensor[:, np.newaxis]
                # Then sum all rows
                d_f_dx = np.sum(A_mult.T, axis=-1)

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
                # Derivation of the following lines is three A4 pages incl. visualization
                B_exp = B[np.newaxis, :]
                M_exp = M[:, np.newaxis, :]
                R = B_exp * M_exp
                df_dA = np.sum(R, axis=2)
                A_exp = A[np.newaxis, :]
                M_exp2 = M.T[:, :, np.newaxis]
                R2 = A_exp * M_exp2
                df_dB = np.sum(R2, axis=1)
                df_dB = df_dB.T
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


def build_networkx_graph(root_node: Tensor):
    """
    Builds a networksx computation graph from the tensor tree, for debug visualization.
    We have to put the node objects as data, otherwise the graphviz_layout function throws an exception,
    it seems that this is a bug in networkx.
    """
    nx_graph = nx.DiGraph()

    def dfs(node: Tensor):
        nx_graph.add_node(node.id, ag_tensor=node)
        if not node.parents:
            return
        for parent in node.parents:
            nx_graph.add_edge(parent.id, node.id)
            dfs(parent)
    dfs(root_node)
    return nx_graph


def draw_computation_graph(root_tensor: Tensor, size=1.):
    """
    Draw the computation graph for a given root tensor
    Opens the window
    """

    nx_graph = build_networkx_graph(root_tensor)

    def node_type_to_color(node: Tensor):
        if node.operation:
            return "#E6BF00"
        elif node.is_variable:
            return "#E6E600"
        else:
            return "#0044cc"
    v_colors = [node_type_to_color(
        attributes["ag_tensor"]) for node, attributes in nx_graph.nodes(data=True)]

    node_positions = graphviz_layout(nx_graph, prog="dot")

    nx.draw_networkx_nodes(
        nx_graph, pos=node_positions, node_color=v_colors, edgecolors="#000000", node_size=size * 300)
    nx.draw_networkx_edges(
        nx_graph, pos=node_positions)

    def get_node_tag(node: Tensor):
        if node.operation:
            return node.operation
        return node.name

    node_labels = {node: get_node_tag(attributes["ag_tensor"])
                   for node, attributes in nx_graph.nodes(data=True)}
    nx.draw_networkx_labels(nx_graph, pos=node_positions, labels=node_labels,
                            font_color='black', font_size=size * 7, font_weight="bold")
    draw_node_result_labels(nx_graph, node_positions, size)
    draw_edge_result_labels(nx_graph, node_positions, size)
    plt.show()


def get_tensor_from_id(nx_graph, node_id):
    return nx_graph.nodes[node_id]["ag_tensor"]


def draw_edge_result_labels(nx_graph, node_positions, size):

    def get_edge_label(from_node: Tensor, to_node: Tensor):
        from_node, to_node = get_tensor_from_id(nx_graph,
                                                from_node), get_tensor_from_id(nx_graph, to_node)
        # Enable drawing before the backward pass
        if to_node.local_grad is not None:
            partial_d = to_node.local_grad[from_node.id]
            partial_d_str = "%.2f" % partial_d if isinstance(
                partial_d, float) else str(partial_d)
        else:
            partial_d_str = "Unknown"
        return f"d{to_node.name}/{from_node.name} = " + partial_d_str

    result_labels = {(from_node, to_node): get_edge_label(from_node, to_node)
                     for from_node, to_node in nx_graph.edges() if get_tensor_from_id(nx_graph, to_node).operation}

    nx.draw_networkx_edge_labels(nx_graph, pos=node_positions, edge_labels=result_labels,
                                 font_color='black', font_size=size * 5,
                                 rotate=False)


def draw_node_result_labels(nx_graph, node_positions, size):
    def op_node_label(node: Tensor):
        value_str = "%.2f" % node.value if isinstance(
            node.value, float) else str(node.value)
        return f"{node.name}={value_str}, grad: {node.grad}"
    result_labels = {node: op_node_label(attributes["ag_tensor"])
                     for node, attributes in nx_graph.nodes(data=True)}

    def get_node_pos(node_id, pos):
        x, y = pos
        # Draw for op-nodes the result on the bottom
        if get_tensor_from_id(nx_graph, node_id).operation:
            return (x, y-10)
        else:
            return (x, y+10)
    op_nodes_positions = {node_id: get_node_pos(
        node_id, pos) for node_id, pos in node_positions.items()}
    nx.draw_networkx_labels(nx_graph, pos=op_nodes_positions, labels=result_labels,
                            font_color='black', font_size=size * 5, verticalalignment="baseline")
