import functools
from tkinter import E
import networkx as nx
import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt


import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import itertools
from dataclasses import dataclass

from typing import List, Union, Deque, Dict


class Node:
    """
    node in computation graph
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

    def __init__(self, value: np.ndarray, name: Union[str, None], is_variable=True, parents=None, op=None) -> None:
        if not name:
            # TODO fix
            prefix = "f" if is_variable else "c"
            name = f"{prefix}_{Tensor.cnt}"
            Tensor.cnt += 1

        super().__init__(parents, None, name)
        self.value = value
        self.grad = None
        # String expression of the gradient, for debug
        self.grad_exp = None
        # Wheter it is a variable and thus a gradient needs to be calculated w.r.t to it
        self.is_variable = is_variable
        # name of operation which produced this tensor, can be none if leaf
        self.operation = op
        self.local_grad = None

    def calc_local_grad_binary_ops(self):
        assert len(self.parents) == 2
        # Some partial derivatices of common functions
        op1, op2 = self.parents[0], self.parents[1]
        both_scalar = (isinstance(op1.value, float)
                       and isinstance(op2.value, float))
        both_scalar_or_same_shape = both_scalar or (
            op1.value.shape == op2.value.shape)

        if self.operation == "+":
            assert both_scalar_or_same_shape
            if not isinstance(op1.value, float):
                self.local_grad = {
                    parent_node.id: np.ones((len(op1.value),)) for parent_node in self.parents}
            else:
                self.local_grad = {
                    parent_node.id: 1. for parent_node in self.parents}
        elif self.operation == "-":
            assert both_scalar_or_same_shape
            if not isinstance(op1.value, float):
                self.local_grad = {
                    self.parents[0].id: np.ones((len(op1.value),)),
                    self.parents[1].id: -np.ones((len(op1.value),))}
            else:
                self.local_grad = {
                    self.parents[0].id: 1.,
                    self.parents[1].id: -1.}
        elif self.operation == "*":
            # only the case for Ax where A \in R^(n x m) and x \in R^m implemented currently
            assert both_scalar_or_same_shape or (
                op1.value.ndim == 2 and op1.value.shape[0] == op1.value.shape[1] and op2.value.ndim == 1 and op1.value.shape[0] == op2.value.shape[0])
            if both_scalar:
                self.local_grad = {op1.id: op2.value, op2.id: op1.value}
            elif op1.value.ndim == 2 and op2.value.ndim == 1:  # scalar and scalar product between vectors case
                # Partial derivative w.r.t to the Matrix is the vector repeated as rows of the matrix
                d_f_dA = np.broadcast_to(
                    op2.value, (op1.value.shape[0], len(op2.value)))
                # TODO correct ?
                d_f_dx = op1.value
                self.local_grad = {
                    self.parents[0].id: d_f_dA,
                    self.parents[1].id: d_f_dx}
            else:
                raise Exception("Invalid mul")

        elif self.operation == "/":
            assert both_scalar_or_same_shape
            self.local_grad = {
                op1.id: 1. / op2.value,
                op2.id: (-op1.value / op2.value ** 2)}
        elif self.operation == "max":
            assert both_scalar_or_same_shape
            if both_scalar:
                self.local_grad = {
                    op1.id: float(op1.value > op2.value),
                    op2.id: float(op2.value > op1.value)}
            else:
                """
                 Does element-wise comparison with numpy, 
                 if True, then it is converted to 1., otherwise zero. 
                 The partial derivatives of the max(a, b)
                  are for a 1 whem a is larger than b, otherwise 0 and vice-versa for b.
                """
                self.local_grad = {
                    op1.id: (op1.value > op2.value).astype(float),
                    op2.id: (op2.value > op1.value).astype(float)
                }
        else:
            raise Exception(f"Binary op {self.operation} not implemented")

    def calc_local_grad_unary_ops(self):
        if self.operation == "mean" or self.operation == "sum" or self.operation == "square":
            op1 = self.parents[0]
            if isinstance(op1.value, float):
                self.local_grad = {
                    op1.id: 2. if self.operation == "square" else 1.
                }
            else:
                normalizer = 1. if self.operation == "sum" else (
                    1. / op1.value.size if self.operation == "mean" else 2.)
                self.local_grad = {
                    op1.id: normalizer * np.ones(op1.value.shape)
                }
        else:
            raise Exception(f"Unary op {self.operation} not implemented")

    def calc_local_gradient(self):
        """
        Get the gradient of the output of this opnode w.r.t to all the parents. 
        This calculates the local gradient
        """
        if len(self.parents) == 1:
            self.calc_local_grad_unary_ops()
        elif len(self.parents) == 2:
            self.calc_local_grad_binary_ops()

        if False:
            if self.operation == "+":
                self.grad_exp = {
                    op1.id: f"1",
                    op2.id: f"1"}
            elif self.operation == "-":
                self.grad_exp = {
                    op1.id: f"1",
                    op2.id: f"-1"}
            elif self.operation == "*":
                # TODO special, binary case
                self.grad_exp = {
                    op1.id: f"{op2.name}",
                    op2.id: f"{op1.name}"}
            elif self.operation == "/":
                self.grad_exp = {
                    op1.id: f"(1. / {op2.name})",
                    op2.id: f"({-op1.name} / {op2.name} ** 2)"}
            else:
                raise Exception(f"Op {self.operation} not implemented")

    def backward(self, clear=False):
        """
        runs backprop
        """
        gradient = {}
        # TODO vector-valued output, multiply parent with child

        def dfs(node: Tensor, accumulated_product=1.):
            node.calc_local_gradient()
            for parent in node.parents:
                grad_elem = node.local_grad[parent.id] * accumulated_product
                if parent.operation:
                    dfs(parent, grad_elem)
                else:
                    # Set the gradient, terminate search
                    if clear:
                        parent.grad = None
                    else:
                        if not parent.grad:
                            parent.grad = 0.
                        parent.grad += grad_elem
                        gradient[parent.name] = grad_elem

        dfs(self)
        return gradient

    def reset_gradient(self):
        self.backward(clear=True)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise Exception(
                f"Add with {other} of type {type(other)} not supported")
        assert other.value.shape == self.value.shape
        return Tensor(self.value + other.value, None, True, parents=[self, other], op="+")

    def __radd__(self, other):
        return type(self).__add__(self, other)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            raise Exception(f"Sub with {type(other)} not supported")
        assert other.value.shape == self.value.shape
        return Tensor(self.value - other.value, None, True, parents=[self, other], op="-")

    def __rsub__(self, other):
        return type(self).__sub__(self, other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise Exception(f"Mul with {type(other)} not supported")
        if isinstance(self.value, float) and isinstance(other.value, float):
            result = self.value * other.value
        else:
            result = np.matmul(self.value, other.value)
        return Tensor(result, None, True, parents=[self, other], op="*")

    def __rmul__(self, other):
        return type(self).__mul__(self, other)


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
    op2_t = op2
    return Tensor(np.max(op1.value, op2.value), None, True, [op1, op2_t], "max")


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


def build_networkx_graph(root_node: Tensor):
    nx_graph = nx.DiGraph()

    def dfs(node: Tensor):
        # attribute with the key "'name'" collides when converting to dot with pydot, thus we name id "ag_name"
        if node.operation:
            print(f"Adding node {node.name}({node.operation})")
        else:
            print(f"Adding node {node.name}({node.id})")
        nx_graph.add_node(node, ag_name=node.name)
        if not node.parents:
            return
        for parent in node.parents:
            if parent.operation:
                print(
                    f"Adding edge from node {node.name}({node.id}) to {parent.name}({parent.operation})")
            else:
                print(
                    f"Adding edge from node {node.name}({node.id}) to {parent.name}({parent.id})")
            nx_graph.add_edge(node, parent)
            dfs(parent)
    dfs(root_node)
    return nx_graph


def draw_computation_graph(nx_graph, size=1.):
    def node_type_to_color(node: Tensor):
        if node.operation:
            return "#E6BF00"
        elif node.is_variable:
            return "#E6E600"
        else:
            return "#0044cc"
    v_colors = [node_type_to_color(
        node) for node, attributes in nx_graph.nodes(data=True)]

    node_positions = graphviz_layout(nx_graph, prog="dot")

    nx.draw_networkx_nodes(
        nx_graph, pos=node_positions, node_color=v_colors, edgecolors="#000000", node_size=size * 300)
    nx.draw_networkx_edges(
        nx_graph, pos=node_positions)
    node_labels = {node: node.name
                   for node, attributes in nx_graph.nodes(data=True)}
    nx.draw_networkx_labels(nx_graph, pos=node_positions, labels=node_labels,
                            font_color='black', font_size=size * 7, font_weight="bold")
    draw_node_result_labels(nx_graph, node_positions, size)
    draw_edge_result_labels(nx_graph, node_positions, size)
    plt.show()


def draw_edge_result_labels(nx_graph, node_positions, size):
    def get_edge_label(from_node, to_node):
        from_node_name = from_node.result_name if from_node.op else from_node.name
        partial_d_value = to_node.local_grad[from_node.id]
        return f"d{to_node.result_name}/{from_node_name} = " + str(partial_d_value)
    result_labels = {(from_node, to_node): get_edge_label(from_node, to_node)
                     for from_node, to_node, data in nx_graph.edges(data=True) if to_node.op}

    nx.draw_networkx_edge_labels(nx_graph, pos=node_positions, edge_labels=result_labels,
                                 font_color='black', font_size=size * 5,
                                 rotate=False)


def draw_node_result_labels(nx_graph, node_positions, size):
    def op_node_label(node: Tensor):
        if node.operation:
            return f"{node.result_name}=" + str(node.value)
        else:
            return f"{node.name}=" + str(node.value)
    result_labels = {node: op_node_label(node)
                     for node, attributes in nx_graph.nodes(data=True)}

    def get_node_pos(node, pos):
        x, y = pos
        # Draw for op-nodes the result on the bottom
        if node.operation:
            return (x, y-10)
        else:
            return (x, y+10)
    op_nodes_positions = {node_id: get_node_pos(
        node_id, pos) for node_id, pos in node_positions.items()}
    nx.draw_networkx_labels(nx_graph, pos=op_nodes_positions, labels=result_labels,
                            font_color='black', font_size=size * 5, verticalalignment="baseline")


class Perceptron():
    random_gen = np.random.default_rng(seed=123456)

    def __init__(self, in_features):
        self.in_features = in_features

        k = 1. / in_features
        k_sqrt = math.sqrt(k)
        init_vals = Perceptron.random_gen.uniform(
            -k_sqrt, k_sqrt, in_features+1)
        self.wheight = [Tensor(
            init_vals[i], name=f"w{i}") for i in range(self.in_features)]

        self.bias = Tensor(init_vals[-1], name="b")

    def forward(self, x):
        assert len(x) == self.in_features
        x_dual_num = [Tensor(
            float(x_i), f"x_{i}", is_variable=False) for i, x_i in enumerate(x)]

        dot_prod = x_dual_num[0] * self.wheight[0]
        for i, (x_i, w_i) in enumerate(zip(x_dual_num, self.wheight)):
            if i == 0:
                continue
            dot_prod += x_i * w_i
        return dot_prod + self.bias


def sin(x):
    if isinstance(x, Tensor):
        x.append_unary_op("sin")
        return math.sin(x.value)
    else:
        return math.sin(x)


def test():

    in_dim = 3
    p = Perceptron(in_features=in_dim)

    print(f"Pw is: {p.wheight}, {p.bias}")

    x = np.arange(in_dim)
    res = p.forward(x)

    res.backward()

    nx_graph = build_networkx_graph(res)
    draw_computation_graph(nx_graph, 2.)


if __name__ == "__main__":
    test()
