import functools
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
    def __init__(self, parents, child, name: str, value=None) -> None:
        """
        value: float
        """
        if parents is not None:
            if isinstance(parents, list):
                if not all(map(lambda parent: isinstance(parent, Node), parents)):
                    raise Exception("All parents have to be other nodes")
        self.parents = parents
        self.child = child
        self.name = name
        self.id = None
        self.value = value


class OpNode(Node):
    cnt = 0

    def __init__(self, parents, child, op) -> None:
        super().__init__(parents, child, op)
        self.operation = op
        self.result_name = f"f_{OpNode.cnt}"
        OpNode.cnt += 1
        self.local_grad = None

    def backward(self):
        """
        Get the gradient of the output of this opnode w.r.t to all the parents. 
        This calculates the local gradient
        """
        assert len(self.parents) == 2
        # Some partial derivatices of common functions
        op1, op2 = self.parents[0], self.parents[1]
        both_scalar = (isinstance(op1.value, float)
                       and isinstance(op2.value, float))
        same_shape = (op1.value.shape == op2.value.shape)
        if self.operation == "+":
            assert both_scalar or same_shape
            if not isinstance(op1.value, float):
                self.local_grad = {
                    parent_node.id: np.ones((len(op1.value),)) for parent_node in self.parents}
            else:
                self.local_grad = {
                    parent_node.id: 1. for parent_node in self.parents}
        elif self.operation == "-":
            assert both_scalar or same_shape
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
            assert both_scalar or same_shape or (
                op1.value.ndim == 2 and op1.value.shape[0] == op1.value.shape[1] and op2.value.ndim == 1 and op1.value.shape[0] == op2.value.shape[0])

            if op1.value.ndim == 2 and op2.value.ndim == 1:
                vec_dim = op2.value.shape[0]
                # Partial derivative w.r.t to the Matrix is the vector repeated as rows of the matrix
                d_f_dA = np.broadcast_to(
                    op2.value, (op1.value.shape[0], len(op2.value)))
                # TODO correct ?
                d_f_dx = op1.value
                self.local_grad = {
                    self.parents[0].id: d_f_dA,
                    self.parents[1].id: d_f_dx}

            else:  # scalar and scalar product between vectors case
                self.local_grad = {op1.id: op2.value, op2.id: op1.value}

        elif self.operation == "/":
            assert both_scalar or same_shape
            self.local_grad = {
                op1.id: 1. / op2.value,
                op2.id: (-op1.value / op2.value ** 2)}
        else:
            raise Exception(f"Op {self.operation} not implemented")

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


class VariableNode(Node):
    def __init__(self, name, val: np.ndarray) -> None:
        super().__init__(None, None, name, val)
        self.val = val
        self.grad = None
        # String expression of the gradient, for debug
        self.grad_exp = None


class ConstantNode(VariableNode):
    def __init__(self, name, val: np.ndarray) -> None:
        super().__init__(name, val)


class ComputationGraph:
    def __init__(self) -> None:
        self.vertices = []
        self.edges = []
        # Only for drawing
        self.nx_graph = nx.DiGraph()
        self.current_root = None

    def insert_vertex(self, v: Node):
        v_cnt = len(self.vertices)
        v.id = v_cnt
        # TODO we add the id to the name to make the NX nodes unique
        self.vertices.append(v)
        print(f"Adding node {v.name}")
        # attribute with the key "'name'" collides when converting to dot with pydot, thus we name id "ag_name"
        self.nx_graph.add_node(v.id, ag_name=v.name)
        self.current_root = v.id

    def insert_edge(self, from_v: Node, to_v: Node):
        edge_cnt = len(self.edges)
        self.edges.append((from_v, to_v))
        print(f"Adding edge from {from_v.name} to {to_v.name}")
        self.nx_graph.add_edge(from_v.id, to_v.id, ag_name=f"e{edge_cnt}")

    def insert_new_vertex_with_edge(self, from_v: Node, to_v: Node):
        self.insert_vertex(to_v)
        self.insert_edge(from_v, to_v)

    def backward(self):
        gradient = {}

        def dfs(node: OpNode, accumulated_product=1.):
            # calculates local grad and stores it
            node.backward()
            for parent in node.parents:
                grad_elem = node.local_grad[parent.id]*accumulated_product

                if type(parent) is OpNode:
                    dfs(parent, grad_elem)
                elif type(parent) is VariableNode:
                    # Set the gradient, terminate search
                    if not parent.grad:
                        parent.grad = 0.
                    parent.grad += grad_elem
                    gradient[parent.name] = grad_elem

        dfs(self.vertices[self.current_root])
        print(f"Grad is: {gradient}")

    def draw(self, size=1.):
        def node_type_to_color(node: Node):
            if type(node) is VariableNode:
                return "#E6E600"
            elif type(node) is ConstantNode:
                return "#0044cc"
            elif type(node) is OpNode:
                return "#E6BF00"
        v_colors = [node_type_to_color(
            self.vertices[node]) for node, attributes in self.nx_graph.nodes(data=True)]
        node_positions = graphviz_layout(self.nx_graph, prog="dot")

        nx.draw_networkx_nodes(
            self.nx_graph, pos=node_positions, node_color=v_colors, edgecolors="#000000", node_size=size * 300)
        nx.draw_networkx_edges(
            self.nx_graph, pos=node_positions)
        node_labels = {node: attributes["ag_name"]
                       for node, attributes in self.nx_graph.nodes(data=True)}
        nx.draw_networkx_labels(self.nx_graph, pos=node_positions, labels=node_labels,
                                font_color='black', font_size=size * 7, font_weight="bold")
        self.draw_node_result_labels(node_positions, size)
        self.draw_edge_result_labels(node_positions, size)
        plt.show()

    def draw_edge_result_labels(self, node_positions, size):
        def get_edge_label(from_node_id, to_node_id):
            to_node: OpNode = self.vertices[to_node_id]
            from_node = self.vertices[from_node_id]
            from_node_name = from_node.result_name if type(
                from_node) is OpNode else from_node.name

            partial_d_value = to_node.local_grad[from_node_id]
            return f"d{to_node.result_name}/{from_node_name} = " + str(partial_d_value)
        result_labels = {(from_node, to_node): get_edge_label(from_node, to_node)
                         for from_node, to_node, data in self.nx_graph.edges(data=True) if type(self.vertices[to_node]) == OpNode}
        nx.draw_networkx_edge_labels(self.nx_graph, pos=node_positions, edge_labels=result_labels,
                                     font_color='black', font_size=size * 5,
                                     rotate=False)

    def draw_node_result_labels(self, node_positions, size):
        def op_node_label(node_id):
            node = self.vertices[node_id]
            if type(node) == OpNode:
                return f"{node.result_name}=" + str(node.value)
            else:
                return f"{node.name}=" + str(node.value)

        result_labels = {node: op_node_label(node)
                         for node, attributes in self.nx_graph.nodes(data=True)}

        def get_node_pos(node_id, pos):
            node = self.vertices[node_id]
            x, y = pos
            # Draw for op-nodes the result on the bottom
            if type(node) is OpNode:
                return (x, y-10)
            else:
                return (x, y+10)
        op_nodes_positions = {node_id: get_node_pos(
            node_id, pos) for node_id, pos in node_positions.items()}

        nx.draw_networkx_labels(self.nx_graph, pos=op_nodes_positions, labels=result_labels,
                                font_color='black', font_size=size * 5, verticalalignment="baseline")


class ReverseModeDualNumber:

    comp_graph = ComputationGraph()

    def __init__(self, value: np.ndarray, name, variable=True, parent=None) -> None:
        self.value = value
        self.name = name
        if variable:
            self.current_node = VariableNode(name, val=value)
        else:
            self.current_node = ConstantNode(name, val=value)
        if parent:
            ReverseModeDualNumber.comp_graph.insert_new_vertex_with_edge(
                parent, self.current_node)
        else:
            ReverseModeDualNumber.comp_graph.insert_vertex(
                self.current_node)

    def backward(self):
        ReverseModeDualNumber.comp_graph.backward()

    @classmethod
    def reset_graph(self):
        ReverseModeDualNumber.comp_graph = ComputationGraph()

    def append_op(self, name, *other_operands):
        node = OpNode(parents=[self.current_node, *
                      other_operands], child=None, op=name)
        ReverseModeDualNumber.comp_graph.insert_new_vertex_with_edge(
            self.current_node, node)
        for other_op in other_operands:
            ReverseModeDualNumber.comp_graph.insert_edge(
                other_op, node)
        self.current_node.child = node
        self.current_node = node

    def append_unary_op(self, name):
        self.append_op(name)

    def append_binary_op(self, name, other_v):
        self.append_op(name, other_v)

    def __add__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(
                f"Add with {other} of type {type(other)} not supported")
        assert other.value.shape == self.value.shape
        self.append_binary_op("+", other.current_node)
        self.value += other.value
        self.current_node.value = self.value
        return self

    def __radd__(self, other):
        return type(self).__add__(self, other)

    def __sub__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Sub with {type(other)} not supported")
        assert other.value.shape == self.value.shape
        self.append_binary_op("-", other.current_node)
        self.value -= other.value
        self.current_node.value = self.value
        return self

    def __rsub__(self, other):
        return type(self).__sub__(self, other)

    def __mul__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Mul with {type(other)} not supported")
        self.append_binary_op("*", other.current_node)
        if isinstance(self.value, float) and isinstance(other.value, float):
            self.value *= other.value
        else:
            self.value = np.matmul(self.value, other.value)
        self.current_node.value = self.value
        return self

    def __rmul__(self, other):
        return type(self).__mul__(self, other)

    def __truediv__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Mul with {type(other)} not supported")
        assert other.value.shape == self.value.shape
        self.append_binary_op("*", other.current_node)
        self.value / other.value
        self.current_node.value = self.value
        return self


class Perceptron():
    random_gen = np.random.default_rng(seed=123456)

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        k = 1. / in_features
        k_sqrt = math.sqrt(k)
        init_vals = Perceptron.random_gen.uniform(
            -k_sqrt, k_sqrt, in_features+1)
        self.wheight = [ReverseModeDualNumber(
            init_vals[i], name=f"w{i}") for i in range(self.in_features)]

        self.bias = ReverseModeDualNumber(init_vals[-1], name="b")

    def forward(self, x):
        assert len(x) == self.in_features
        x_dual_num = [ReverseModeDualNumber(
            x_i, f"x_{i}", variable=False) for i, x_i in enumerate(x)]

        dot_prod = x_dual_num[0] * self.wheight[0]
        for i, (x_i, w_i) in enumerate(zip(x_dual_num, self.wheight)):
            if i == 0:
                continue
            dot_prod += x_i * w_i
        return dot_prod + self.bias


def sin(x):
    if isinstance(x, ReverseModeDualNumber):
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

    ReverseModeDualNumber.comp_graph.draw(size=2.)

    plt.show()


if __name__ == "__main__":
    test()
