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
        if parents is not None:
            if isinstance(parents, list):
                if not all(map(lambda parent: isinstance(parent, Node), parents)):
                    raise Exception("All parents have to be other nodes")
        self.parents = parents
        self.child = child
        self.name = name
        self.id = None
        self.value = value
        self.grad = None

    def backward(self):
        raise NotImplementedError("")


class OpNode(Node):
    cnt = 0

    def __init__(self, parents, child, op) -> None:
        super().__init__(parents, child, op)
        self.operation = op
        self.result_name = f"f_{OpNode.cnt}"
        OpNode.cnt += 1
        # intermediate result, cached
        #self.value = None

    def backward(self):
        """
        Get the gradient of the output of this opnode w.r.t to all variable_nodes
        """
        gradient = {}

        def dfs(node: OpNode):
            # Some partial derivatices of common functions
            if node.operation == "+":
                node.grad = {
                    parent_node.id: 1. for parent_node in node.parents}
            elif node.operation == "*":
                node.grad = {parent_node.id: math.prod(
                    [other_parent_node.value for other_parent_node in node.parents if other_parent_node != parent_node]) for parent_node in node.parents}

            for parent in node.parents:
                if type(parent) is not OpNode:
                    continue
                dfs(parent)

        dfs(self)


class VariableNode(Node):
    def __init__(self, name, val) -> None:
        super().__init__(None, None, name, val)
        self.val = None


class ConstantNode(VariableNode):
    def __init__(self, name, val) -> None:
        super().__init__(name, val)


class ComputationGraph:
    def __init__(self) -> None:
        self.vertices = []
        self.edges = []
        # Only for drawing
        self.nx_graph = nx.DiGraph()

    def insert_vertex(self, v: Node):
        v_cnt = len(self.vertices)
        v.id = v_cnt
        # TODO we add the id to the name to make the NX nodes unique
        self.vertices.append(v)
        print(f"Adding node {v.name}")
        # attribute with the key "'name'" collides when converting to dot with pydot, thus we name id "ag_name"
        self.nx_graph.add_node(v.id, ag_name=v.name)

    def insert_edge(self, from_v: Node, to_v: Node):
        edge_cnt = len(self.edges)
        self.edges.append((from_v, to_v))
        print(f"Adding edge from {from_v.name} to {to_v.name}")
        self.nx_graph.add_edge(from_v.id, to_v.id, ag_name=f"e{edge_cnt}")

    def insert_new_vertex_with_edge(self, from_v: Node, to_v: Node):
        self.insert_vertex(to_v)
        self.insert_edge(from_v, to_v)

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

    def draw_edge_result_labels(self, node_positions, size):
        def get_edge_label(from_node_id, to_node_id):
            to_node: OpNode = self.vertices[to_node_id]
            from_node = self.vertices[from_node_id]
            from_node_name = from_node.result_name if type(
                from_node) is OpNode else from_node.name

            partial_d_value = to_node.grad[from_node_id]
            #return f"\\partial{{\\frac{{to_node.result_name}}{{from_node_name}}}} = " + "%.2f" % partial_d_value
            return f"d{to_node.result_name}/{from_node_name} = " + "%.2f" % partial_d_value
        result_labels = {(from_node, to_node): get_edge_label(from_node, to_node)
                         for from_node, to_node, data in self.nx_graph.edges(data=True) if type(self.vertices[to_node]) == OpNode}
        nx.draw_networkx_edge_labels(self.nx_graph, pos=node_positions, edge_labels=result_labels,
                                            font_color='black', font_size=size * 5,
                                            rotate=False)

    def draw_node_result_labels(self, node_positions, size):
        def op_node_label(node_id):
            node = self.vertices[node_id]
            if type(node) == OpNode:
                return f"{node.result_name}=" + "%.2f" % node.value
            else:
                return f"{node.name}=" + "%.2f" % node.value

        result_labels = {node: op_node_label(node)
                         for node, attributes in self.nx_graph.nodes(data=True)}
        op_nodes_positions = {node: (
            x, y-30) for node, (x, y) in node_positions.items()}

        nx.draw_networkx_labels(self.nx_graph, pos=op_nodes_positions, labels=result_labels,
                                font_color='black', font_size=size * 5, verticalalignment="baseline")


class ReverseModeDualNumber:

    comp_graph = ComputationGraph()

    def __init__(self, val: float, name, variable=True, parent=None) -> None:
        self.val = val
        self.name = name
        if variable:
            self.current_node = VariableNode(name, val=val)
        else:
            self.current_node = ConstantNode(name, val=val)
        if parent:
            ReverseModeDualNumber.comp_graph.insert_new_vertex_with_edge(
                parent, self.current_node)
        else:
            ReverseModeDualNumber.comp_graph.insert_vertex(
                self.current_node)

    def backward(self):
        self.current_node.backward()

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
        self.append_binary_op("+", other.current_node)
        self.val += other.val
        self.current_node.value = self.val
        return self

    def __radd__(self, other):
        return type(self).__add__(self, other)

    def __sub__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Sub with {type(other)} not supported")
        self.append_binary_op("-", other.current_node)
        self.val -= other.val
        self.current_node.value = self.val
        return self

    def __rsub__(self, other):
        return type(self).__sub__(self, other)

    def __mul__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Mul with {type(other)} not supported")
        self.append_binary_op("*", other.current_node)
        self.val *= other.val
        self.current_node.value = self.val
        return self

    def __rmul__(self, other):
        return type(self).__mul__(self, other)

    def __truediv__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Mul with {type(other)} not supported")
        self.append_binary_op("*", other.current_node)
        self.val / other.val
        self.current_node.value = self.val
        return self


class Perceptron():
    random_gen = np.random.default_rng(seed=123456)

    def __init__(self, in_features):
        self.in_features = in_features

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
        return math.sin(x.val)
    else:
        return math.sin(x)


def test():

    in_dim = 2
    p = Perceptron(in_features=in_dim)

    print(f"Pw is: {p.wheight}, {p.bias}")

    x = np.arange(in_dim)
    res = p.forward(x)

    res.backward()
    for v in ReverseModeDualNumber.comp_graph.vertices:
        print(f"V is {v.__dict__}")
    ReverseModeDualNumber.comp_graph.draw(size=2.)

    plt.show()


test()
