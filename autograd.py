import functools
import networkx as nx
import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

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

    def is_root(self):
        return self.parents is None

    def is_leaf(self):
        return self.child is None


class OpNode(Node):
    def __init__(self, parents, child, op) -> None:
        super().__init__(parents, child, op)
        self.operation = op
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
                node.grad =  [1., 1.]
            elif node.operation == "*":
                node.grad =  [self.parents[1].value, self.parents[0].value]
        

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
        pos = graphviz_layout(self.nx_graph, prog="dot")

        nx.draw_networkx_nodes(
            self.nx_graph, pos=pos, node_color=v_colors, edgecolors="#000000", node_size=size * 300)
        nx.draw_networkx_edges(
            self.nx_graph, pos=pos)
        node_labels = {node: attributes["ag_name"]
                       for node, attributes in self.nx_graph.nodes(data=True)}
        nx.draw_networkx_labels(self.nx_graph, pos=pos, labels=node_labels,
                                font_color='black', font_size=size * 7, font_weight="bold")

        result_labels = {node: "%.2f" % self.vertices[node].value
                         for node, attributes in self.nx_graph.nodes(data=True)}
        node_labels_pos = {node: (x, y-30) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(self.nx_graph, pos=node_labels_pos, labels=result_labels,
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
    if isinstance(x, DualNumber):
        # mind the chain rule
        return DualNumber(math.sin(x.x), math.cos(x.x) * x.dx)
    elif isinstance(x, ReverseModeDualNumber):
        x.append_unary_op("sin")
        return math.sin(x.val)
    else:
        return math.sin(x)

# sin for Dualnumbers


def cos(x):
    if type(x) is DualNumber:
        # mind the chain rule
        return DualNumber(math.cos(x.x), -math.sin(x.x) * x.dx)
    else:
        return math.cos(x)


class dx:

    def __init__(self, function):
        self.function = function

    def __call__(self, x):
        dual_argument = None
        if type(x) is DualNumber:
            dual_argument = x
        else:
            dual_argument = DualNumber(x, 1.)
        ret = self.function(dual_argument)
        return


def reverse_mode_test():

    in_dim = 9
    p = Perceptron(in_features=in_dim)

    print(f"Pw is: {p.wheight}, {p.bias}")

    x = np.arange(in_dim)
    res = p.forward(x)

    ReverseModeDualNumber.comp_graph.draw(size=2.)

    plt.show()


reverse_mode_test()
