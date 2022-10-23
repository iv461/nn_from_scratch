import functools
from lib2to3.pgen2.token import OP
import networkx as nx
import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt

from dataclasses import dataclass


class DualNumber:

    def __init__(self, _x, _dx=0.):
        if type(_x) is DualNumber:
            self.x = _x.x
            self.dx = _x.dx
        else:
            self.x = float(_x)
            self.dx = _dx

    def __add__(self, other):
        # copyes if other is a dual or creates a dual number if other is a constant
        other_dual = DualNumber(other)
        return DualNumber(self.x + other_dual.x, self.dx + other_dual.dx)

    def __radd__(self, other):
        return type(self).__add__(self, other)

    def __sub__(self, other):
        # copyes if other is a dual or creates a dual number if other is a constant
        other_dual = DualNumber(other)
        return DualNumber(self.x - other_dual.x, self.dx - other_dual.dx)

    def __rsub__(self, other):
        return type(self).__sub__(self, other)

    def __mul__(self, other):
        # copyes if other is a dual or creates a dual number if other is a constant
        other_dual = DualNumber(other)
        return DualNumber(self.x * other_dual.x, other_dual.x * self.dx + other_dual.dx * self.x)

    def __rmul__(self, other):
        return type(self).__mul__(self, other)

    def __truediv__(self, other):
        # copyes if other is a dual or creates a dual number if other is a constant
        other_dual = DualNumber(other)
        other_x_inv = 1. / other_dual.x
        new_x = self.x * other_x_inv
        # product rule
        new_dx = (self.dx - other_dual.dx * new_x) * other_x_inv
        return DualNumber(new_x, new_dx)


class Node:
    def __init__(self, parents, child, name) -> None:
        self.parents = parents
        self.child = child
        self.name = name
        self.id = None

    def is_root(self):
        return self.parents is None

    def is_leaf(self):
        return self.child is None


class OpNode(Node):
    def __init__(self, parents, child, op) -> None:
        super().__init__(parents, child, op)
        self.operation = op


class VariableNode(Node):
    def __init__(self, name) -> None:
        super().__init__(None, None, name)


class ConstantNode(VariableNode):
    def __init__(self, name) -> None:
        super().__init__(name)


class ComputationGraph:
    def __init__(self) -> None:
        self.vertices = []
        self.edges = []
        self.nx_graph = nx.DiGraph()

    def insert_vertex(self, v: Node):
        v_cnt = len(self.vertices)
        v.id = v_cnt
        # TODO we add the id to the name to make the NX nodes unique
        if type(v) is OpNode:
            v.name = f"{v.name}_{v.id}"
        self.vertices.append(v)
        print(f"Adding node {v.name}")
        self.nx_graph.add_node(v.name, id=v_cnt)

    def insert_edge(self, from_v: Node, to_v: Node):
        edge_cnt = len(self.edges)
        self.edges.append((from_v, to_v))
        print(f"Adding edge from {from_v.name} to {to_v.name}")
        self.nx_graph.add_edge(from_v.name, to_v.name, name=f"e{edge_cnt}")

    def insert_new_vertex_with_edge(self, from_v: Node, to_v: Node):
        self.insert_vertex(to_v)
        self.insert_edge(from_v, to_v)

    def draw(self):
        def node_type_to_color(node: Node):
            if type(node) is VariableNode:
                return "#E6E600"
            elif type(node) is ConstantNode:
                return "#0044cc"
            elif type(node) is OpNode:
                return "#E6BF00"
        v_colors = [node_type_to_color(
            self.vertices[attributes["id"]]) for node, attributes in self.nx_graph.nodes(data=True)]
        pos = nx.planar_layout(self.nx_graph)
        nx.draw_networkx_nodes(
            self.nx_graph, pos=pos, node_color=v_colors, edgecolors="#000000", node_size=300)
        nx.draw_networkx_edges(
            self.nx_graph, pos=pos)
        nx.draw_networkx_labels(self.nx_graph, pos=pos, font_color='w')


class ReverseModeDualNumber:

    comp_graph = ComputationGraph()

    def __init__(self, val: float, name, variable=True, parent=None) -> None:
        self.val = val
        self.name = name
        if variable:
            self.current_node = VariableNode(name)
        else:
            self.current_node = ConstantNode(name)
        if parent:
            ReverseModeDualNumber.comp_graph.insert_new_vertex_with_edge(
                parent, self.current_node)
        else:
            ReverseModeDualNumber.comp_graph.insert_vertex(
                self.current_node)

    def backward(self):
        pass

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
        return self

    def __radd__(self, other):
        return type(self).__add__(self, other)

    def __sub__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Sub with {type(other)} not supported")
        self.append_binary_op("-", other.current_node)
        self.val -= other.val
        return self

    def __rsub__(self, other):
        return type(self).__sub__(self, other)

    def __mul__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Mul with {type(other)} not supported")
        self.append_binary_op("*", other.current_node)
        self.val *= other.val
        return self

    def __rmul__(self, other):
        return type(self).__mul__(self, other)

    def __truediv__(self, other):
        if not isinstance(other, ReverseModeDualNumber):
            raise Exception(f"Mul with {type(other)} not supported")
        self.append_binary_op("*", other.current_node)
        self.val / other.val
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

    in_dim = 4
    p = Perceptron(in_features=in_dim)

    print(f"Pw is: {p.wheight}, {p.bias}")

    x = np.arange(in_dim)
    res = p.forward(x)

    ReverseModeDualNumber.comp_graph.draw()
    plt.show()


reverse_mode_test()
