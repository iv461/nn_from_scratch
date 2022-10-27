import numpy as np
import math

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

    def __init__(self, value: np.ndarray, name: Union[str, None], is_variable=True, parents=None, op=None) -> None:
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
        # The gradient vecto
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

    def calc_local_grad_binary_ops(self):
        """
        Calculates the "local" partial derivatives of this function, for common binary operations like +, -, * etc.
        """
        assert len(self.parents) == 2

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
        """
        Calculates the "local" partial derivatives of this function, 
        for common unary operations like unary sum (including single operand case), mean and square
        """
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
        Runs backprop, returns dict of the Tensor names as dict
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
        partial_d = to_node.local_grad[from_node.id]
        partial_d_str = "%.2f" % partial_d if isinstance(
            partial_d, float) else str(partial_d)
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
        return f"{node.name}={value_str}"
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
