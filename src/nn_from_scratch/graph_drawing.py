from typing import Union
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from nn_from_scratch.autograd import Tensor


def build_networkx_graph(root_node: Tensor) -> nx.DiGraph:
    """
    Builds a networkx computation graph from the tensor tree, for debug visualization.
    Args:
        root_node (Tensor): 

    Returns:
        nx.DiGraph: the build networkx graph with the tensors as node data, available with the key "ag_tensor" 
    """
    nx_graph = nx.DiGraph()

    def dfs(node: Tensor):
        # We have to put the node objects as data, otherwise the graphviz_layout function throws an exception, it seems that this is a bug in networkx.
        nx_graph.add_node(node.id, ag_tensor=node)
        if not node.parents:
            return
        for parent in node.parents:
            nx_graph.add_edge(parent.id, node.id)
            dfs(parent)
    dfs(root_node)
    return nx_graph


def build_and_draw_computation_graph(root_tensor: Tensor, size=1.):
    """
    Draw the computation graph for a given root tensor and opens a window.
    Can be called before or after the backward pass (calling backward()). If gradients are available, they will be plotted as well.
    Args:
        root_tensor (Tensor): the root tensor which is the result of a computation. Can also be called on tensors which do not result from a computation, in this case only one node will be plotted.
        size (_type_, optional): Scale. Defaults to 1..

    Returns:
        _type_: _description_
    """

    nx_graph = build_networkx_graph(root_tensor)

    def node_type_to_color(node: Tensor):
        if node.operation:
            return "#E6BF00"
        elif node.requires_grad:
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


def value_to_string(value: Union[np.ndarray, float, None], max_tensor_size_to_print):
    if isinstance(value, float):
        value_string = f"{value}:.2f"
    elif isinstance(value, np.ndarray):
        # Do not plot big matrices
        if value.size > max_tensor_size_to_print:
            value_string = f"Shape {value.shape}"
        else:
            value_string = str(value)
    else:
        value_string = "Unknown"
        #raise Exception(f"Unknown type of gradient: {type(value)}")
    return value_string


def draw_edge_result_labels(nx_graph, node_positions, size, max_tensor_size_to_print=15):

    def get_edge_label(from_node: Tensor, to_node: Tensor):
        from_node, to_node = get_tensor_from_id(nx_graph,
                                                from_node), get_tensor_from_id(nx_graph, to_node)
        # Enable drawing before the backward pass
        if to_node.local_grad is not None:
            partial_d = to_node.local_grad[from_node.id]
        else:
            partial_d = None
        partial_d_str = value_to_string(
            partial_d, max_tensor_size_to_print)

        return f"d{to_node.name}/{from_node.name} = " + partial_d_str

    result_labels = {(from_node, to_node): get_edge_label(from_node, to_node)
                     for from_node, to_node in nx_graph.edges() if get_tensor_from_id(nx_graph, to_node).operation}

    nx.draw_networkx_edge_labels(nx_graph, pos=node_positions, edge_labels=result_labels,
                                 font_color='black', font_size=size * 5,
                                 rotate=False)


def draw_node_result_labels(nx_graph, node_positions, size, max_tensor_size_to_print=15):
    def op_node_label(node: Tensor):
        value_str = value_to_string(
            node.value, max_tensor_size_to_print)
        grad_str = value_to_string(
            node.grad, max_tensor_size_to_print)
        return f"{node.name}={value_str}, grad: {grad_str}"
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
