import numpy as np
from nn_from_scratch.layers import Linear
from nn_from_scratch.autograd import Tensor
from nn_from_scratch.graph_drawing import build_and_draw_computation_graph

input_dimensions = 3
neural_net = Linear(in_features=input_dimensions, out_features=2)
params = neural_net.named_parameters()

x_input = Tensor(np.arange(input_dimensions))
y_output = neural_net.forward(x_input)
y_output.backward()

build_and_draw_computation_graph(y_output, 2.)
