import numpy as np
from nn_from_scratch.layers import Linear
from nn_from_scratch.autograd import Tensor
from nn_from_scratch.graph_drawing import build_and_draw_computation_graph

input_dimensions = 3
neural_net = Linear(in_features=input_dimensions, out_features=2)

print(f"Linear layer: w: {neural_net.weight}, b: {neural_net.bias}")
params = neural_net.get_parameters()
print(f"params: {params}")

x_input = Tensor(np.arange(input_dimensions), "x", requires_grad=False)

y_output = neural_net.forward(x_input)

print(f"Result: {y_output}")
y_output.backward()
gradients = [param.grad for param in neural_net.get_parameters().values()]
print(
    f"Gradients: {gradients}")
build_and_draw_computation_graph(y_output, 2.)
