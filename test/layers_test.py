

import numpy as np
from nn_from_scratch.layers import Perceptron, Linear, Sequential, ReLu
from nn_from_scratch.autograd import Tensor


def test_perceptron():
    in_dim = 3
    p = Perceptron(in_features=in_dim)

    print(f"Perceptron params: {p.wheight}, {p.bias}")

    x = np.arange(in_dim)
    # We create a list of tensors to test the scalar case, normally we wouldn't do this
    x_t = [Tensor(np.array(x_i), f"x_{i}", is_variable=False)
           for i, x_i in enumerate(x)]
    res = p.forward(x_t)

    res.backward()

    draw_computation_graph(res, 2.)


def test_linear():

    in_dim = 3
    l = Linear(in_features=in_dim, out_features=2)

    print(f"Linear layer: w: {l.weight}, b: {l.bias}")
    params = l.get_parameters()
    print(f"params: {params}")

    x_t = Tensor(np.arange(in_dim), "x", is_variable=False)

    res = l.forward(x_t)

    print(f"Result: {res}")
    gradients = res.backward()
    print(f"Gradients: {gradients}")
    draw_computation_graph(res, 2.)


def test_sequential_model():

    in_dim = 28*28
    out_dim = 10
    intemediate_dim = 512
    nn_model = Sequential([
        Linear(in_dim, intemediate_dim),
        ReLu(),
        Linear(intemediate_dim, intemediate_dim),
        ReLu(),
        Linear(intemediate_dim, out_dim)
    ])
    nn_params = nn_model.get_parameters()
    print(f"NN params: {nn_params}")

    x_t = Tensor(np.arange(in_dim), "x", is_variable=False)

    start_ = time.perf_counter()
    res = nn_model.forward(x_t)

    end_ = time.perf_counter()
    print(f"Forward took {(end_ - start_) * 1000.}ms")

    print(f"Result: {res.value}")

    start_ = time.perf_counter()
    res.backward()
    end_ = time.perf_counter()
    print(f"Backward took {(end_ - start_) * 1000.}ms")
    draw_computation_graph(res, 2.)


test_perceptron()
test_linear()
test_sequential_model()
