# Neural networks with backprop from scratch 
Neural networks and backprop from scratch in Python with numpy.
Why ? Learning how backprop works and how/why the PyTorch API is designed.

PyTorch is of of course much more complex, so this repo is not supposed to be an re-implementation of PyTorch or even imply implementation details of PyTorch.  

# Features  

- Linear/Dense Layers
- ReLU, Sigmoid, TanH activation functions 
- MSELoss 
- Backprop/autograd for up to 2d-arrays
- Uniform weights initializer, same as pytorch
- Plotting of computational graphs including partial derivatives

# Examples 

[function_approximation.py](examples/function_approximation.py) 
![](images/function_approximation.png)

[Visualization of Compuational graph](examples/visualize_computational_graph.py)
![](images/computational_graph_linear_layer.png)

# Usage 

```python 
model = Sequential([
        Linear(in_features=10, out_features=30),
        ReLu(),
        Linear(in_features=30,
               out_features=30),
        ReLu(),
        Linear(in_features=30, out_features=1),
    ])
x_input = Tensor(np.arange(10))
y_output = model.forward(x_input)
y_output.backward()
```
# Install/Dependencies 

As no proper installer is available for graphviz, on Windows you have to download, unpack the folder anywhere and add the bin subfolder to the PATH environment variable such that the drawing of the graph works.

```
pip install -e .
```


# Run tests 

To run the tests, pytorch is required as dependency to suit as a reference. 
```
python -m unittest
```

# TODO 

# Fixes 

- Make backward-functions for each operator, reference the function object instead of large switch-case 
- [] Refactor to function classes which store the operands and derive from node. Do not use tensor for functions (arch from micrograd)
- [] implement matmul
- [] Remove ids of nodes, use list as parents
- [] Clean up tensor-API, do not require names
- [] Fix global states in node if any
- [] Impl leaky ReLU
- [X] Fix NaNs

# Features

- [x] matrix-vector backprop
- [] batching 
- [x] matrix-matrix backprop
- [] Add fashion mnist example
- [] Add viz example 
- [] Add more tests, refactor current tests 


# Contributing 

Contributions are greatly appreciated, especially corrections regarding backprop, vectorization and refactorings/simplifications to improve readability of the code and/or consistency with pytorch. 
Also better visualization, documentation and more examples are appreciated. But this repo is written for educational purposes and thus speed optimizations for the cost of code complexity are not desired.

# Reference: 

- [The excellent course CS231n from Andrej Karpathy](https://www.youtube.com/watch?v=i94OvYb6noo)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b

# License 

MIT 

# Similar projects 

- [tinygrad](https://github.com/geohot/tinygrad)
- [micrograd](https://github.com/karpathy/micrograd)