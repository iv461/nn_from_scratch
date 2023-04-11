# Neural networks with backprop from scratch 

API is designed to be similar to pytorch, but in general the pytorch framework is much more complex, so the internals in this repo are not supposed to be a simple pytorch. 

# Features  

- Linear/Dense Layers
- ReLU, Sigmoid, TanH activation functions 
- MSELoss 
- Backprop/autograd for arbitrary tensors, with vectorization
- Uniform weights initializer, same as pytorch
- Plotting of computational graphs including partial derivatives
- based on numpy

# Dependencies 

- networkx 
- matplotlib 3.5.0, this version is needed due to this bug in NX
- graphviz
- pydot

As no proper installer is available for graphviz, on Windows you have to download, unpack the folder anywhere and add the bin subfolder to the PATH environment variable such that the drawing of the graph works.

# Examples 

[function_approximation](function_approximation.py) 
TODO image 

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
- [] matrix-matrix backprop
- [] Add fashion mnist example
- [] Add viz example 
- [] Add more tests, refactor current tests 


# Contributing 

Contributions are greatly appreciated, especially corrections regarding backprop, vectorization and refactorings/simplifications to improve readability of the code and/or consistency with pytorch. 
Also better visualization, documentation and more examples are appreciated. Keep in mind, that this repo is written for education and thus optimizations, CUDA implementation are probably not well 

# Reference: 

- [The excellent course CS231n from Andrej Karpathy](https://www.youtube.com/watch?v=i94OvYb6noo)
- Pytorch Docs
- https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b

# License 

MIT 

# Similar projects 

- [tinygrad](https://github.com/geohot/tinygrad)
- [micrograd](https://github.com/karpathy/micrograd)