import autograd.numpy as ag_np
import math

from layers import Linear
from losses import mse_loss
from train import gradient_descent
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np 

def vectorize(f):
    def v(x):
        out = []
        for x_ in x:
            out.append(f(x_))
        return out
    return v

def qubic_fitting_problem():
    ...

def gradient_descent_demo():

    def target_f(x):
        return ag_np.sin(x)+ .5 * ag_np.exp(x-1.)

    space_to_show = [-5, 5]
    pts = np.linspace(*space_to_show, num=100)
    plt.plot(pts, vectorize(target_f)(pts), label='f(x)')
    #plt.plot(pts, df(pts), label='df/dx')
    plt.legend()
    plt.xlim(tuple(space_to_show))
    plt.ylim((-5, 5))

    
    x_iter = 2.5
    steps = []
    for i in range(100):
        new_x = gradient_descent(x_iter, target_f, .1)

        step = abs(new_x - x_iter)
        steps.append(new_x)
        if step < .00001:
            print(f"GD took {i} iterations")
            break

        print(f"new x is: {new_x}")
        x_iter = new_x
    
    print(f"Minimum is at: {x_iter}")
    plt.scatter(steps, vectorize(target_f)(steps), label='Gradient descent', c="g")
    plt.show()


gradient_descent_demo()