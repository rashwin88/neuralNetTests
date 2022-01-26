# All computational models
import numpy as np
from nn_utils import sanity_tests as st
from nn_utils import deep_initializers as di

# inspired by the assignments in Andrew Ng's course.
def linear_forward(A_prev, W, b):
    '''
    :param A: Actually refers to A(n-1), the activations from the previous layer.
    :param W: Weights of this specific layer
    :param b: constant of this specific layer
    :return:
    '''
    # perform the computations Z = WxA + b.
    # W is a matrix of size = (l X l-1) and A has size (l-1 X 1)
    # Where l represents the number of neurons in a specific layer.
    Z = np.dot(W,A_prev) + b
    cache = {
        'W' : W,
        'A_prev' : A_prev,
        'b' : b
    }
    # Z which is computed laters is cached in the forward activation function.
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    '''
    Performs the actual forward propogation step of linear transformation followed by activation on output from previous layer.
    :param A_prev:
    :param W:
    :param b:
    :param activation:
    :return:
    '''
    Z, cache = linear_forward(A_prev, W, b)
    A, activation_cache = di.allowable_activations().get(activation).get('forward')(Z)
    # append to the output cache from the activation cache.
    cache['Z'] = activation_cache.get('Z')
    assert st.activation_computation_check(cache['W'], cache['A_prev'], A),\
        "Shape Mismatch between matrices in layer, check dimensions of the weight and activation matrices"
    return A,cache

