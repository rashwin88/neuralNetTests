# implements the sigmoid activation function

# We cache Z on the forward activation.
import numpy as np

def forward(Z):
    '''
    This is a vectorized function, so a linear forward output is taken and activations is computed using a sigmoid function
    :param Z: Z is an array of any shape.
    :return:
    '''
    #bradcasting across the array.
    # Can even be used for multidimensional arrays.
    A = 1/(1+np.exp(-Z))
    cache = {
        'Z':Z
    }
    return A, cache


def backward(dA, cache):
    '''
    Here the parameter dA comes computed from the previous layer. It is the partial deivative of the loss w.r.t the current set of activations
    for this layer.
    :param dA:
    :param cache: We need the cache from the forward prop on the same layer. The cache simply contains the value of Z.
    :return:
    '''

    Z = cache['Z']
    # here we make use of the forward prop ro get the value of s (this is basically A(1-A) in the case of sigmoid
    s, temp_cache = forward(Z)
    # Notice that we are using element wise multiplication and not matric multiplications.
    dZ = dA * s * (1-s)
    assert dZ.shape == Z.shape, 'Issue in backprop with mismatched shape of foo and dfoo'

    return dZ

