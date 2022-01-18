# implements the relu activation function

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
    A = np.maximum(0,Z)
    cache = {
        'Z' : Z
    }
    assert A.shape == Z.shape, "A and Z are not the same shape in the forward activation"
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
    # here we make use of the forward prop to get the value of
    ## Section taken verbatim from https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week4/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/dnn_utils.py
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    # Notice that we are using element wise multiplication and not matric multiplications.
    dZ = dA * s * (1-s)
    assert dZ.shape == Z.shape, 'Issue in backprop with mismatched shape of foo and dfoo'
    return dZ

