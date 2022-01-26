import numpy as np

def activation_computation_check(weights, prev_activations, current_activations):
    '''
    Checks to ensure that the current set of activations produced match the shape
    expectations of the layer.
    :param weights:
    :param prev_activations:
    :return: boolean result of the check
    '''
    result = False
    # defaulting to check failure
    if weights.shape[0] == current_activations.shape[0] and \
        weights.shape[1] == prev_activations.shape[0] :
        output = True
    return output

