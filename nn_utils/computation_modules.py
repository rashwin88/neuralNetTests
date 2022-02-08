# All computational models
import numpy as np
from nn_utils import sanity_tests as st
from nn_utils import deep_initializers as di
import copy

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

def multilayer_forward(inputs, parameters, nn_structure):
    '''
    Implements multilayer forward propogation upto the last layer
    :param inputs:
    :param parameters:
    :param nn_structure:
    :return:
    '''
    # we first initialize allowable activations
    activation_dict = di.allowable_activations()
    # we first create a deep copy of the parameters dictionary
    fp_stages = copy.deepcopy(parameters)
    # we then add the input layer directly to the fp_stages dictionary at key 0
    fp_stages[0]['A'] = inputs
    # now we simply loop thorugh the various layers with the forward prop
    layer_count = nn_structure.\
        get('overall_structure').\
        get('layer_count_excluding_input_layer')
    # set the current layer = 1
    current_layer = 1
    # begin looping
    while current_layer<= layer_count:
        # first perform the linear activation forward
        # Storing the cache from the forward pass as forward cache.
        fp_stages[current_layer]['A'], fp_stages[current_layer]['forward_cache'] = \
            linear_activation_forward(
                A_prev = fp_stages[current_layer - 1].get('A'),
                W = fp_stages[current_layer]['W'],
                b = fp_stages[current_layer]['b'],
                activation= nn_structure.\
                    get('detailed_structure')[current_layer].\
                        get('activation')
            )
        current_layer+=1
    return fp_stages


def compute_cost(parameters, nn_structure, actuals):
    '''
    Computes the cost given the parameter object after a forward pass and
    the actuals.
    :param parameters:
    :param actuals:
    :param cost_function:
    :return:
    '''
    fp_stages = copy.deepcopy(parameters)
    fp_stages['cost'] =\
        nn_structure.\
        get('overall_structure').\
        get('cost_function')(
            # For getting the predictions parameters --> Last Layer --> A
            predictions = parameters.get(
                nn_structure.\
                    get('overall_structure').\
                    get('layer_count_excluding_input_layer')
            ).get('A'),
            # Actuals are fed in as an input
            # Actuals are technically immutable they are final.
            actuals = actuals,
            # Size can be obtained from the shape of A in the parameters.
            size = parameters.get(0).get('A').shape[1]
            )

    return fp_stages





