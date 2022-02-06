# Here we initialize the weights for a set of layers in a neural network.
# To do this initializtion, we need number of layers followed by the number of neurons in each layer (dimensions of layer)
# Will be following the conventions used by Andrew Ng in his NN course.

# first we create a function that can form the structure of the input given to the initialization function.
# The idea is to input the neural network parameters as a dictionary keyed at the layer level
# The input to this helper function will be a list of tuples which will contain parameters of the neural network

import numpy as np
from nn_activations import relu_activation as relu
from nn_activations import sigmoid_activation as sigm

def allowable_activations():
    '''
    Returns a list of allowable activation codes
    :return:
    '''
    # Forming as a dictionary to store any functions related to these activations
    output ={
        'relu':{'forward': relu.forward, 'backward' : relu.backward},
        'sigmoid' : {'forward': sigm.forward, 'backward' : sigm.backward},
        'input' : {}
    }
    return(output)

def create_nn_structure(input_list):
    '''
    Creates a nn structure dictionary
    :param input_list: should contain the input layer also.
    :return: nn_structure_dictionary
    '''
    nn_structure_dictionary = {}
    # using this space to define some overall structure for the neural network
    nn_structure_dictionary['overall_structure'] = {}
    nn_structure_dictionary['overall_structure'].update(layer_count_excluding_input_layer = len(input_list) - 1)
    nn_structure_dictionary['overall_structure'].update(features_in_input_layer = input_list[0][0])
    # Defining the actual structure of the deep neural network
    nn_structure_dictionary['detailed_structure'] = {}
    # initialize a layer tracker
    layer_tracker = 0
    for layer in input_list:
        nn_structure_dictionary['detailed_structure'][layer_tracker] = {}
        nn_structure_dictionary['detailed_structure'][layer_tracker].update(layer_number = layer_tracker)
        nn_structure_dictionary['detailed_structure'][layer_tracker].update(layer_dimensions = layer[0])
        # Adding dimensions of previous layer here as it may be useful in parameter initialization
        if layer_tracker == 0:
            pass
        else:
            nn_structure_dictionary['detailed_structure'][layer_tracker].update(prev_layer_dimensions=input_list[layer_tracker-1][0])
        nn_structure_dictionary['detailed_structure'][layer_tracker].update(activation=layer[1])
        layer_tracker+=1
    # perform all necessary checks
    for k,v in nn_structure_dictionary['detailed_structure'].items():
        assert isinstance(v.get('layer_dimensions'), int), 'Layer dimensions for layer ' + str(k) + ' should be an integer'
        assert isinstance(v.get('activation'), str), 'ACtivation for layer ' + str(k) + ' should be specified as a string'
        assert v.get('activation') in allowable_activations().keys(), 'Activation ' + str(v.get('activation')) + ' is not allowed'
        if k == 0:
            assert v.get('activation') == 'input', 'First layer must have activation set as input'
    return(nn_structure_dictionary)

def initialize_parameters(input_list):
    '''
    Here we provide the perviously defined tuple type input for the structure of the neural network.
    A Call is made to create_nn_structure to generate the structure first and then initialize the parameters.
    One idea is to store the values of the parameters in the layer itself, this may be able to act as a useful cache.
    But it may be more useful to create a parameter object to store data.
    :param input_list:
    :return: a tuple of structure and initialized parameters for that structure.
    '''
    structure = create_nn_structure(input_list)
    # creating an empty parameters dictionary
    parameters = {}
    for k,v in structure.get('detailed_structure').items():
        parameters[k] = {}
        if k == 0:
            # there is nothing to initialize if we are at an input layer
            pass
        else:
            parameters[k] = {}
            parameters[k]['W'] = np.random.randn(v.get('layer_dimensions'), v.get('prev_layer_dimensions'))*0.01
            parameters[k]['b'] = np.random.randn(v.get('layer_dimensions'), 1)
    return structure, parameters







