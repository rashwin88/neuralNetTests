# A sample NN structure for reference
# Here we use a simple model with 100 dimensional input with 200 input instances.
# Followed by 4 layers - RELU, RELU, RELU, SIGMOID
import numpy as np

# Represents the structure before initialization
sample_nn_structure = {
    'overall_structure' : {
        # Stores some overall structure details
        'layer_count_excluding_input_layer' : 4,
        'features_in_input_layer' : 100
    },

    'detailed_structure' : {
        0 : {
            'layer_number' : 0,
            'layer_dimensions' : 100,
            'activation' : 'input'
        },

        1 : {
            'layer_number' : 1,
            'layer_dimensions' : 50,
            'previous_layer_dimensions' : 100,
            'activation' : 'relu'
        },

        2 : {
            'layer_number' : 2,
            'layer_dimensions' : 50,
            'previous_layer_dimensions' : 50,
            'activation' : 'relu'
        },

        3 : {
            'layer_number' : 3,
            'layer_dimensions' : 50,
            'previous_layer_dimensions' : 50,
            'activation' : 'relu'
        },

        4 : {
            'layer_number' : 4,
            'layer_dimensions' : 1,
            'previous_layer_dimensions' : 50,
            'activation' : 'sigmoid'
        }

    }
}
