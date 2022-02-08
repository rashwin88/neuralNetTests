from nn_utils import deep_initializers as di
structure, param = di.initialize_parameters([(100,'input'),
                                             (1,  'sigmoid')], 'cross_entropy')

# forward prop one step
import numpy as np
from nn_utils import computation_modules as cm
m = cm.multilayer_forward(
    inputs = np.random.randn(100,10), # 100 features and 10 training examples
    parameters= param,
    nn_structure=structure
)

m = cm.multilayer_forward(
    inputs = m[0]['A'], # 100 features and 10 training examples
    parameters= param,
    nn_structure=structure
)

n = cm.compute_cost(parameters = m, nn_structure = structure,
                    actuals = np.random.randn(1,100))