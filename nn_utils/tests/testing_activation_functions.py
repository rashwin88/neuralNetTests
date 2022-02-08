import numpy as np
from nn_utils import computation_modules as cm
g, cache = cm.linear_activation_forward(
    A_prev = np.random.randn(5,1),
    W = np.random.randn(4,5),
    b = np.random.randn(1,1),
    activation = 'sigmoid'
)


