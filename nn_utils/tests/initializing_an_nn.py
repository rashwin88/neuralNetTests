from nn_utils import deep_initializers as di
structure, param = di.initialize_parameters([(100,'input'),
                                             (10,'sigmoid'),
                                             (20, 'relu'),
                                             (1,  'sigmoid')])
