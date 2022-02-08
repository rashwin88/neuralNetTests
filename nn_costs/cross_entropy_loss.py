import numpy as np

def cost_function(predictions, actuals, size):
    '''
    Computes cross entropy loss
    :param predictions:
    :param actuals:
    :return:
    '''
    assert actuals.shape[1] == predictions.shape[1],\
          'Actuals and Predictions donot have the same number of data points'

    cost = (-1/size) *\
           np.sum(
               np.multiply(actuals,np.log(predictions)) +\
               np.multiply((1-actuals), np.log(1-predictions))
           )

    return cost