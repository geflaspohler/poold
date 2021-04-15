import numpy as np

def neg_entropy_reg(w):
    """ Negative entropy regularizer, shifted to min of zero. """
    return np.sum(w[w>0.0] * np.log(w[w>0.0])) + np.log(self.d)
            
