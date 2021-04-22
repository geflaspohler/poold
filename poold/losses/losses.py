from abc import ABC, abstractmethod
import numpy as np
import copy
import pdb 
from sklearn.metrics import mean_squared_error
from poold.utils import tic, toc, printf

class OnlineLoss(ABC):
    """ Abstract Loss object for online learning. """ 
    def __init__(self):
        pass

    def loss_regret(self, partition, partition_keys, g, w):
        ''' Computes the loss regret w.r.t. a partition of the 
        weight vector.

        Args:
            partition (np.array): masks for partitions of the vectors
                e.g., np.array([1, 1, 2, 3, 3])
            partition_keys (list): unique partition keys
            g (np.array): gradient vector
            w (np.array): weight vector
        '''
        regret = np.zeros(g.shape)
        for k in partition_keys:
            p_ind = (partition == k)
            regret[p_ind] = np.dot(g[p_ind], w[p_ind]) - g[p_ind] 
        return regret

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def loss_experts(self, *args, **kwargs):
        pass

    @abstractmethod
    def loss_gradient(self, *args, **kwargs):
        pass

class SELoss(OnlineLoss): 
    """ Squared error loss || y - y_hat ||_2^2 """
    def __init__(self):
        pass

    def loss(self, y, y_hat):
        """Computes the L2 squared loss for a prediction. 

        Args:
           y (np.array): d x 1, true value 
           y_hat (np.array): d x 1, predicted value 
        """     
        return 0.5 * np.linalg.norm(y -y_hat, ord=2)**2
    
    def loss_experts(self, X, y):
        """Computes the SE loss for each expert model. 

        Args:
           X (np.array): d x n, prediction of d values from n experts        
           y (np.array): d x 1, ground truth value at d points
        """    
        # Unpack arguments
        n = X.shape[1] 
        return 0.5 * np.sum(
            (X - np.matlib.repmat(y.reshape(-1, 1), 1, n))**2, axis=0)
    
    def loss_gradient(self, **kwargs):
        """Computes the gradient of the SE loss of ||Xw - y||_2^2

        Args:
           X (np.array): d x n, prediction from n experts
           y (np.array): d x 1, ground truth 
           w (np.array): n x 1, predictive weights
        """
        # Unpack arguments
        d = X.shape[0] # Number of gradient values 
        n = X.shape[1] # Number of experts
        
        err = X @ w - y
        return (X.T @ err).reshape(-1,)
