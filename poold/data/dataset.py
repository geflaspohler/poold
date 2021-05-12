from abc import ABC, abstractmethod
import os

class Environment(ABC):
    """ Abstract dataset class for online learning """ 
    def __init__(self, times, models, **kwargs): 
        """ Initialize dataset.

        Args:
            time (list): list of prediction times
            models (list): list of expert model names
        """
        self.times = times 
        self.models = models
        self.T = len(times) # algorithm duration 

    @abstractmethod
    def get_gt(self, t, **kwargs):
        """ Get ground truth value at time t

        Args:
            t: a time represetnation 
        """
        pass

    @abstractmethod
    def get_pred(self, t, **kwargs):
        """  Get all model predictions and return a 
        merged set of predictions for a time t.

        Args:
            t: a time represetnation 
        """
        pass

    @abstractmethod
    def get_feedback(self, t, os_times=None, **kwargs):
        """  Get subset of oustanding feedback (os_times) 
        avaliable at time t. If os_times is None, get
        all avaliable feedback.

        Args:
            t: a time represetnation 
            os_times (list): list of outstanding feedback times
        """
        pass
