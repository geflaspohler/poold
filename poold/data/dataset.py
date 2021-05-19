from abc import ABC, abstractmethod
import os

class Environment(ABC):
    """ Abstract dataset class for online learning """ 
    def __init__(self, times, **kwargs): 
        """ Initialize dataset.

        Args:
            time (list): list of prediction times
            models: representation of input models
        """
        self.times = times 
        self.T = len(times) # algorithm duration 

    @abstractmethod
    def get_loss(self, t, **kwargs):
        """ Get loss function at time t

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
