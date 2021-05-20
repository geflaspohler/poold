""" Environment class to provide loss objects for online learning.

This abstract base class defines a template for an online learning 
Environment. The Environment object must implement two methods:
    * get_loss: gets a loss function object at time t
    * get_feedback: gets the set of incoming feedback at time t
Users should implement their own derived Environment classes that
overwrite these two abstract functions and perform whatever 
bookkeeping necessary to provide loss outputs.  

    For example:

    from poold import Environment

    class WeatherEnvironment(Environment):
        def __init__(self, times, delay):
            # Call base class constructor 
            super().__init__(times)

            self.D = delay 

        def get_loss(t):
            return get_weather(t)

        def get_feedback(self, t, os_times=None):
            if os_times is None:
                os_times = range(self.T)
            return [t_os for t_os os_times if t_os < t - self.D]
"""

from abc import ABC, abstractmethod

class Environment(ABC):
    """ Abstract dataset class for online learning """ 
    def __init__(self, times, **kwargs): 
        """ Initialize dataset.

        Args:
            time (list): list of prediction times
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
