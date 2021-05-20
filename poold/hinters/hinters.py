""" Hinter class to provide optimistic hint objects 
for online learning.

This abstract base class defines a template for an online learning 
Hinter. The Hinter object must implement two methods:
    * get_loss: gets a loss function object at time t
    * get_feedback: gets the set of incoming feedback at time t
Users should implement their own derived Environment classes that
overwrite these two abstract functions and perform whatever 
bookkeeping necessary to provide loss outputs.  

    For example:

    from poold import Hinter 

    class WeatherHinter(Hinter):
        def __init__(self, hint_type="prev_gradient):
            self.hint_type = hint_type

        def get_hint(self, t):
            return self.get_weather_hint(t, self.hint_type)
"""

from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd

class Hinter(ABC):
    """ Hinter abstract base class. """    
    def __init__(self, **kwargs):
        """Initialize hinter. """                
        pass

    @abstractmethod
    def get_hint(self, t, os_times, **kwargs):
        """ Get hint at time t.

        Args:
            t: current time
            os_times (list): outstanding feedback times

        Returns: hint object
        """
        pass
