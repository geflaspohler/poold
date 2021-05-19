from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd
import copy
import pdb 

from poold.utils import tic, toc, printf

class Hinter(ABC):
    """ Hinter abstract base class. """    
    def __init__(self, **kwargs):
        """Initialize hinter. """                
        pass

    @abstractmethod
    def get_hint(t):
        """ Algorithm specific parameter updates.

        Args:
            t: current time
        """
        pass
