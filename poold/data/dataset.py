from abc import ABC, abstractmethod
import os

class Dataset(ABC):
    """ Abstract dataset class for online learning """ 
    def __init__(self, targets, models, **kwargs): 
        """ Initialize dataset.

        Args:
            targets (list): list of target prediction times
            models (list): list of expert model names
        """
        self.targets = targets
        self.models = models

    @abstractmethod
    def get_model(self, target, model, **kwargs):
        """ Get model prediction for a target time

        Args:
            target: a target represetnation 
            model: a model name representation
        
        Returns: 
        """
    pass

    @abstractmethod
    def get_pred(self, target, **kwargs):
        """  Get all model predictions and return a 
        merged set of predictions for a target.

        Args:
            target: a target represetnation 
        """
    pass
