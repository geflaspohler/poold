# from .learners import DORM, DORMP, AdaHedgeD
from .learners import AdaHedgeD 
from .environment import Environment 
from .run import run

def create(learner, model_list, partition=None, T=None, **kwargs):
    """
    Returns an online_expert object, instantiated with the passed in parameters.

    Args:
        learner (str): online learning algorithm name [dorm | dormp | adahedged | dub]
        loss (OnlineLoss): an OnlineLoss object
        T (int): > 0, algorithm duration
        model_list (list): list of indicating expert model names
        partition (numpy.array): mask partitioning learners for different 
            delay periods into separate simpilces,
            e.g., np.array([1, 1, 2, 3, 3]) 

    Returns:
        oe (OnlineLearner): online learning object
    """
    if learner == "dorm":
        oe = DORM(model_list, partition, T)  
    elif learner == "dormp":
        oe = DORMP(model_list, partition, T) 
    elif learner == "adahedged":
        oe = AdaHedgeD(model_list, partition, T, reg="adahedged")  
    elif learner == "dub":
        oe = AdaHedgeD(model_list, partition, T, reg="dub")  
    else: 
        raise ValueError(f"Unknown learning algorithm {learner}.")
    return oe