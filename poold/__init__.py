# from .learners import DORM, DORMP, AdaHedgeD
from .learners import AdaHedgeD 
from .run import run

def create(learner, loss, T, model_list, partition, **kwargs):
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
    if alg == "dorm":
        oe = DORM(loss, T, model_list, partition)  
    elif alg == "dormp":
        oe = DORMP(loss, T, model_list, partition)  
    elif alg == "adahedged":
        oe = AdaHedgeD(loss, T, model_list, partition, reg="adahedged")  
    elif alg == "dub":
        oe = AdaHedgeD(loss, T, model_list, partition, reg="dub")  
    else: 
        raise ValueError(f"Unknown learning algorithm {alg}.")
    return oe