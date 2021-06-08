from .learners import AdaHedgeD, DORM, DORMPlus
from .environment import Environment 

def create(learner, model_list, groups=None, T=None, **kwargs):
    """
    Returns an online_expert object, instantiated with the passed in parameters.

    Args:
        learner (str): online learning algorithm name [dorm | dormp | adahedged | dub]
        loss (OnlineLoss): an OnlineLoss object
        T (int): > 0, algorithm duration
        model_list (list): list of indicating expert model names
        groups (numpy.array): mask grouping learners for different 
            delay periods into separate simpilces,
            e.g., np.array([1, 1, 2, 3, 3]) 
            corresponds to models[0:2] playing on one simplex,
            models[2] playing on another, and models[3:] playing 
            on the final simplex. Ususally set to None to perform
            single-simplex hinting.

    Returns:
        ol (OnlineLearner): online learning object
    """
    if learner == "dorm":
        ol = DORM(model_list, groups, T)  
    elif learner == "dormplus":
        ol = DORMPlus(model_list, groups, T) 
    elif learner == "adahedged":
        ol = AdaHedgeD(model_list, groups, T, reg="adahedged")  
    elif learner == "dub":
        ol = AdaHedgeD(model_list, groups, T, reg="dub")  
    else: 
        raise ValueError(f"Unknown learning algorithm {learner}.")

    return ol