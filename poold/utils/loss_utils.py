import numpy as np
import copy
import pdb

def loss_regret(g, w, groups):
    ''' Computes the loss regret w.r.t. a grouping of the 
    weight vector using loss gradient.

    Args:
        g (np.array): gradient vector
        w (np.array): weight vector
        groups (np.array): mask grouping learners 
            for different tasks into separate simplicies,
            e.g., np.array([1, 1, 2, 3, 3]) means to use
            model_list[0:2] for the first task,
            model_list[2] for the second task,
            model_list[3:] for the third task
    '''
    group_keys = list(set(groups))
    regret = np.zeros(g.shape)
    for k in group_keys:
        p_ind = (groups == k)
        regret[p_ind] = np.dot(g[p_ind], w[p_ind]) - g[p_ind] 
    return regret

def normalize_by_groups(w, groups):
    """ Normalize weight vector by groups.

    Args:
        w (np.array): weight vector 
        groups (np.array): mask grouping learners 
            for different tasks into separate simplicies,
            e.g., np.array([1, 1, 2, 3, 3]) means to use
            model_list[0:2] for the first task,
            model_list[2] for the second task,
            model_list[3:] for the third task
    """
    wout = np.zeros(w.shape)
    group_keys = list(set(groups))
    for k in group_keys:     
        p_ind = (groups == k)                             
        if np.sum(w[p_ind]) > 0.0:
            wout[p_ind] = (w[p_ind]/ np.sum(w[p_ind]))
        else: 
            n_k = sum(p_ind)
            wout[p_ind] = 1./n_k * np.ones(n_k,) # Uniform
    return wout
    