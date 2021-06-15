import numpy as np
import copy

def loss_regret(g, w, groups):
    ''' Computes the loss regret w.r.t. a grouping of the 
    weight vector using loss gradient.

    Args:
        g (np.array): gradient vector
        w (np.array): weight vector
        groups (numpy.array): mask grouping learners for different 
            delay periods into separate simpilces,
            e.g., np.array([1, 1, 2, 3, 3]) 
            corresponds to models[0:2] playing on one simplex,
            models[2] playing on another, and models[3:] playing 
            on the final simplex. Ususally set to None to perform
            single-simplex hinting.
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
        groups (numpy.array): mask grouping learners for different 
            delay periods into separate simpilces,
            e.g., np.array([1, 1, 2, 3, 3]) 
            corresponds to models[0:2] playing on one simplex,
            models[2] playing on another, and models[3:] playing 
            on the final simplex. Ususally set to None to perform
            single-simplex hinting.
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
    