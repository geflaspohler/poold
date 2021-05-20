import numpy as np

def loss_regret(g, w, partition):
    ''' Computes the loss regret w.r.t. a partition of the 
    weight vector using loss gradient.

    Args:
        g (np.array): gradient vector
        w (np.array): weight vector
        partition (list[int]): mask partitioning learners 
            for different tasks into separate simplicies,
            e.g., np.array([1, 1, 2, 3, 3]) means to use
            model_list[0:2] for the first task,
            model_list[2] for the second task,
            model_list[3:] for the third task
    '''
    partition_keys = list(set(partition))
    regret = np.zeros(g.shape)
    for k in partition_keys:
        p_ind = (partition == k)
        regret[p_ind] = np.dot(g[p_ind], w[p_ind]) - g[p_ind] 
    return regret

def normalize_by_partition(w, partition):
    """ Normalize weight vector by partition.

    Args:
        w (np.array): weight vector 
        partition (list[int]): mask partitioning learners 
            for different tasks into separate simplicies,
            e.g., np.array([1, 1, 2, 3, 3]) means to use
            model_list[0:2] for the first task,
            model_list[2] for the second task,
            model_list[3:] for the third task
    """
    partition_keys = list(set(partition))
    for k in partition_keys:     
        p_ind = (partition == k)                             
        w[p_ind] = (w[p_ind]/ np.sum(w[p_ind]))
    return w
    