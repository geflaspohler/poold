""" Learner class implements online learning algorithms.

This abstract base class defines a template for an online learning 
algorithm. The OnlineLearnere object must implement several methods:
    * update : updates the learner play and parameters 
    * get_params: returns learner parameters
    * reset_params: initializes learner parameters
Several online learning algorithms are implemented as derived
OnlineLearning classes. Learners can be instantiated using
the module's create function.

For example:

    import poold

    models = ["model1", "model2"]
    duration = 20
    learner = poold.create("adahedged", model_list=models, T=duration)

"""
from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd
import copy

from ..utils import loss_regret, normalize_by_partition
from ..utils import History

import pdb # TODO: delete this import

class OnlineLearner(ABC):
    """ OnlineLearner abstract base class. """    
    def __init__(self, model_list, partition=None, T=None, **kwargs):
        """Initialize online learner. 
        Args:
            model_list (list[str]): list of strings indicating 
                expert model names
                e.g., ["doy", "cfsv2"]
            partition (list[int]): mask partitioning learners 
                for different tasks into separate simplicies,
                e.g., np.array([1, 1, 2, 3, 3]) means to use
                model_list[0:2] for the first task,
                model_list[2] for the second task,
                model_list[3:] for the third task
            T (int): > 0, algorithm duration, optional
        """                
        self.t = 0 # current algorithm time
        self.T = T # algorithm horizon

        # Set up expert models
        self.expert_models = model_list
        self.d = len(self.expert_models) # number of experts

        # Initialize partition of weight vector into simplices
        if partition is not None:
            self.partition = np.array(partition)
        else:
            self.partition = np.ones(len(self.expert_models),)
        self.partition_keys = list(set(self.partition))

        # Create online learning history 
        self.history = History(model_list, T)

        # Oustanding losses 
        self.outstanding = set() 

    @abstractmethod
    def update(self, t_fb, fb, hint, **kwargs):
        """ Algorithm specific parameter updates. If t_fb 
        is None, perform a hint-only parameter update.

        Args:
            t_fb (int): feedback time
            fb (dict): dictionary of play details at 
                feedback time
            hint (np.array): hint vector at time t
        """
        pass

    @abstractmethod
    def get_params(self):
        """ Returns current algorithm parameters as a dictionary. """
        pass

    def update_and_play(self, t, losses_fb, hint):
        """ Update online learner and generate a new play.
        
        Update weight vector with received feedback
        and any available hints. Update history and return 
        play for time t.
        
        Args:
            t (int): current time
            losses_fb (list[(int, loss)]): list of 
                (feedback time, loss_feedback) tuples
            hint (dict): hint dictionary of the form:
                {
                    "fun" (callable, optional): function handle for 
                        the hint as a function of play w
                    "grad" (callable): pseudo-gradient vector.
                }
                for the hint pseudoloss at time self.t 
        """
        # Add to set missing feedback
        self.outstanding.add(t)

        # Update the history with received losses
        self.history.record_losses(t, losses_fb)

        # Get hint from input 
        if hint is None:
            # Default of zero optimistic hint
            self.h = np.zeros((self.d,))
        else:
            # Compute loss gradient at current self.w
            self.h = hint['grad'](self.w)  

        # Compute all algorithm updates
        if len(losses_fb) == 0:
            # Hint-only algorithm updates
            self._single_time_update(t_fb=None, hint=self.h)
        else:
            for t_fb, loss_fb in losses_fb:
                self._single_time_update(t_fb=t_fb, hint=self.h)

                # Update history
                self.outstanding.remove(t_fb)

        # Get algorithm parameters
        params = self.get_params()

        # Update play history 
        self.history.record_play(t, self.w)
        self.history.record_hint(t, self.h)
        self.history.record_params(t, params)
        self.history.record_os(t, self.outstanding)

        # Update algorithm iteration 
        self.t += 1
        return self.w

    def _single_time_update(self, t_fb, hint): 
        """ Update weight vector with received feedback
        and any available hints.
        
        Args:
            t_fb (int): feedback for round t_fb
            hint (np.array): hint vector
        """
        if t_fb is None:
            self.update(t_fb=None, fb=None, hint=hint)
            return

        # Get play history at time t_fb
        fb = self.history.get(t_fb)

        # Algorithm specific parameter updates
        self.update(t_fb, fb, hint)

    def log_params(self, t):
        """ Return dictionary of algorithm parameters """
        params = {'t': t} + self.get_params()
        # Log model weights
        for i in range(self.d):
            params[f"model_{self.expert_models[i]}"] = float(self.w[i])
        return params

    def reset_params(self, T):
        """ Resets algorithm parameters for new duration T. 
        
        Args:
            T (int): > 0, duration
        """
        # Record keeping 
        self.outstanding = set() # currently outstanding feedback
        self.history.reset(self.t)

        # Reset algorithm duration
        self.t = 0 # current algorithm time
        self.T = T # algorithm duration 
        self.h = np.zeros((self.d,)) # last provided hint vector 

    def softmin_by_partition(self, theta, lam):
        """ Return a vector w corresponding to a softmin of
        vector theta with temperature parameter lam

        Args:
            theta (np.array): input vector
            lam (float): temperature parameter 
        """
        # Initialize weight vector
        w = np.zeros((self.d,))

        # Iterate through partitions
        for k in self.partition_keys:     
            # Get partition subset
            p_ind = (self.partition == k)                             
            theta_sub = theta[p_ind]
            w_sub = w[p_ind]

            if np.isclose(lam, 0):
                # Return uniform weights over minimizing values
                w_i =  (theta_sub == theta_sub.min()) # get minimum index
                w_sub[w_i] = 1.0  / np.sum(w_i)
            else:
                # Return numerically stable softmin
                minval = np.min(theta_sub)
                w_sub =  np.exp((-theta_sub + minval) / lam) 
                w_sub = w_sub / np.sum(w_sub, axis=None)
            
            w[p_ind] = w_sub
            if not np.isclose(np.sum(w_sub), 1.0):
                raise ValueError(f"Play w does not sum to 1: {w}")

        # Check computation 
        if np.isnan(w).any():
            raise ValueError(f"Update produced NaNs: {w}")

        return w

    def init_weights(self):
        """ Returns uniform initialization weight vector. """          
        w =  np.ones(self.d) / self.d
        w = normalize_by_partition(w, self.partition)
        return w
    
    def get_weights(self):
        ''' Returns dictionary of expert model names and current weights '''
        return dict(zip(self.expert_models, self.w))

    def get_outstanding(self, t, include=True):
        """ Gets outstanding predictions at time t 

        Args: 
            t (int): a time 
            include (bool): if True, include current time
                t in oustanding set
        """
        # Add t to oustanding if not already present
        if include: 
            self.outstanding.add(t)
        return list(self.outstanding)

class AdaHedgeD(OnlineLearner):
    """
    AdaHedgeD module implements delayed AdaHedge 
    online learning algorithm.
    """    
    def __init__(self, model_list, partition=None, T=None, reg="adahedged"):
        """ Initializes online_expert 

        Args:
           reg (str): regularization strategy [ "dub | "adahedged" ], 
                for Delayed Upper bound or AdaHedgeD-style 
                adaptive regularization

            Other args defined in OnlineLearner base class.
        """                
        # Base class constructor 
        super().__init__(model_list, partition, T)

        # Check and store regulraization 
        supported_reg = ["adahedged", "dub"]
        if reg not in supported_reg:
            raise ValueError(
                f"Unsupported regularizer for AdaHedgeD {reg}.")

        self.reg = reg
        self.reset_params(T)

    def get_params(self):
        """ Returns current algorithm parameters as a dictionary. """
        return { 
            'lam': self.lam,
            'delta': self.delta,
            'Delta': self.Delta,
        }

    def reset_params(self, T):
        """ Resets algorithm parameters for new duration T. 
        
        Args:
            T (int): > 0, duration
        """
        # Base class reset 
        super().reset_params(T)
        
        # Initialize play
        self.w = self.init_weights() # uniform weights

        #  Algorithm parameters 
        self.theta = np.zeros((self.d, )) # dual-space parameter 
        self.lam = 0.0 # time varying regularization

        # Regularization parameters
        self.alpha = np.log(self.d) # alpha parameter
        self.at_max = 0.0 # running max of a_t terms for DUB
        self.delta = 0.0 # per-iteration increase in step size
        self.Delta = 0.0 # cummulative sum of a_t^2 + 2b_t terms for DUB

    def update(self, t_fb, fb, hint):
        """ Algorithm specific parameter updates. If t_fb 
        is None, perform a hint-only parameter update 

        Args:
            t_fb (int): feedback time
            fb (dict): dictionary of play details at 
                feedback time
            hint (np.array): hint vector at time t
        """
        # Hint only update
        if t_fb is None:
            self.w = self.softmin_by_partition(self.theta + hint, self.lam)
            return

        # Get feedback gradient
        g_fb = fb['g']

        # Update dual-space parameter value with standard 
        # gradient update, sum of gradients
        self.theta = self.theta + g_fb 

        # Update regularization
        assert("lam" in fb["params"])
        if self.reg == "adahedged":
            self.lam, self.delta  = self.get_reg(
                g_fb, fb["w"], fb["h"],
                fb["g_os"], fb["params"]["lam"])
        elif self.reg == "dub":
            self.lam, self.delta  = self.get_reg_uniform(
                g_fb, fb["w"], fb["h"],
                fb["g_os"], fb["D"])
        else:
            raise ValueError(f"Unrecognized regularizer {self.reg}")

        # Update expert weights 
        self.w = self.softmin_by_partition(self.theta + hint, self.lam)

    def get_reg(self, g_fb, w_fb, hint_fb, g_os, lam_fb):
        """ Returns an updated AdaHedgeD-style regularizer
            g_fb (numpy.array): most recent feedback gradient t-D
            w_fb (numpy.array): play at time t-D
            hint_fb (numpy.array): hint at t-D
            g_os (numpy.array): sum of gradients outstanding at time t-D
            lam_fb (float): value of regularizer at t-D
        """
        # Get delta value
        delta = self.get_delta(g_fb, w_fb, hint_fb, g_os, lam_fb)

        # Update regularization
        eta = self.lam + delta / self.alpha 
        return eta, delta

    def get_delta(self, g_fb, w_fb, hint_fb, g_os, lam_fb):
        """ Computes change to AdaHedgeD-style regularizer.
            g_fb (numpy.array): most recent feedback gradient t-D
            w_fb (numpy.array): play at time t-D
            hint_fb (numpy.array): hint at t-D
            g_os (numpy.array): sum of gradients outstanding at time t-D
            lam_fb (float): value of regularizer at t-D
        """
        # Compute the Be-The-Regularized-Leader Solution using
        # \lam_{t-D} and g_{1:t-D}
        w_btrl = self.softmin_by_partition(self.theta, lam_fb)

        # Compute drift delta
        delta_drift = np.dot(g_fb, w_fb - w_btrl)

        # Compute auxiliary regret delta 
        if np.isclose(lam_fb, 0):
            delta_aux = np.dot(self.theta, w_fb) - np.min(self.theta)
        else:
            g_diff = hint_fb - g_os
            maxval = np.max(g_diff[w_fb != 0.0])

            delta_aux = lam_fb * \
                np.log(np.sum(w_fb * np.exp((g_diff - maxval) / lam_fb))) + \
                np.dot(-g_diff, w_fb) + maxval

        # Compute final delta term
        delta = max(min(delta_drift, delta_aux), 0.0)
        return delta

    def get_reg_uniform(self, g_fb, w_fb, hint_fb, g_os, D):
        """ Delayed upper bound regularization value
            g_fb (numpy.array): most recent feedback gradient t-D
            hint_fb (numpy.array): hint at t-D
            g_os (numpy.array): sum of gradients outstanding at time t-D
            D (int): length of delay
        """
        # Get a_t and b_t upper bounds
        a_t = self.get_at_bound(g_fb, hint_fb, g_os)
        b_t = self.get_bt_bound(g_fb, hint_fb, g_os)

        # Update max a_t term
        self.at_max = np.max([self.at_max, a_t])

        # Update delta term
        delta = a_t**2 + 2*b_t
        self.Delta += delta

        eta = 2*D*self.at_max + np.sqrt(self.Delta)

        # Ensure monotonic increases
        return np.max([self.lam, eta]), delta 

    def get_at_bound(self, g_fb, hint_fb, g_os):
        """ Get bound on the value of a_t terms, assume diam(W) = 2

        Args:
            g_fb (numpy.array): most recent feedback gradient t-D
            hint_fb (numpy.array): hint at t-D
            g_os (numpy.array): sum of gradients outstanding at time t-D
        """
        g_norm = np.linalg.norm(g_fb, ord=np.inf)
        err_norm = np.linalg.norm(hint_fb - g_os, ord=np.inf)
        return 2*np.min([g_norm, err_norm])

    def get_bt_bound(self, g_fb, hint_fb, g_os):
        """ Get bound on the value of b_t terms, assume diam(W) = 2

        Args:
            g_fb (numpy.array): most recent feedback gradient t-D
            hint_fb (numpy.array): hint at t-D
            g_os (numpy.array): sum of gradients outstanding at time t-D
        """
        g_norm = np.linalg.norm(g_fb, ord=np.inf)
        err_norm = np.linalg.norm(hint_fb - g_os, ord=np.inf)
        return np.min([0.5*(err_norm**2), g_norm * err_norm])

class DORM(OnlineLearner):
    """
    DORM module implements delayed optimistc regret matching 
    online learning algorithm.
    """    
    def __init__(self, model_list, partition=None, T=None):
        """ Initializes online_expert 

        Args:
            Args defined in OnlineLearner base class.
        """                
        # Base class constructor 
        super().__init__(model_list, partition, T)

        #  Initialize learner
        self.reset_params(T)

    def get_params(self):
        """ Returns current algorithm parameters as a dictionary. """
        return {}

    def reset_params(self, T):
        """ Resets algorithm parameters for new duration T. 
        
        Args:
            T (int): > 0, duration
        """
        # Base class reset 
        super().reset_params(T)

        # Initialize play
        self.w = self.init_weights() # uniform weights

        #  Algorithm parameters 
        self.regret = np.zeros((self.d,)) # cumulative regret vector 

    def update(self, t_fb, fb, hint):
        """ Algorithm specific parameter updates. If t_fb 
        is None, perform a hint-only parameter update 

        Args:
            t_fb (int): feedback time
            fb (dict): dictionary of play details at 
                feedback time
            hint (np.array): hint vector at time t
        """
        print("time_fb:", t_fb)
        # Hint only update
        if t_fb is None:
            regret_pos = np.maximum(0, hint)
            self.w = normalize_by_partition(regret_pos, self.partition)
            return 

        # Update dual-space parameter value with standard 
        # regret gradient update, sum of gradients
        assert("w" in fb)
        assert("g" in fb)
        g_fb = fb['g'] # get feedback gradient
        w_fb = fb["w"] # get feedback play 
        print("g_fb:", g_fb)
        print("w_fb:", w_fb)
        regret_fb = loss_regret(g_fb, w_fb, self.partition) # compute regret w.r.t. partition 
        print("r_fb:", regret_fb)
        self.regret = self.regret + regret_fb 
        print("regret:", self.regret)

        # Update regret
        print("hint:", hint)
        regret_pos = np.maximum(0, self.regret + hint)
        print("regret pos:", regret_pos)

        # Update expert weights 
        self.w = normalize_by_partition(regret_pos, self.partition)
        print("w:", self.w)
        # pdb.set_trace()

class DORMPlus(OnlineLearner):
    """
    DORMPlus module implements delayed optimistc regret matching+
    online learning algorithm.
    """    
    def __init__(self, model_list, partition=None, T=None):
        """ Initializes online_expert 

        Args:
            Args defined in OnlineLearner base class.
        """                
        # Base class constructor 
        super().__init__(model_list, partition, T)

        #  Initialize learner
        self.reset_params(T)

    def get_params(self):
        """ Returns current algorithm parameters as a dictionary. """
        return {}

    def reset_params(self, T):
        """ Resets algorithm parameters for new duration T. 
        
        Args:
            T (int): > 0, duration
        """
        # Base class reset 
        super().reset_params(T)

        #  Algorithm parameters 
        self.w = self.init_weights() # uniform weights, initial play
        self.p = np.zeros((self.d,)) # must initialize initial pseudo-play to zero
        self.hint_prev = np.zeros((self.d,)) # past hint

    def update(self, t_fb, fb, hint):
        """ Algorithm specific parameter updates. If t_fb 
        is None, perform a hint-only parameter update 

        Args:
            t_fb (int): feedback time
            fb (dict): dictionary of play details at 
                feedback time
            hint (np.array): hint vector at time t
        """
        # Hint only update
        if t_fb is None:
            self.p = np.maximum(0, self.p + hint - self.hint_prev)
            self.w = normalize_by_partition(self.p, self.partition)
            self.hint_prev = copy.deepcopy(hint)
            return 

        # Update dual-space parameter value with standard 
        # regret gradient update, sum of gradients
        assert("w" in fb)
        assert("g" in fb)
        w_fb = fb["w"]
        g_fb = fb["g"]
        regret_fb = loss_regret(g_fb, w_fb, self.partition) # compute regret w.r.t. partition 

        # Update psuedo-play 
        self.p = np.maximum(0, self.p + regret_fb + hint - self.hint_prev)

        # Update expert weights 
        self.w = normalize_by_partition(self.p, self.partition)
