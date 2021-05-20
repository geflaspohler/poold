""" Learner class implements online learning algorithms.

This abstract base class defines a template for an online learning 
algorithm. The OnlineLearnere object must implement several methods:
    * learner_update: updates the learner play and parameters 
    * get_learner_params: returns learner parameters
    * init_alg_params: initializes learner parameters
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

    def update(self, t, times_fb, losses_fb, hint):
        """ Update online learner and generate a new play.
        
        Update weight vector with received feedback
        and any available hints. Update history and return 
        play for time t.
        
        Args:
            t (int): current time
            times_fb (list[int]): list of avaliable feedback times
            losses_fb (list[dict]): list of loss dictionaries for 
                feedback times, where each dict is of the form:
                {
                    "fun" (callable, optional): function handle for 
                        the loss as a function of play w
                    "jac" (callable): function handle for the loss
                        gradient as a function of play w
                }
                for loss at feedback time t_fb
            hint (dict): hint dictionary of the form:
                {
                    "fun" (callable, optional): function handle for 
                        the hint as a function of play w
                    "g" (np.array, optional): pseudo-gradient vector.
                        If provided, "jac" is ignored.
                    "jac" (callable, optional): function handle for 
                        the hint gradient as a function of play w
                }
                for the hint pseudoloss at time self.t 
        """
        # Check agreement between algorithm plays and 
        # internal time
        assert(self.t == t)

        # Add to set missing feedback
        self.outstanding.add(t)

        # Get hint from input 
        if hint is None:
            # Default of zero optimistic hint
            self.h = np.zeros((self.d,))
        elif "g" in hint:
            # Use pre-computed hint gradient
            self.h = hint["g"]
        else:
            # Compute loss gradient at current self.w
            self.h = hint['jac'](self.w)  

        # Compute all algorithm updates
        if len(times_fb) == 0:
            # Hint-only algorithm updates
            self.learner_update(t_fb=None, g_fb=None, hist_fb=None, hint=self.h)
        else:
            for t_fb, loss_fb in zip(times_fb, losses_fb):
                # Update learner for a single timestep 
                g_fb = self.update_single(t_fb, loss_fb, self.h)

                # Update history
                self.outstanding.remove(t_fb)

        # Get algorithm parameters
        params = self.get_learner_params()

        # Update play history 
        self.play_history[t] = (
            copy.copy(self.w), 
            copy.copy(self.h), 
            copy.copy(self.outstanding),
            copy.copy(params))

        # Update algorithm iteration 
        self.t += 1
        return self.w

    def update_single(self, t_fb, loss_fb, hint): 
        """ Update weight vector with received feedback
        and any available hints.
        
        Args:
            t_fb (int): feedback for round t_fb
            loss_fb (dict): loss dictionary as described in :update
            hint (np.array): hint vector

        Returns:
            g_fb: gradient at time t_fb
        """
        # Get previous play 
        w_fb = self.get_play(t_fb)

        # Get linearized loss at feedback time
        g_fb = loss_fb['jac'](w=w_fb)  
        # Update gradient history 
        self.gradient_history[t_fb] = copy.copy(g_fb)

        # Get play history at time t_fb
        hist_fb = self.get_history(t_fb)

        # Algorithm specific parameter updates
        self.learner_update(t_fb, g_fb, hist_fb, hint)

        return g_fb

    @abstractmethod
    def learner_update(self, t_fb, g_fb, hist_fb, hint, **kwargs):
        """ Algorithm specific parameter updates. If t_fb 
        is None, perform a hint-only parameter update.

        Args:
            t_fb (int): feedback time
            g_fb (np.array): feedback loss gradient
            hist_fb (dict): dictionary of play history 
            hint (np.array): hint vector at time t
        """
        pass

    @abstractmethod
    def get_learner_params(self):
        """ Returns current algorithm parameters as a dictionary. """
        pass

    def log_params(self):
        """ Return dictionary of algorithm parameters """
        params = {'t': self.t} + self.get_learner_params()
        # Log model weights
        for i in range(self.d):
            params[f"model_{self.expert_models[i]}"] = float(self.w[i])
        return params

    def init_alg_params(self, T):
        """ Resets algorithm parameters for new duration T. 
        
        Args:
            T (int): > 0, duration
        """
        # Reset algorithm duration
        self.t = 0 # current algorithm time
        self.T = T # algorithm duration 
        self.h = np.zeros((self.d,)) # last provided hint vector 

        # Record keeping 
        self.play_history = {} # history of past plays
        self.gradient_history = {} # history of loss gradients
        self.outstanding = set() # currently outstanding feedback

    def get_play(self, t_fb):
        """ Get past play history at time t_fb """
        assert(t_fb in self.play_history)
        w_fb = self.play_history[t_fb][0] # past play
        return w_fb

    def get_history(self, t_fb):
        """ Get parameters from history at time t_fb and delete
        from storage to ensure sublinear memory """
        assert(t_fb in self.play_history)
        assert(t_fb in self.outstanding)

        w_fb = self.play_history[t_fb][0] # past play
        hint_fb = self.play_history[t_fb][1] # past hint
        t_os = self.play_history[t_fb][2] # set of outstanding feedbacks at t_fb
        params_fb = self.play_history[t_fb][3] # past play parameters  (optional)

        D_fb = len(t_os) # length of delay at t_fb

        # sum of outstanding gradients at t_fb
        g_os = sum([self.gradient_history[t] \
            for t in t_os]) 

        del self.play_history[t_fb]

        # Keep only gradient elements that remain outstanding 
        os = [x[2] for x in self.play_history.values()]

        # Flatten and find unique outstanding feedbacks
        os_all = set([x for sublist in os for x in sublist])

        # Subset gradient history to only outstanding feedbacks
        self.gradient_history = \
            {k: self.gradient_history[k] for k in os_all if k in self.gradient_history}

        return {
            "t": t_fb,
            "w": w_fb, 
            "h": hint_fb,
            "D": D_fb,
            "g_os": g_os,
            "params": params_fb
        }

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

        # Check computation 
        if np.isnan(w).any():
            raise ValueError(f"Update produced NaNs: {w}")
        if not np.isclose(np.sum(w), 1.0):
            raise ValueError(f"Play w does not sum to 1: {w}")

        return w

    def init_weight_vector(self):
        """ Returns uniform initialization weight vector. """          
        w =  np.ones(self.d) / self.d
        w = normalize_by_partition(w, self.partition)
        return w
    
    def get_weights(self):
        ''' Returns dictionary of expert model names and current weights '''
        return dict(zip(self.expert_models, self.w))

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
        self.init_alg_params(T)

    def get_learner_params(self):
        """ Returns current algorithm parameters as a dictionary. """
        return { 
            'lam': self.lam,
            'delta': self.delta,
            'Delta': self.Delta,
        }

    def init_alg_params(self, T):
        """ Resets algorithm parameters for new duration T. 
        
        Args:
            T (int): > 0, duration
        """
        # Base class reset 
        super().init_alg_params(T)

        #  Algorithm parameters 
        self.w = self.init_weight_vector() # uniform weights
        self.theta = np.zeros((self.d, )) # dual-space parameter 
        self.lam = 0.0 # time varying regularization

        # Regularization parameters
        self.alpha = np.log(self.d) # alpha parameter
        self.at_max = 0.0 # running max of a_t terms for DUB
        self.delta = 0.0 # per-iteration increase in step size
        self.Delta = 0.0 # cummulative sum of a_t^2 + 2b_t terms for DUB

    def learner_update(self, t_fb, g_fb, hist_fb, hint):
        """ Algorithm specific parameter updates. If t_fb 
        is None, perform a hint-only parameter update 

        Args:
            t_fb (int): feedback time
            g_fb (np.array): feedback loss gradient
            hist_fb (dict): dictionary of play history 
            hint (np.array): hint vector at time t
        """
        # Hint only update
        if t_fb is None:
            self.w = self.softmin_by_partition(self.theta + hint, self.lam)
            return

        # Update dual-space parameter value with standard 
        # gradient update, sum of gradients
        self.theta = self.theta + g_fb 

        # Update regularization
        assert("lam" in hist_fb["params"])
        if self.reg == "adahedged":
            self.lam, self.delta  = self.get_reg(
                g_fb, hist_fb["w"], hist_fb["h"],
                hist_fb["g_os"], hist_fb["params"]["lam"])
        elif self.reg == "upper_bound":
            self.lam, self.delta  = self.get_reg_uniform(
                g_fb, hist_fb["w"], hist_fb["h"],
                hist_fb["g_os"], hist_fb["D"])
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
        g_diff = hint_fb - g_os
        maxval = np.max(g_diff[w_fb != 0.0])

        if np.isclose(lam_fb, 0):
            delta_aux = -np.dot(g_diff - maxval, w_fb)
        else:
            delta_aux = lam_fb * \
                np.log(np.sum(w_fb * np.exp((g_diff - maxval) / lam_fb))) - \
                np.dot(g_diff - maxval, w_fb)

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
        err_norm = np.linalg.norm(hint_t - g_os, ord=np.inf)
        return 2*np.min([g_norm, err_norm])

    def get_bt_bound(self, g_fb, hint_fb, g_os):
        """ Get bound on the value of b_t terms, assume diam(W) = 2

        Args:
            g_fb (numpy.array): most recent feedback gradient t-D
            hint_fb (numpy.array): hint at t-D
            g_os (numpy.array): sum of gradients outstanding at time t-D
        """
        g_norm = np.linalg.norm(g_fb, ord=np.inf)
        err_norm = np.linalg.norm(hint_t - g_os, ord=np.inf)
        return np.min([0.5*(err_norm**2), g_norm * err_norm])
