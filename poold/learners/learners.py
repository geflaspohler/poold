from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd
import copy
import pdb 

from poold.utils import tic, toc, printf

class OnlineLearner(ABC):
    """ OnlineLearner abstract base class. """    
    def __init__(self, model_list, partition=None, T=None, **kwargs):
        """Initialize online learner. 
        Args:
            model_list (list[str]): list of strings indicating 
                expert model names
                e.g., ["doy", "cfsv2"]
            partition (numpy.array): mask partitioning learners 
                for different delay periods into separate simplicies,
                e.g., np.array([1, 1, 2, 3, 3]) 
            T (int): > 0, algorithm duration, optional
        """                
        self.t = 1 # current algorithm time
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

    @abstractmethod
    def update(self, t_fb, g_fb, hist_fb):
        """ Algorithm specific parameter updates.

        Args:
            t_fb (int): feedback time
            g_fb (np.array): feedback loss gradient
            hist_fb (dict): dictionary of play history 
        """
        pass

    @abstractmethod
    def get_learner_params(self):
        """ Returns current algorithm parameters as a dictionary. """
        pass

    def play(self): 
        """ Play current weight vector and update history."""
        # Add to set missing feedback
        self.outstanding.add(self.t)

        # Get algorithm parameters
        params = self.get_learner_params()

        # Update play history 
        self.play_history[self.t] = (
            copy.copy(self.w), 
            copy.copy(self.h), 
            copy.copy(self.outstanding),
            copy.copy(params))

        # Update algorithm iteration 
        self.t += 1

        return self.w

    def feedback(self, t_fb, loss_fb, hint=None): 
        """ Update weight vector with received feedback
        and any available hints.
        
        Args:
            t_fb (int): feedback for round t_fb
            loss (dict): dictionary of the form:
                {
                    "fun" (callable, optional): function handle for 
                        the loss as a function of play w
                    "jac" (callable): function handle for the loss
                        gradient as a function of play w
                }
                for loss at feedback time t_fb
            hint (dict): dictionary of the form:
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

        # Get previous play 
        w_fb = self.get_play(t_fb)

        # Get linearized loss at feedback time
        g_fb = loss_fb['jac'](w_fb)  
        self.gradient_history[t_fb] = copy.copy(g_fb)

        if hint is None:
            # Default of zero optimistic hint
            self.h = np.zeros((self.d,))
        elif "g" in hint:
            # Use pre-computed hint gradient
            self.h = hint["g"]
        else:
            # Compute loss gradient at current self.w
            self.h = hint['jac'](self.w)  

        # Get play history at time t_fb
        hist_fb = self.get_history(t_fb)

        # Algorithm specific parameter updates
        self.update(t_fb, g_fb, hist_fb)

        # Update history
        self.outstanding.remove(t_fb)

    def log_params(self):
        # Update parameter logging 
        params = {'t': self.t} + self.get_learner_params()
        # Log model weights
        for i in range(self.d):
            params[f"model_{self.expert_models[i]}"] = float(self.w[i])
        return params

    def reset_alg_params(self, T):
        """ Resets algorithm parameters for new duration T. 
        
        Args:
            T (int): > 0, duration
        """
        # Reset algorithm duration
        self.t = 1 # current algorithm time
        self.T = T # algorithm duration 

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
        os = [x[3] for x in self.play_history.values()]
        os_all = set([t for t in os])
        self.gradient_history = \
            {k: self.gradient_history[k] for k in os_all}

        return {
            "t": t_fb,
            "w": w_fb, 
            "h": hint_fb,
            "D": D_fb,
            "g_os": g_os,
            "params": params_fb
        }

    def normalize_by_partition(self, w):
        """ Normalize weight vector by partition.

        Args:
            w (np.array): weight vector 
        """
        for k in self.partition_keys:     
            p_ind = (self.partition == k)                             
            w[p_ind] = (w[p_ind]/ np.sum(w[p_ind]))
        return w
        
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

            if np.isclose(self.lam, 0):
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
            raise ValueError(f"Update produced NaNs: {self.w}")
        if not np.isclose(np.sum(w), 1.0):
            raise ValueError(f"Play w does not sum to 1: {self.w}")

        return w

    def init_weight_vector(self):
        """ Returns uniform initialization weight vector. """          
        w =  np.ones(self.d) / self.d
        w = self.normalize_by_partition(w)
        return w
    
    def get_weights(self):
        ''' Returns dictionary of expert model names and current weights '''
        return dict(zip(self.expert_models, self.w))

    def loss_regret(self, g, w):
        ''' Computes the loss regret w.r.t. a partition of the 
        weight vector using loss gradient.

        Args:
            g (np.array): gradient vector
            w (np.array): weight vector
        '''
        regret = np.zeros(g.shape)
        for k in self.partition_keys:
            p_ind = (self.partition == k)
            regret[p_ind] = np.dot(g[p_ind], w[p_ind]) - g[p_ind] 
        return regret
            
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
        self.reset_alg_params(T)

    def get_learner_params(self):
        """ Returns current algorithm parameters as a dictionary. """
        return { 
            'lam': self.lam,
            'delta': self.delta,
            'Delta': self.Delta,
        }

    def reset_alg_params(self, T):
        """ Resets algorithm parameters for new duration T. 
        
        Args:
            T (int): > 0, duration
        """
        # Base class reset 
        super().reset_alg_params(T)

        #  Algorithm parameters 
        self.w = self.init_weight_vector() # uniform weights
        self.theta = np.zeros((self.d, )) # dual-space parameter 
        self.h = np.zeros((self.d,)) # dual-space parameter 
        self.lam = 0.0 # time varying regularization

        # Regularization parameters
        self.alpha = np.log(self.d) # alpha parameter
        self.at_max = 0.0 # running max of a_t terms for DUB
        self.delta = 0.0 # per-iteration increase in step size
        self.Delta = 0.0 # cummulative sum of a_t^2 + 2b_t terms for DUB

    def update(self, t_fb, g_fb, hist_fb):
        """ Algorithm specific parameter updates.

        Args:
            t_fb (int): feedback time
            g_fb (np.array): feedback loss gradient
            hist_fb (dict): dictionary of play history 
        """
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

        print(self.lam, self.delta)
        
        # Update expert weights 
        self.w = self.softmin_by_partition(self.theta + self.h, self.lam)

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

# class AdaHedgeFO(OnlineExpert):
#     """
#     AdaHedge module implements AdaHedge online learning algorithm from Francesco Orabona's monograph
#     """    
#     def __init__(self, loss, T, init_method, expert_models, reg="orig", partition=None, update_learner=True):
#         """Initializes online_expert 
#         Args:
#            alg: online learning algorithm, one of: "forel", "adahedge", "adahedge_robust", "flip_flop"
#         """                
#         # Base class constructor 
#         super().__init__(loss, T, init_method, expert_models, partition, update_learner)

#         # Check and store regulraization 
#         supported_reg = ['orig', 'plusplus', 'delay_hint', 'delay_nohint', 'nodelay_hint', "forward_drift"]
#         if reg not in supported_reg:
#             raise ValueError(f"Unsupported regularizer for AdaHedgeFO {reg}.")
#         self.reg = reg
#         self.reset_alg_params(T)

#     def reset_alg_params(self, T):
#         """ Set hyperparameter values of online learner 
#         Args:
#             T: integer > 0, algorithm duration
#         """ 
#         # Reset algorithm duration
#         self.t = 1
#         self.T = T

#         # Initilaize algorithm hyperparmetersj
#         self.w = self.init_weight_vector(self.expert_models, self.method) # Expert weight vector
#         self.theta = np.zeros(self.w.shape) # The dual-space parameter 
#         self.delta = 0.0 # per-iteration increase in step size
#         self.Delta = 0.0 # cummulative step size increase
#         self.lam = 0.0 # algorithm step-size or time varying regularization, called lambda_t in Orabona
#         self.l_t = 0.0  # Cummulative loss of weights played
#         self.alpha = np.log(self.d)
#         self.hint_prev = np.zeros(self.w.shape) # The dual-space parameter 

#     def log_params(self):
#         # Update parameter logging 
#         params = {
#             'delta': self.delta,
#             'Delta': self.Delta,
#             'lambda': self.lam,
#             't': self.t
#         }
#         # Allow for duplicate model names in logging
#         for i in range(self.d):
#             params[self.expert_models[i]] = float(self.w[i])
#         return params

#     def get_reg(self, w_eval, g_fb, hint, hint_prev, u_t, update_Delta=True):
#         """ Returns the delayed optimistic foresight gains. If w_eval = self.w, returns
#             optimistic feedback gains. If hint, hint_prev = 0, returns delayed feedback gains.
       
#             w_eval: (d,) numpy array, location to evaluted loss 
#             g_fb: (d,) numpy array, most recent feedback gradient
#             hint: (d,) numpy array, pseudo-gradient hint at current timestep
#             hint_prev : (d,) numpy array, pseudo-gradient hint at previous timestep
#             u_t: (d,) numpy array, the best expert so far, one-hot encoding
#                 only required in the cost of optimistim
#             set_Delta (boolean): if True, compute delta value and update self.Delta
#                 otherwise, use current value of self.Delta
#         """
#         if update_Delta:
#             delta = self.get_delta(w_eval, g_fb, hint, hint_prev)
#             Delta = self.Delta + delta  
#         else:
#             Delta = self.Delta
#             delta = self.delta
            
#         #$eta = (np.dot(hint, u_t - self.w) + Delta) / self.neg_entropy_reg(u_t)
#         eta = (np.dot(hint, u_t - self.w) + Delta) / self.alpha

#         # Enforce regularizer monotonicity
#         return np.max([self.lam, eta]), Delta, delta

#     def get_delta(self, w_eval, g_fb, hint, hint_prev):
#         """ Returns the delayed optimistic foresight gains. If w_eval = self.w, returns
#             optimistic feedback gains. If hint, hint_prev = 0, returns delayed feedback gains.
       
#             w_eval: (d,) numpy array, location to evaluted loss 
#             g_fb: (d,) numpy array, most recent feedback gradient
#             hint: (d,) numpy array, pseudo-gradient hint at current timestep
#             hint_prev : (d,) numpy array, pseudo-gradient hint at previous timestep
#         """
#         base = np.dot(w_eval, g_fb) + np.dot(self.w, hint - hint_prev) 
#         maxv = np.max(g_fb + hint - self.hint_prev)
#         if np.isclose(self.lam, 0):
#             delta = base + np.max(self.theta - hint, axis=None) - \
#                  np.max(self.theta + g_fb - hint_prev, axis=None)
#         else:
#             delta = base + self.lam * \
#                 np.log(np.sum(self.w * np.exp(-((g_fb + hint - self.hint_prev) - maxv) / self.lam))) 
#         return delta

#     def update_and_predict(self, X_cur, hint=None, X_fb=None, y_fb=None, w_fb=None): 
#         """Performs one step of AdaHedge

#         Args:
#            X_cur: G x self.d - current prediction at G grid point locations from self.d experts
#            hint: d x 1 - linearized hint loss
#            X_fb : G x self.d - feedback prediction at G grid point locations from self.d experts
#            y_fb: G x 1 - ground truth feedback forecast at G grid points
#            w_fb: d x 1 np.array - prediction weights used in round when X was expert and y was gt
#         """     
#         '''
#         Incorporate loss from previous timestep
#         '''  
#         if X_fb is not None and y_fb is not None: 
#             # Compute the loss of the previous play
#             g_fb = self.loss_gradient(X_fb, y_fb, w_fb)  # linearize arround current w value        
#             self.l_t += np.dot(g_fb, w_fb) # Loss of play

#         else:
#             # If no feedback is provided use zero gradient
#             g_fb = np.zeros(self.theta.shape)
#             w_fb = self.w

#         if hint is None:
#             hint = np.zeros(self.theta.shape)

#         '''
#         Update dual-space parameter value with standard gradient update
#         '''
#         self.theta = self.theta - g_fb 
#         #self.theta = (self.theta * (self.t - 1)/self.t)  - g_fb / self.t # compute as a moving average 

#         '''
#         Update best expert so far
#         '''

#         # Set u to be the argmax of the hint
#         maximizers = (self.theta == self.theta.max())
#         h = copy.copy(hint)
#         h[~maximizers] = -np.inf
#         u_i = np.argmax(h)
#         u_t = np.zeros((self.d,))
#         u_t[u_i] = 1.0 

#         '''
#         u_t = np.zeros((self.d,))
#         u_i = np.argmax(self.theta)
#         u_t[u_i] = 1.0 
#         '''
        
#         '''
#         if np.isclose(self.lam, 0):
#             h = copy.copy(hint)
#             h[~maximizers] = np.inf
#             u_i = (h == h.min())
#             u_t = np.zeros((self.d,))
#             u_t[u_i] = 1.0 
#         else:
#             u_t = np.exp(-hint / self.lam) * maximizers 
#         '''
#         u_t = u_t / np.sum(u_t, axis=0)
        
#         ''' 
#         Get regularization value for current timestep
#         '''
#         zero = np.zeros(self.d,) # Zero hint dummy variable
#         if self.reg == "orig":
#             w_eval = self.w
#             compute_Delta = True
#             hc = zero # current hint

#         elif self.reg == "delay_hint":
#             w_eval = w_fb
#             compute_Delta = True
#             hc = hint # current hint

#         elif self.reg == "nodelay_hint":
#             w_eval = self.w
#             compute_Delta = True
#             hc = hint # current hint

#         elif self.reg == "delay_nohint":
#             w_eval = w_fb
#             compute_Delta = True
#             hc = zero # current hint
#         elif self.reg == "plusplus":
#             hc = hint
#         elif self.reg == "forward_drift":
#             hc = zero
#         else:
#             raise ValueError(f"Unrecognized regularization {self.reg}") 

#         '''
#         Update regularization
#         '''
#         if self.reg not in ["plusplus", "forward_drift"]:
#             self.lam, self.Delta, self.delta  = self.get_reg(w_eval, g_fb, hc, self.hint_prev, u_t, compute_Delta)
#         else:
#             if np.isclose(self.lam, 0):
#                 Delta = np.inner(hc, u_t) + self.l_t + np.max(self.theta - hc, axis=None)
#             else:
#                 maxv = np.max(self.theta - hc)
#                 Delta = np.inner(hc, u_t) + self.l_t + \
#                     self.lam * np.log(np.sum(np.exp((self.theta - hc - maxv) / self.lam))) + maxv - self.lam * np.log(self.d)

#             self.delta = Delta - self.Delta
#             self.Delta = Delta
#             self.lam = np.max([self.lam, self.Delta / self.alpha])
            
#         '''
#         Update expert weights 
#         '''
#         if np.isclose(self.lam, 0):
#             w_i =  ((self.theta - hint) == (self.theta - hint).max()) # Set to the best expert so far
#             self.w = np.zeros((self.d,))
#             self.w[w_i] = 1.0  / np.sum(w_i)

#         elif not np.isclose(self.lam, 0):
#             maxv = np.max(self.theta - hint)
#             self.w =  np.exp((self.theta - hint - maxv) / self.lam) 
#             self.w = self.w / np.sum(self.w, axis=None)

#         if np.isnan(self.w).any():
#             raise ValueError(f"Bad w{self.w}")

#         self.hint_prev = copy.deepcopy(hint)
#         self.t += 1
        
#         # Return prediction 
#         return X_cur @ self.w

# class AdaHedgeSR(OnlineExpert):
#     """
#     AdaHedge module implements AdaHedge online learning algorithm
#     """    
#     def __init__(self, loss,  T, init_method, expert_models, partition=None, update_learner=True):
#         """Initializes online_expert 
#         """                
#         # Base class constructor 
#         super().__init__(loss, T, init_method, expert_models, partition, update_learner)
#         self.reset_alg_params(T)

#     def reset_alg_params(self, T, expert_df=None):
#         """ Set hyperparameter values of online learner 
#         Args:
#             T: integer > 0, algorithm duration
#             expert_df: dataframe of predictions for estimating L constant
#         """ 
#         #TODO: briefly define what these parameters are; mention that initialization for alpha comes from ...
#         self.t = 1
#         self.T = T
#         self.w = self.init_weight_vector(self.expert_models, self.method) # Expert weight vector
#         self.ada_loss = np.zeros((1, self.d))
#         self.delta = 0.0
#         self.lam = 0.0

#     def log_params(self):
#         # Update parameter logging 
#         params = {
#             'delta': self.delta,
#             't': self.t
#         }
#         # Allow for duplicate model names in logging
#         dup_count = 1
#         for i in range(self.d):
#             if self.expert_models[i] in params:
#                 alias = self.expert_models[i] + str(dup_count)
#                 dup_count += 1
#             else:
#                 alias = self.expert_models[i]
#             params[alias] = float(self.w[i])
#         return params

#     def update_expert(self, X, y, w):
#         """Performs one step of adahedge with respect to self.w. 

#         Args:
#            X: G x self.d - prediction at G grid point locations from self.d experts
#            y: G x 1 - ground truth forecast at G grid points
#            w: G x 1 np.array - prediction weights used in round when X was expert and y was gt
#         """                        
#         # Get gradient and update optimistc hints        
#         g = self.loss_gradient(X, y, self.w)
#         self.update_hint(X, y, update_pred=False)

#         if self.delta == 0:
#             eta = np.inf
#         else:
#             eta = np.log(self.d) / self.delta
            
#         w, Mprev = self.adahedge_mix(eta, self.ada_loss)
        
#         l_t = self.loss_experts(X, y)
#         h = w @ l_t.T
        
#         self.ada_loss += l_t
        
#         w, M = self.adahedge_mix(eta, self.ada_loss)
#         self.w = w.T
        
#         # Max clips numeric Jensen violation
#         delta = np.max([0, h - (M - Mprev)])
#         self.delta += delta
        
#         self.t += 1  
        
#     def adahedge_mix(self, eta, L):
#         m = np.min(L)
#         if (eta == np.inf):
#             w = (L == m)
#         else:
#             w = np.exp(-eta * (L-m))
        
#         s = np.sum(w, axis=None)
#         w = w / s
#         M = m - np.log(s / len(L)) / eta
        
#         return w, M

#     def predict(self, X, hint=None): 
#         """Returns an ensemble of experts, given the current weights 

#         Args:
#            X: G x self.d np array - prediction at G grid point locations from self.d experts
#         """        
#         self.update_hint(X, update_pred=True)
#         return X @ self.w

# class RegretMatching(OnlineExpert):
#     """
#     Implements RegretMatching online learning algorithm
#     """    
#     def __init__(self, loss, T, init_method, expert_models=None, partition=None, update_learner=True):
#         """
#         Initializes online_expert 
#         Args:
#             loss: an OnlineLoss object
#             T: integer > 0, algorithm duration
#             init_method: method for initializing expert weights, one of "uniform", "doy"
#             expert_models: list of strings indicating expert names
#             partition: (np Array) masks for partitions of the vectors ([1, 1, 2, 3, 3]) into separate simpilces
#         """                
#         # Base class constructor 
#         super().__init__(loss, T, init_method, expert_models, partition, update_learner)
#         self.reset_alg_params(T)

#     def reset_alg_params(self, T):
#         """ Set hyperparameter values of online learner 
#         Args:
#             T: integer > 0, algorithm duration
#         """ 
#         self.t = 1
#         self.T = T
#         self.w = self.init_weight_vector(self.expert_models, self.method) # Expert weight vector
#         self.regret = np.zeros((1, self.d))
#         self.hint_prev = np.zeros((1, self.d))

#     def log_params(self):
#         # Update parameter logging 
#         params = {
#             't': self.t
#         }
#         for i in range(self.d):
#             params[self.expert_models[i]] = float(self.w[i])
#         return params

#     def update_and_predict(self, X_cur, hint=None, X_fb=None, y_fb=None, w_fb=None, **params): 
#         """Performs one step of Regret Matching

#         Args:
#            X_cur: G x self.d - current prediction at G grid point locations from self.d experts
#            hint: d x 1 - linearized hint loss
#            X_fb : G x self.d - feedback prediction at G grid point locations from self.d experts
#            y_fb: G x 1 - ground truth feedback forecast at G grid points
#            w_fb: d x 1 np.array - prediction weights used in round when X was expert and y was gt
#         """     

#         '''
#         Incorporate loss from previous timestep
#         '''  
#         if X_fb is not None and y_fb is not None: 
#             # Compute the loss of the previous play
#             g_fb = self.loss_gradient(X=X_fb, y=y_fb, w=w_fb, **params)  # linearize arround current w value        
#             regret_fb = self.loss_regret(g_fb, w_fb) # compute regret w.r.t. partition
#         else:
#             # If no feedback is provided use zero gradient
#             g_fb = np.zeros((self.d,))
#             regret_fb = np.zeros((self.d,))
#             w_fb = self.w

#         if hint is None:
#             hint = np.zeros((self.d,))

#         if self.update_learner:
#             '''
#             Update regret
#             '''
#             self.regret += regret_fb
#             regret_pos = np.maximum(0, self.regret + hint)

#             '''
#             Update expert weights 
#             '''
#             if np.sum(regret_pos) > 0.0:
#                 self.w = (regret_pos / np.sum(regret_pos)).reshape(self.d,)
#             else:
#                 self.w = 1./self.d * np.ones(self.d) # Uniform

#         self.hint_prev = copy.deepcopy(hint)
#         self.t += 1

#         # Return prediction 
#         return X_cur @ self.w

# class RegretMatchingPlus(OnlineExpert):
#     """
#     Implements RegretMatching+ online learning algorithm
#     """    
#     def __init__(self, loss, T, init_method, expert_models=None, partition=None, update_learner=True):
#         """
#         Initializes online_expert 
#         Args:
#             loss: an OnlineLoss object
#             T: integer > 0, algorithm duration
#             init_method: method for initializing expert weights, one of "uniform", "doy"
#             expert_models: list of strings indicating expert names
#             partition: (np Array) masks for partitions of the vectors ([1, 1, 2, 3, 3]) into separate simpilces
#         """                
#         # Base class constructor 
#         super().__init__(loss, T, init_method, expert_models, partition, update_learner)
#         self.reset_alg_params(T)

#     def reset_alg_params(self, T):
#         """ Set hyperparameter values of online learner 
#         Args:
#             T: integer > 0, algorithm duration
#         """ 
#         self.t = 1
#         self.T = T
#         self.p = self.init_weight_vector(self.expert_models, "uniform") # defined to be uniform 
#         self.w = np.zeros((self.d,)) # Must initialize weight vector to zero
#         self.hint_prev = np.zeros((1, self.d))

#     def log_params(self):
#         # Update parameter logging 
#         params = {
#             't': self.t
#         }
#         for i in range(self.d):
#             params[self.expert_models[i]] = float(self.p[i])
#         return params

#     def update_and_predict(self, X_cur, hint=None, X_fb=None, y_fb=None, w_fb=None, **params): 
#         """Performs one step of Regret Matching+

#         Args:
#            X_cur: G x self.d - current prediction at G grid point locations from self.d experts
#            hint: d x 1 - linearized hint loss
#            X_fb : G x self.d - feedback prediction at G grid point locations from self.d experts
#            y_fb: G x 1 - ground truth feedback forecast at G grid points
#            w_fb: d x 1 np.array - prediction weights used in round when X was expert and y was gt
#         """     
#         '''
#         Incorporate loss from previous timestep
#         '''  
#         if X_fb is not None and y_fb is not None: 
#             # Compute the loss of the previous play
#             g_fb = self.loss_gradient(X=X_fb, y=y_fb, w=w_fb, **params)  # linearize arround current w value        
#             regret_fb = self.loss_regret(g_fb, w_fb) # compute regret w.r.t. partition
#         else:
#             # If no feedback is provided use zero gradient
#             g_fb = np.zeros((self.d,))
#             w_fb = self.p
#             regret_fb = np.zeros((self.d,))

#         if hint is None:
#             hint = np.zeros((self.d,))

#         '''
#         Update regret
#         '''
#         if self.update_learner:
#             self.w = np.maximum(0, self.w + regret_fb + hint - self.hint_prev).reshape(-1,)

#         '''
#         Update expert weights 
#         '''
#         #  Get updated weight vector
#         self.p = np.zeros((self.d,))
#         for k in self.partition_keys:     
#             p_ind = (self.partition == k)                             
#             if np.sum(self.w[p_ind]) > 0.0:
#                 self.p[p_ind] = (self.w[p_ind]/ np.sum(self.w[p_ind]))
#             else: 
#                 n_k = sum(p_ind)
#                 self.p[p_ind] = 1./n_k * np.ones(n_k,) # Uniform

#         self.hint_prev = copy.deepcopy(hint)
#         self.t += 1

#         # Return prediction 
#         return X_cur @ self.p

# class FlipFlop(AdaHedgeSR):
#     def __init__(self, loss, T, init_method, expert_models, partition=None, update_learner=True):
#         """Initializes online_expert 
#         """                
#         # Base class constructor 
#         super().__init__(loss, T, init_method, expert_models, partition, update_learner)
#         self.reset_alg_params(T)

#     def reset_alg_params(self, T, expert_df=None):
#         """ Set hyperparameter values of online learner """ 
#         #TODO: briefly define what these parameters are; mention that initialization for alpha comes from ...
#         self.t = 1
#         self.T = T
#         self.w = self.init_weight_vector(self.expert_models, self.method) # Expert weight vector
#         self.ada_loss = np.zeros((1, self.d))
#         self.delta = np.zeros((2,))
#         self.alpha = 1.243
#         self.phi = 2.37
#         self.regime = 0
#         self.scale = np.array([self.phi/self.alpha, self.alpha])
#         self.lam = 0.0

#     def log_params(self):
#         # Update parameter logging 
#         params = {
#             'regime': self.regime,
#             't': self.t
#         }
#         dup_count = 1
#         for i in range(self.d):
#             if self.expert_models[i] in params:
#                 alias = self.expert_models[i] + str(dup_count)
#                 dup_count += 1
#             else:
#                 alias = self.expert_models[i]

#             params[alias] = float(self.w[i])
#         return params
    
#     def update_expert(self, X, y, w):
#         """Performs one step of RegretMatching+

#         Args:
#            X: G x self.d - prediction at G grid point locations from self.d experts
#            y: G x 1 - ground truth forecast at G grid points
#            w: G x 1 np.array - prediction weights used in round when X was expert and y was gt
#         """                        
#         # Get gradient and update optimistc hints        
#         g = self.loss_gradient(X, y, self.w) # Take gradient at current self.w
#         self.update_hint(X, y, update_pred=False)

#         if self.regime == 0 or self.delta[1] == 0.0 :
#             eta = np.inf
#         else:
#             eta = np.log(self.d) / self.delta[1]

#         w, Mprev = self.adahedge_mix(eta, self.ada_loss)

#         l_t = self.loss_experts(X, y)
#         h = w @ l_t.T
        
#         self.ada_loss += l_t
#         w, M = self.adahedge_mix(eta, self.ada_loss)
#         self.w = w.T        
        
#         # Max clips numeric Jensen violation
#         delta = np.max([0, h - (M - Mprev)])
#         self.delta[self.regime] += delta                    

#         if self.delta[self.regime] > self.scale[self.regime] + self.delta[1-self.regime]:
#             self.regime = 1 - self.regime   

#         self.t += 1

#     def predict(self, X, hint=None): 
#         """Returns an ensemble of experts, given the current weights 

#         Args:
#            X: G x self.d np array - prediction at G grid point locations from self.d experts
#         """        
#         self.update_hint(X, update_pred=True)
#         return X @ self.w
