# General imports
import pandas as pd
import numpy as np
import copy, os
from datetime import datetime, timedelta
from functools import partial

# PoolD imports
from poold.environment import Environment 

# Custom subseasonal forecasting libraries 
from src.utils.models_util import get_forecast_filename
from src.utils.general_util import tic, toc, printf
from src.utils.experiments_util import get_measurement_variable, get_ground_truth, get_start_delta

# TODO: remove this import
import pdb

class S2SEnvironment(Environment):
    """ S2S data class for online learning """ 
    def __init__(self, times, models, gt_id, horizon): 
        """ Initialize dataset.

        Args:
            times (list[datetime]): list of prediction times
            models (list[str]): list of expert model names
            gt_id (str): ground truth id
            horizon (str):  horizon
        """
        # Call base class constructor
        super().__init__(times)
        self.models = models
        self.gt_id = gt_id
        self.horizon = horizon

        var = get_measurement_variable(gt_id)
        self.gt = get_ground_truth(gt_id).loc[:,['lat', 'lon', 'start_date', var]]

        # Important to sort df in order to ensure lat/lon points are in consistant order 
        self.gt = self.gt.set_index(['start_date', 'lat', 'lon']).squeeze().sort_index()

        # Store delta between target date and forecast issuance date
        self.start_delta = timedelta(days=get_start_delta(horizon, gt_id))

        # Rodeo loss object
        self.rodeo_loss = RodeoLoss()

    def get_feedback(self, t, os_times=None):
        """ Get feedback avaliable at time t 

        Args:
            t (int): current time
            os_times (list[int]): list of times with outstanding forecasts
                if None, os_times = [t]
        """
        assert(t <= self.T)

        if os_times is None:
            os_times = range(0, self.T)

        date = self.times[t]
        date_str = datetime.strftime(date, '%Y%m%d')

        # Outstanding prediction dates
        os_dates = [self.times[t] for t in os_times]

        # Oustanding targets
        os_targets = [self.date_to_target(d) for d in os_dates]

        # Get times with targets earlier than current prediction date
        return [t for t, d in zip(os_times, os_targets) if d < date]

    def get_loss(self, t):
        """ Get loss function at time t 

        Args:
            t (int): current time 
        """
        X_t = self.get_pred(t, verbose=False)
        y_t = self.get_gt(t)

        loss = {
            "fun": partial(self.rodeo_loss.loss, 
                            X=X_t.to_numpy(copy=False), 
                            y=y_t.to_numpy(copy=False)),
            "jac": partial(self.rodeo_loss.loss_gradient, 
                            X=X_t.to_numpy(copy=False), 
                            y=y_t.to_numpy(copy=False))
        }

        return loss

    def get_gt(self, t):
        """ Get the ground truth value for a time t

        Args:
            t (int): current time 
        """
        assert(t <= self.T)
        date = self.times[t]
        target = self.date_to_target(date)
        target_str = datetime.strftime(target, '%Y%m%d')      

        return self.gt[self.gt.index.get_level_values("start_date").isin([target_str])]

    def get_pred(self, t, verbose=False):
        """  Get all model predictions and return a 
        merged set of predictions for a time.

        Args:
            t (int): current time  
            verbose (bool): print model load status 
        """
        assert(t <= self.T)

        present_models = [] # list of models with forecasts
        missing_models = []
        merged_df = None # df for all model predictions

        for model in self.models:
            df = self.get_model(t, model)
            if df is None:
                missing_models.append(model)
                continue

            # Append model to list of present models
            present_models.append(model)

            if merged_df is None:
                merged_df = copy.copy(df)
            else:
                merged_df = pd.merge(merged_df, df, 
                    on=["start_date", "lat", "lon"])


        if merged_df is None and verbose:
            print(f"Warning: No model forecasts for {target}")
            return None

        if verbose:
            print(f"Target {t}: missing models {missing_models}")

        return merged_df

    def time_to_target(self, t):
        """ Convert prediction time to a target date """
        return self.date_to_target(self.times[t])

    def date_to_target(self, date):
        """ Convert issuance date to target date for forecasting """
        return date + self.start_delta

    def most_recent_obs(self, t):
        """ Gets the most recent observation available time t

        Args:
            t (int):  time t
        """  
        assert(t <= self.T)
        date = self.times[t]
        date_str = datetime.strftime(date, '%Y%m%d')      

        if self.gt.index.get_level_values('start_date').isin([date_str]).any():                
            return self.gt[self.gt.index.get_level_values('start_date') == date_str]
        else:
            printf(f"Warning: ground truth observation not avaliable on {date_str}")
            obs = self.gt[self.gt.index.get_level_values('start_date') < date_str]
            last_date = obs.tail(1).index.get_level_values('start_date')[0]
            return self.gt[self.gt.index.get_level_values('start_date') == last_date]

    def get_model(self, t, model, verbose=False):
        """ Get model prediction for a target time

        Args:
            t (int): current time  
            model (str):  model name
            verbose (bool): print model load status 
        """
        assert(t <= self.T)
        date = self.times[t]
        target = self.date_to_target(date)
        target_str = datetime.strftime(target, '%Y%m%d')      

        try:
            fname = get_forecast_filename(
                    model=model, 
                    submodel=None,
                    gt_id=self.gt_id,
                    horizon=self.horizon,
                    target_date_str=target_str)
        except:
            pdb.set_trace()
            fname = get_forecast_filename(
                    model=model, 
                    submodel=None,
                    gt_id=self.gt_id,
                    horizon=self.horizon,
                    target_date_str=target_str)

        if not os.path.exists(fname) and verbose:
            printf("Warning: no forecast found for model {model} on target {target}.")
            return None

        df = pd.read_hdf(fname).rename(columns={"pred": f"{model}"})

        # If any of expert predictions are NaN
        if df.isna().any(axis=None) and verbose: 
            printf("Warning: NaNs in forecast for model {model} on target {target}.")
            return None

        # Important to sort in order to ensure lat/lon points are in consistant order 
        df = df.set_index(['start_date', 'lat', 'lon']).squeeze().sort_index()   
        
        return df

class RodeoLoss(object):
    """ Rodeo loss object """
    def __init__(self):
        pass

    def loss(self, X, y, w):
        """Computes the geographically-averaged rodeo RMSE loss. 

        Args:
           X (np.array): G x self.d, prediction at G grid point locations from self.d experts        
           y (np.array): G x 1, ground truth at G grid points
           w (np.array): d x 1, location at which to compute gradient.

        """     
        return np.sqrt(np.mean((X@w - y)**2, axis=0))    
    
    def loss_experts(self, X, y):
        """Computes the geographically-averaged rodeo RMSE loss. 

        Args:
           X (np.array): G x self.d, prediction at G grid point locations from self.d experts        
           y (np.array): G x 1, ground truth at G grid points
        """     
        d = X.shape[1]
        return np.sqrt(np.mean(
            (X - np.matlib.repmat(y.reshape(-1, 1), 1, d))**2, axis=0))    
    
    def loss_gradient(self, X, y, w):
        """Computes the gradient of the rodeo RMSE loss at location w. 

        Args:
           X (np.array): G x d, prediction at G grid point locations from self.d experts
           y (np.array): G x 1, ground truth at G grid points
           w (np.array): d x 1, location at which to compute gradient.
        """
        G = X.shape[0] # Number of grid points
        d = X.shape[1] # Number of experts 

        err = X @ w - y

        if np.isclose(err, np.zeros(err.shape)).all():
            return np.zeros((d,))

        return (X.T @ err / \
            (np.sqrt(G)*np.linalg.norm(err, ord=2))).reshape(-1,)

class HintingLossODAFTRL(object): 
    def __init__(self, q):
        if q == 2 or q == np.inf:
            self.q = q
        else:
            raise ValueError(f"Gradients for {q}-norm not implemented. Use q = 2 or infty")

    def loss_gradient(self, H, y, om, g):
        """Computes the hint loss location om. 

        Args:
           H: d x n np array - prediction from self.n hinters 
           y: d x 1 np.array - ground truth cumulative loss 
           om: n x 1 np.array - omega weight play of hinter
           g (np.array) - d x 1 vector of gradient at time t
        """
        return np.linalg.norm(g, ord=self.q) * np.linalg.norm(y - H @ om, ord=self.q)
    
    def loss_gradient(self, H, y, om, g):
        """Computes the gradient of the hint loss location om. 

        Args:
           H: d x n np array - prediction from self.n hinters 
           y: d x 1 np.array - ground truth cumulative loss 
           om: n x 1 np.array - omega weight play of hinter
           g (np.array) - d x 1 vector of gradient at time t
        """
        # Unpack arguments
        d = H.shape[0] # Number of gradient values 
        n = H.shape[1] # Number of experts

        err = H @ om - y

        if self.q == np.inf:
            # Return the inf norm gradient
            max_idx = np.argmax(err)
            max_err = err[max_idx]

            if np.isclose(max_err, 0.0):
                return np.zeros((n,))

            err_sign = np.sign(max_err)
            g_norm = np.linalg.norm(g, ord=self.q)

            # Gradient of the inf-norm
            return (g_norm  * err_sign * H.T[:, max_idx]).reshape(-1,)  
        else:
            # Return the q-norm gradient
            err_norm = np.linalg.norm(err, ord=self.q)
            if np.isclose(err_norm, 0.0):
                return np.zeros((n,))

            g_norm = np.linalg.norm(g, ord=self.q)

            # Gradient of the inf-norm
            return ((g_norm / err_norm) * (H.T @ err)).reshape(-1,)

class HintingLossDOOMD(object): 
    def __init__(self):
        pass

    def loss(self, **kwargs):
        """Computes the MSE loss. 

        Args:
           y: d x 1 np.array - true cumulative loss gradient, t-2D:t-D
           g_hat: d x 1 np.array - predicted loss gradients
           g_tD: d x 1 np.array -  true loss gradient at t-2D
           h_t: d x 1 np.array -  hint at time t-D
           h_tp: d x 1 np.array -  hint at time t-D+1
        """     
        # Unpack arguments
        g_os = kwargs['y']
        g_hat = kwargs['g_hat']
        g_tD = kwargs['g_tD']
        h_t = kwargs['hint_t']
        h_tp = kwargs['hint_tp']

        return np.linalg.norm(g_tD + h_tp - h_t, ord=2) * np.linalg.norm(g_hat - g_os, ord=2)
    
    def loss_experts(self, **kwargs):
        """Computes the MSE loss for each expert. 

        Args:
           X: d x n np array - prediction at d gradient values from n experts        
           y: d x 1 np.array - true cumulative loss gradient, t-2D:t-D
           g_hat: d x 1 np.array - predicted loss gradients
           g_tD: d x 1 np.array -  true loss gradient at t-2D
           h_t: d x 1 np.array -  hint at time t-D
           h_tp: d x 1 np.array -  hint at time t-D+1
        """    
        # Unpack arguments
        X = kwargs['X']
        g_os = kwargs['y']
        g_tD = kwargs['g_tD']
        h_t = kwargs['hint_t']
        h_tp = kwargs['hint_tp']

        n = X.shape[1] 
        return np.linalg.norm(g_tD, ord=2) * np.linalg.norm(X - np.matlib.repmat(g_os.reshape(-1, 1), 1, n), ord=2, axis=0)
    
    def loss_gradient(self, **kwargs):
        """Computes the gradient of the MSE loss at location w. 

        Args:
           X: d x n np array - prediction from self.d experts
           y: d x 1 np.array - ground truth 
           w: n x 1 np.array - location at which to compute gradient.
           g_tD: d x 1 np.array -  true loss gradient at t-2D
           h_t: d x 1 np.array -  hint at time t-D
           h_tp: d x 1 np.array -  hint at time t-D+1
        """
        # Unpack arguments
        X = kwargs['X']
        y = kwargs['y']
        w = kwargs['w']
        g_tD = kwargs['g_tD']
        h_t = kwargs['hint_t']
        h_tp = kwargs['hint_tp']

        d = X.shape[0] # Number of gradient values 
        n = X.shape[1] # Number of experts

        # Add default values
        err = X @ w - y
        err_norm = np.linalg.norm(err, ord=2)
        g_norm = np.linalg.norm(g_tD + h_tp - h_t, ord=2)

        if np.isclose(err_norm, 0.0):
            return np.zeros((n,))

        return ((g_norm / err_norm) * (X.T @ err)).reshape(-1,)

class HintingLossODAFTRL(object): 
    def __init__(self, q):
        if q == 2 or q == np.inf:
            self.q = q
        else:
            raise ValueError(f"Gradients for {q}-norm not implemented. Use q = 2 or infty")

    def loss(self, **kwargs):
        """Computes the MSE loss. 

        Args:
           y: d x 1 np.array - true cumulative loss gradient
           y_prev: d x 1 np.array - true previous loss gradient
           y_hat: d x 1 np.array - predicted loss gradient 
        """     
        # Unpack arguments
        g_os = kwargs['y']
        g_hat = kwargs['g_hat']
        g_t = kwargs['g_t']

        return np.linalg.norm(g_t, ord=self.q) * np.linalg.norm(g_os - g_hat, ord=self.q)
    
    def loss_experts(self, **kwargs):
        """Computes the MSE loss for each expert. 

        Args:
           X: d x n np array - prediction at d gradient values from n experts        
           y_prev: d x 1 np.array - true previous loss gradient
           y: d x 1 np.array - ground truth gradient value at d points
        """    
        # Unpack arguments
        X = kwargs['X']
        y = kwargs['y']
        g_t = kwargs['g_t']

        n = X.shape[1] 
        return np.linalg.norm(g_t, ord=self.q) * np.linalg.norm(X - np.matlib.repmat(y.reshape(-1, 1), 1, n), ord=self.q, axis=0)
    
    def loss_gradient(self, **kwargs):
        """Computes the gradient of the MSE loss at location w. 

        Args:
           X: d x n np array - prediction from self.d experts
           y: d x 1 np.array - ground truth 
           w: n x 1 np.array - location at which to compute gradient.
        """
        # Unpack arguments
        X = kwargs['X']
        y = kwargs['y']
        w = kwargs['w']

        d = X.shape[0] # Number of gradient values 
        n = X.shape[1] # Number of experts

        # Add default values
        g_t = kwargs.get('g_t', np.zeros((d,)))

        err = X @ w - y

        if self.q == np.inf:
            # Return the inf norm gradient
            max_idx = np.argmax(err)
            max_err = err[max_idx]
            err_sign = np.sign(max_err)
            g_norm = np.linalg.norm(g_t, ord=self.q)

            # TODO: Which subgradient to return 
            if np.isclose(max_err, 0.0):
                return np.zeros((n,))

            # Gradient of the inf-norm
            return (g_norm  * err_sign * X.T[:, max_idx]).reshape(-1,)  
        else:
            # Return the q-norm gradient
            err_norm = np.linalg.norm(err, ord=self.q)
            g_norm = np.linalg.norm(g_t, ord=self.q)

            # TODO: Which subgradient to return 
            if np.isclose(err_norm, 0.0):
                return np.zeros((n,))

            # Gradient of the inf-norm
            return ((g_norm / err_norm) * (X.T @ err)).reshape(-1,)

class HintingLossDOOMD(object): 
    def __init__(self):
        pass

    def loss(self, **kwargs):
        """Computes the MSE loss. 

        Args:
           y: d x 1 np.array - true cumulative loss gradient, t-2D:t-D
           g_hat: d x 1 np.array - predicted loss gradients
           g_tD: d x 1 np.array -  true loss gradient at t-2D
           h_t: d x 1 np.array -  hint at time t-D
           h_tp: d x 1 np.array -  hint at time t-D+1
        """     
        # Unpack arguments
        g_os = kwargs['y']
        g_hat = kwargs['g_hat']
        g_tD = kwargs['g_tD']
        h_t = kwargs['hint_t']
        h_tp = kwargs['hint_tp']

        return np.linalg.norm(g_tD + h_tp - h_t, ord=2) * np.linalg.norm(g_hat - g_os, ord=2)
    
    def loss_experts(self, **kwargs):
        """Computes the MSE loss for each expert. 

        Args:
           X: d x n np array - prediction at d gradient values from n experts        
           y: d x 1 np.array - true cumulative loss gradient, t-2D:t-D
           g_hat: d x 1 np.array - predicted loss gradients
           g_tD: d x 1 np.array -  true loss gradient at t-2D
           h_t: d x 1 np.array -  hint at time t-D
           h_tp: d x 1 np.array -  hint at time t-D+1
        """    
        # Unpack arguments
        X = kwargs['X']
        g_os = kwargs['y']
        g_tD = kwargs['g_tD']
        h_t = kwargs['hint_t']
        h_tp = kwargs['hint_tp']

        n = X.shape[1] 
        return np.linalg.norm(g_tD, ord=2) * np.linalg.norm(X - np.matlib.repmat(g_os.reshape(-1, 1), 1, n), ord=2, axis=0)
    
    def loss_gradient(self, **kwargs):
        """Computes the gradient of the MSE loss at location w. 

        Args:
           X: d x n np array - prediction from self.d experts
           y: d x 1 np.array - ground truth 
           w: n x 1 np.array - location at which to compute gradient.
           g_tD: d x 1 np.array -  true loss gradient at t-2D
           h_t: d x 1 np.array -  hint at time t-D
           h_tp: d x 1 np.array -  hint at time t-D+1
        """
        # Unpack arguments
        X = kwargs['X']
        y = kwargs['y']
        w = kwargs['w']
        g_tD = kwargs['g_tD']
        h_t = kwargs['hint_t']
        h_tp = kwargs['hint_tp']

        d = X.shape[0] # Number of gradient values 
        n = X.shape[1] # Number of experts

        # Add default values
        err = X @ w - y
        err_norm = np.linalg.norm(err, ord=2)
        g_norm = np.linalg.norm(g_tD + h_tp - h_t, ord=2)

        if np.isclose(err_norm, 0.0):
            return np.zeros((n,))

        return ((g_norm / err_norm) * (X.T @ err)).reshape(-1,)