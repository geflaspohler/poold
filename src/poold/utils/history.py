
""" History object for book-keeping online learning progress. 

For example:

    import poold

    models = ["model1", "model2"]
    duration = 20
    learner, history = poold.create("adahedged", model_list=models, T=duration)
"""
import copy
import numpy as np
import pdb

class History(object):
    def __init__(self, models, default_play, low_memory=False):
        """ Online leanring history object.

        Args:
            models (list): list of model names
            T (int): algorithm duration
        """
        self.models = models
        self.d = len(models)
        self.default_play = default_play

        self.play_history = {} # history of past algorithm plays
        self.loss_history = {} # history of past observed loss objects
        self.hint_history = {} # history of past observed loss objects
        self.grad_history = {} # history of past observed gradients
        self.param_history = {} # history of past parameters 
        self.os_history = {} # history of outstanding feedbacks 
        self.realized_losses = {} # history of past realized losses

        self.low_memory = low_memory # if True, avoid saving full loss functions

    def get_times(self):
        """ Return history times """
        return list(self.play_history.keys())

    def record_play(self, t, w):
        """ Record play at round t.

        Args:
            t: a time representation 
            w: a play representation
        """
        self.play_history[t] = copy.copy(w)

    def record_losses(self, losses_fb, verbose=False):
        """ Record the received loss at time t.

        Args:
            losses_fb: list of (time, loss objects) tuples
        """
        for t_fb, loss_fb in losses_fb:
            # t_fb += self.learner_base_time
            assert(t_fb in self.play_history)
            if t_fb in self.grad_history:
                if verbose:
                    print(f"Warning: time {t_fb} is already in gradient history and won't be recomputed.")
                continue

            if not self.low_memory:
                self.loss_history[t_fb] = copy.deepcopy(loss_fb)

            self.grad_history[t_fb] = loss_fb['grad'](w=self.play_history[t_fb])
            self.realized_losses[t_fb] = loss_fb['fun'](w=self.play_history[t_fb])

    def record_hint(self, t, hint):
        """ Record the received hint at time t.

        Args:
            t: a time representation 
            hint (dict): hint dictionary  
        """
        # t += self.learner_base_time
        self.hint_history[t] = copy.deepcopy(hint)

    def record_params(self, t, params):
        """ Record the received hint at time t.

        Args:
            t: a time representation 
            params (dict): parameter dictionary  
        """
        # t += self.learner_base_time
        self.param_history[t] = copy.deepcopy(params)

    def record_os(self, t, os):
        """ Record the outstanding feedbacks at time t.

        Args:
            t: a time representation 
            os (list): list of oustanding feedback times
        """
        # t += self.learner_base_time
        self.os_history[t] = copy.deepcopy(os)

    def get(self, t):
        """ Get full history at time t """
        g = self.get_grad(t)
        w = self.get_play(t)
        h = self.get_hint(t)
        hp = self.get_hint(t-1)
        params = self.get_params(t)
        os = self.get_os(t)
        D = len(os)
        g_os = sum([self.get_grad(t_fb) for t_fb in os])

        return {
            "t": t, # time
            "w": w, # play
            "g": g, # gradient 
            "g_os": g_os, # outstanding gradient sum
            "h": h, # hint
            "hp": hp, # previous hint
            "D": D, # delay length 
            "params": params # parameters
        }

    def get_loss(self, t):
        """ Get the loss at time t """
        # t += self.learner_base_time
        assert(t in self.grad_history)
        if self.low_memory:
            return None, self.realized_losses[t], self.grad_history[t]
        return self.loss_history[t], self.realized_losses[t], self.grad_history[t]

    def get_grad(self, t):
        """ Get the loss gradient at time t """
        # t += self.learner_base_time
        assert(t in self.grad_history)
        return self.grad_history[t]

    def get_hint(self, t):
        """ Get the hint at time t """
        # t += self.learner_base_time
        if t not in self.hint_history:
            return np.zeros((self.d,))

        assert(t in self.hint_history)
        return self.hint_history[t]

    def get_params(self, t):
        """ Get the parameters at time t """
        # t += self.learner_base_time
        assert(t in self.param_history)
        return self.param_history[t]

    def get_os(self, t):
        """ Get the parameters at time t """
        # t += self.learner_base_time
        assert(t in self.os_history)
        return self.os_history[t]

    def get_play(self, t, return_past=True):
        """ Get the play at time t. If return_past is True,
        will return the play at t-1 if the play at time t 
        is not yet available.  
        """
        # Initial value before the first play
        if len(self.play_history) == 0:
            return copy.deepcopy(self.default_play)

        # If past play is in history, return most recent play
        if t not in self.play_history and return_past:
            return self.get_last_play()

        assert(t in self.play_history)
        return self.play_history[t]

    def get_last_play(self):
        """ Get the most recent play """ 
        t_max = max(list(self.play_history.keys()))
        return self.play_history[t_max]
