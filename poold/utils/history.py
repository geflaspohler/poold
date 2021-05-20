
""" History object for book-keeping online learning progress. 

For example:

    import poold

    models = ["model1", "model2"]
    duration = 20
    learner, history = poold.create("adahedged", model_list=models, T=duration)
"""
import copy

class History(object):
    def __init__(self, w_init):
        """ Online leanring history object.

        Args:
            w_init: initialization value for online learner
        """
        self.play_history = {} # history of past algorithm plays
        self.loss_history = {} # history of past observed loss objects
        self.grads = {} # history of past observed gradients
        self.losses = {} # history of past realized losses
        self.os_preds = [] # list of currently outstanding prediction times

        # Initialize the first play to algorithm init value
        self.play_init = copy.copy(w_init)

    def init_round(self, t):
        """ Begin round t """
        # Add current time to oustanding predictions
        self.os_preds.append(t)

    def record_play(self, t, w):
        """ Record play at round t.

        Args:
            t: a time representation 
            w: a play representation
        """
        self.play_history[t] = copy.copy(w)

    def record_loss(self, t, loss):
        """ Record the received loss at time t.

        Args:
            t: a time representation 
            w: a loss representation
        """
        assert(t in self.play_history)
        self.loss_history[t] = copy.copy(loss)
        self.grads[t] = loss['jac'](w=self.play_history[t])
        self.losses[t] = loss['fun'](w=self.play_history[t])
        self.os_preds.remove(t)

    def get_loss(self, t):
        """ Get the loss at time t """
        assert(t in self.loss_history)
        return self.loss_history[t]

    def get_grad(self, t):
        """ Get the loss gradient at time t """
        assert(t in self.grads)
        return self.grads[t]

    def get_play(self, t, return_past=True):
        """ Get the play at time t. If return_past is True,
        will return the play at t-1 if the play at time t 
        is not yet available.  
        """
        # Return initial value before the first play
        if t == 0 and t not in self.play_history:
            return self.play_init

        # If past play is in history, return most recent play
        if t not in self.play_history and \
            return_past and t-1 in self.play_history:
            return self.play_history[t-1]

        assert(t in self.play_history)
        return self.play_history[t]

    def get_last_play(self):
        """ Get the most recent play """ 
        t_max = max(list(self.play_history.keys()))
        return self.play_history[t_max]
