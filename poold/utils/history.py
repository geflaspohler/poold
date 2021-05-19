from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd
import copy
import pdb 

from poold.utils import tic, toc, printf

class History(object):
    def __init__(self, w_init):
        self.play_history = {}
        self.loss_history = {}
        self.grads = {}
        self.losses = {}

        self.play_init = copy.copy(w_init)

    def record_play(self, t, w):
        self.play_history[t] = copy.copy(w)

    def update_loss(self, t, loss):
        assert(t in self.play_history)
        if t in self.play_history:
            self.loss_history[t] = copy.copy(loss)
            self.grads[t] = loss['jac'](w=self.play_history[t])
            self.losses[t] = loss['fun'](w=self.play_history[t])

    def get_loss(self, t):
        assert(t in self.loss_history)
        return self.loss_history[t]

    def get_grad(self, t):
        assert(t in self.grads)
        return self.grads[t]

    def get_play(self, t, return_past=True):
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
        t_max = max(list(self.play_history.keys()))
        return self.play_history[t_max]
