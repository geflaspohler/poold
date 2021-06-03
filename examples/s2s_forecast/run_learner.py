# System imports
import numpy as np
from functools import partial
from datetime import datetime, timedelta
import copy
import pickle
from functools import partial

# Subseasonal forecasting imports
from src.s2s_environment import S2SEnvironment 
from src.utils.eval_util import get_target_dates
from src.utils.experiments_util import get_start_delta
from vis_params import model_alias, alg_naming

# PoolD imports
from poold import create
from poold.utils import loss_regret
from poold.utils import visualize

#TODO: remove this import
import pdb

# Set print parameters
np.set_printoptions(precision=3)

# Learner parameters
alg = "dormplus"

# Set alias for online learner
model_alias["online_learner"] = f"{alg_naming[alg]}"

# Task parameters
horizon = "56w"
gt_id = "contest_precip"

# Input models
models = ["tuned_doy", "tuned_cfsv2", "tuned_salient_fri", "tuned_catboost", "multillr", "llr"]

# Forecast targets
date_str = "std_contest_eval"
targets = get_target_dates(date_str=date_str, horizon=horizon) # forecast target dates
targets = targets[-26:] # TODO: delete this

start_delta = timedelta(days=get_start_delta(horizon, gt_id)) # difference between issuance + target
dates = [t - start_delta for t in targets] # forecast issuance dates
T = len(dates) # algorithm duration 

# Online learning algorithm 
partition = None
learner = create(alg, model_list=models, partition=partition, T=T)

# Subseasonal forecasting environment
s2s_env = S2SEnvironment(dates, models, gt_id=gt_id, horizon=horizon)

# Iterate through algorithm times
for t in range(T):
    print("Starting round", t)
    # Check expert predictions
    pred = s2s_env.check_pred(t)
    if pred is False:
        print(f"Missing expert predictions for round {t}.")
        continue 

    # Get available learner feedback
    losses_fb = s2s_env.get_losses(t, os_times=learner.get_outstanding(t))

    # Update learner with hint and feedback 
    w = learner.update_and_play(t, losses_fb, hint=None)

    # Display metrics
    print(f"t: {t} \n w: {w} \n params: {learner.get_params()}")

# Get the remainder of the losses
losses_fb = s2s_env.get_losses(T-1, os_times=learner.get_outstanding(T, include=False), override=True)
learner.history.record_losses(T, losses_fb)

exp_string = f"{gt_id}_{horizon}_{date_str}_A{alg}"
fl = open(f"learner_history_{exp_string}.pickle", "wb")
pickle.dump([targets, learner.history], fl)

# Visualize history
visualize(learner.history, targets, model_alias)