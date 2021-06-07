# System imports
import numpy as np
from functools import partial
from datetime import datetime, timedelta
import copy
import pickle
from functools import partial

# Subseasonal forecasting imports
from src.s2s_environment import S2SEnvironment 
from src.s2s_hints import S2SHinter
from src.s2s_hint_environment import S2SHintEnvironment
from src.utils.eval_util import get_target_dates
from src.utils.experiments_util import get_start_delta
from vis_params import *

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

# Subseasonal forecasting hinter
learn_to_hint = False 
hz_hints = False
hint_alg = "adahedged"
constant_hint = ["prev_g"]
regret_hints = False if alg == "adahedged" else True

# Set alias for online learner
model_alias["online_learner"] = f"{alg_naming[alg]}, Hint: {alg_naming[hint_alg] if learn_to_hint else constant_hint[0]}"

# Task parameters
horizon = "56w"
gt_id = "contest_precip"

# Input models
models = ["tuned_doy", "tuned_cfsv2", "tuned_salient_fri", "tuned_catboost", "multillr", "llr"]

# Forecast targets
date_str = "std_contest_eval"
targets = get_target_dates(date_str=date_str, horizon=horizon) # forecast target dates
# targets = targets[340:370] # TODO: delete this
targets = targets[-26:] # TODO: delete this

start_delta = timedelta(days=get_start_delta(horizon, gt_id)) # difference between issuance + target
dates = [t - start_delta for t in targets] # forecast issuance dates
T = len(dates) # algorithm duration 

partition = None

# Online learning algorithm 
learner = create(alg, model_list=models, partition=partition, T=T)

# Subseasonal forecasting environment
s2s_env = S2SEnvironment(dates, models, gt_id=gt_id, horizon=horizon)

# Get name and parition for each of the hints
if hz_hints:
    horizon_hints = {"1day": ["prev_y"], 
                    "12w": ["mean_g", "prev_g", "trend_y"],
                    "34w": ["mean_g", "prev_g", "trend_y"],
                    "future": ["mean_g", "prev_g", "trend_y"],
                    "default":["prev_g"]}  
    n_hints = [sum(len(x) for x in horizon_hints)]
    hint_models = ["h" + str(i) + "_" + "".join(item) \
        for i, sublist in enumerate(horizon_hints.values()) \
            for item in sublist]
    hint_partition = [i for i, sublist in enumerate(horizon_hints.values()) \
        for item in sublist]
else:
    horizon_hints = {"default": constant_hint}  
    n_hints = [sum(len(x) for x in horizon_hints)]
    hint_models = ["h" + str(i) + "_" + "".join(item) \
        for i, sublist in enumerate(horizon_hints["default"]) \
            for item in sublist]
    hint_partition = [i for i, sublist in enumerate(horizon_hints["default"])]

# Instantiate hinter
# s2s_hinter = S2SHinter(hint_types=horizon_hints, gt_id=gt_id, 
#                     horizon=horizon, dim=len(models), 
#                     s2s_env=s2s_env, s2s_history=history, 
#                     loss_gradient=s2s_env.rodeo_loss.loss_gradient,
#                     loss_regret=partial(loss_regret, partition=learner.partition), 
#                     partition=hint_partition, regret_hints=regret_hints, hz_hints=hz_hints)

# s2s_hint_env = S2SHintEnvironment(dates, hint_models, gt_id=gt_id, horizon=horizon, learner=learner)
# Create hint learner
# if learn_to_hint:
    # hint_learner, hint_history = create(hint_alg, model_list=hint_models, partition=hint_partition, T=T)

# Iterate through algorithm times
for t in range(T):
    # Check expert predictions
    pred = s2s_env.check_pred(t)
    if pred is False:
        print(f"Missing expert predictions for round {t}.")
        continue 

    # Initialize round
    # history.init_round(t)
    # if learn_to_hint:
        # hint_history.init_round(t)

    # Get available learner feedback
    losses_fb = s2s_env.get_losses(t, os_times=history.os_preds)

    # Record learner losses 
    # learner.record_loss(t, times_fb, losses_fb)
    # history.record_loss(t, times_fb, losses_fb)

    # Update hinter
    # s2s_hinter.update_hint_data(t, times_fb)

    # if learn_to_hint:
    #     H = s2s_hinter.get_hint_matrix(t, history.os_preds)

    #     # Record hint output
    #     s2s_hint_env.log_hint_matrix(t, H)

    #     # Get available hinter feedback
    #     # hint_times_fb = s2s_hint_env.get_feedback(t, os_times=history.os_preds)
    #     hint_times_fb = times_fb
    #     hint_losses_fb = [s2s_hint_env.get_loss(t_fb) for t_fb in times_fb]

    #     # Record hinter losses
    #     hint_learner.record_loss(t, hint_times_fb, hint_losses_fb)
    #     if learn_to_hint:
    #         hint_history.record_loss(t, hint_times_fb, hint_losses_fb)

    #     # Update hint learner with feedback
    #     omega = hint_learner.update(t, hint_times_fb, hint=None)

    #     # Record hint play
    #     if learn_to_hint:
    #         hint_history.record_play(t, omega)

    #     # Create hint
    #     h = {"g": H @ omega}
    # else:
    #     # Create hint
    #     h = s2s_hinter.get_hint(t, history.os_preds)

    # Update learner with hint and feedback 
    # w = learner.update(t, times_fb, hint=h)
    w = learner.update(t, times_fb, hint=None)

    # Record play
    # history.record_play(t, w)
    print(f"t: {t} \n w: {w} \n params: {learner.get_learner_params()}")
    # if learn_to_hint:
        # print(f"\t omega: {omega} \n\t params: {hint_learner.get_learner_params()}")

# Record the remainder of the losses
losses_fb = s2s_env.get_losses(t, os_times=history.os_preds)

# Record learner losses 
# learner.record_loss(T, times_fb, losses_fb)
# history.record_loss(T, times_fb, losses_fb)


exp_string = f"{gt_id}_{horizon}_{date_str}_A{alg}_HL{hint_alg if learn_to_hint else constant_hint[0]}_HZ{hz_hints}"
fl = open(f"learner_history_{exp_string}.pickle", "wb")
pickle.dump([targets, learner.history], fl)
# if learn_to_hint:
#     fh = open(f"hint_history_{exp_string}.pickle", "wb")
#     pickle.dump([targets, hint_history], fh)

# Visualize history
# visualize(learner.history, targets, model_alias)
# if learn_to_hint:
#     visualize(hint_history, targets)