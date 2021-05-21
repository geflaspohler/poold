# System imports
import numpy as np
from functools import partial
from datetime import datetime, timedelta

# Subseasonal forecasting imports
from src.s2s_environment import S2SEnvironment 
from src.s2s_hints import S2SHinter
from src.s2s_hint_environment import S2SHintEnvironment
from src.utils.eval_util import get_target_dates
from src.utils.experiments_util import get_start_delta

# PoolD imports
from poold import create
from poold.utils import loss_regret

#TODO: remove this import
import pdb

# Task parameters
horizon = "34w"
gt_id = "contest_precip"

# Input models
models = ["tuned_doy", "tuned_cfsv2", "tuned_salient_fri", "tuned_catboost", "multillr", "llr"]

# Forecast targets
targets = get_target_dates(date_str="std_contest", horizon=horizon) # forecast target dates
start_delta = timedelta(days=get_start_delta(horizon, gt_id)) # difference between issuance + target
dates = [t - start_delta for t in targets] # forecast issuance dates
T = len(dates) # algorithm duration 

partition = None

# Online learning algorithm 
learner, history = create("dormplus", model_list=models, partition=partition, T=T)

# Subseasonal forecasting environment
s2s_env = S2SEnvironment(dates, models, gt_id=gt_id, horizon=horizon)
s2s_hint_env = S2SHintEnvironment(dates, models, gt_id=gt_id, horizon=horizon, learner=learner)

# Subseasonal forecasting hinter
learn_to_hint = True
hz_hints = True
if hz_hints:
    horizon_hints = {"1day": ["prev_y"], 
                    "12w": ["mean_g", "prev_g"],
                    "34w": ["trend_y", "prev_g"],
                    "future": ["prev_g"],
                    "default":["prev_g"]}  
else:
    horizon_hints = {"default": ["mean_g", "prev_g"]}  

#  Get name and parition for each of the hints
n_hints = [sum(len(x) for x in horizon_hints)]
if hz_hints: 
    hint_models = ["h" + str(i) + "_" + "".join(item) for i, sublist in enumerate(horizon_hints.values()) for item in sublist]
    hint_partition = [i for i, sublist in enumerate(horizon_hints.values()) for item in sublist]
else:
    hint_models = ["h" + str(i) + "_" + "".join(item) for i, sublist in enumerate(horizon_hints["default"]) for item in sublist]
    hint_partition = [i for i, sublist in enumerate(horizon_hints["default"]) for item in sublist]    

s2s_hinter = S2SHinter(hint_types=horizon_hints, gt_id=gt_id, 
                    horizon=horizon, dim=len(models), 
                    s2s_env=s2s_env, s2s_history=history, 
                    loss_gradient=s2s_env.rodeo_loss.loss_gradient,
                    loss_regret=loss_regret, partition=hint_partition,
                    regret_hints=False, hz_hints=hz_hints)

if learn_to_hint:
    hint_learner, hint_history = create("dormplus", model_list=hint_models, partition=hint_partition, T=T)

# Iterate through algorithm times
for t in range(T):
    # Initialize round
    history.init_round(t)
    hint_history.init_round(t)

    # Get available learner feedback
    times_fb = s2s_env.get_feedback(t, os_times=history.os_preds)
    losses_fb = [s2s_env.get_loss(t_fb) for t_fb in times_fb]
    learner.record_loss(t, times_fb, losses_fb)

    if learn_to_hint:
        H = s2s_hinter.get_hint_matrix(t, history.os_preds)
        s2s_hint_env.log_hint_matrix(t, H)

        # Get available hinter feedback
        hint_times_fb = s2s_hint_env.get_feedback(t, os_times=history.os_preds)
        hint_losses_fb = [s2s_hint_env.get_loss(t_fb) for t_fb in hint_times_fb]
        hint_learner.record_loss(t, hint_times_fb, hint_losses_fb)

        # Update hint learner with feedback
        om = hint_learner.update(t, hint_times_fb, hint=None)
        hint_history.record_play(t, om)

        for t_fb, loss_fb in zip(hint_times_fb, hint_losses_fb):
            hint_history.record_loss(t_fb, loss_fb)

        # Create hint
        h = {"g": H @ om}
    else:
        h = s2s_hinter.get_hint(t, history.os_preds)

    # Update learner with hint and feedback 
    w = learner.update(t, times_fb, hint=h)

    # Record play
    history.record_play(t, w)
    print(f"t: {t} \n\t w: {w} \n\t params: {learner.get_learner_params()}")
    print(f"\t\t om: {om} \n\t\t params: {hint_learner.get_learner_params()}")

    # Update history and hinter
    for t_fb, loss_fb in zip(times_fb, losses_fb):
        history.record_loss(t_fb, loss_fb)
        s2s_hinter.update_hint_data(t_fb)



