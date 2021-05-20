import numpy as np
from functools import partial
from datetime import datetime, timedelta
import pdb

from s2s_environment import S2SEnvironment, RodeoLoss
from s2s_hints import S2SHinter
from src.utils.eval_util import get_target_dates
from src.utils.experiments_util import get_start_delta

from poold.utils import History, loss_regret
from poold import create

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
learner, history = create("adahedged", model_list=models, partition=partition, T=T)

# Subseasonal forecasting environment
s2s_env = S2SEnvironment(dates, models, gt_id=gt_id, horizon=horizon)

# Subseasonal forecasting hinter
# horizon_hints = {"1day": ["prev_y"], 
#                  "12w": ["mean_g", "prev_g"],
#                  "34w": ["trend_y", "prev_g"],
#                  "future": ["prev_g"],
#                  "default":["prev_g"]}  

horizon_hints = {"1day": ["mean_g", "prev_g"], 
                 "12w": ["mean_g", "prev_g"],
                 "34w": ["mean_g", "prev_g"],
                 "future": ["mean_g", "prev_g"],
                 "default": ["mean_g", "prev_g"]}  

s2s_hinter = S2SHinter(hint_types=horizon_hints, gt_id=gt_id, horizon=horizon, dim=len(models), 
                    s2s_env=s2s_env, s2s_history=history, 
                    loss_gradient=s2s_env.rodeo_loss.loss_gradient,
                    loss_regret=loss_regret, partition=partition,
                    regret_hints=False, hz_hints=False)

# Iterate through algorithm times
for t in range(T):
    # Initialize round
    history.init_round(t)

    # Get available feedback
    times_fb = s2s_env.get_feedback(t, os_times=history.os_preds)
    losses_fb = [s2s_env.get_loss(t_fb) for t_fb in times_fb]

    # Get hinter before considering feedback
    hint = s2s_hinter.get_hint(t, history.os_preds)

    # Update learner with hint and feedback 
    w = learner.update(t, times_fb, losses_fb, hint)

    # Record play
    history.record_play(t, w)
    print(f"t: {t} lam: {learner.lam} \n\t w: {w}")

    # Update history and hinter
    for t_fb, loss_fb in zip(times_fb, losses_fb):
        history.record_loss(t_fb, loss_fb)
        s2s_hinter.update_hint_data(t_fb)



