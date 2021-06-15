""" Run simple, non-optimisic learner.

Example usage:
    python run_learner.py contest_precip 56w --alg adahedged --vis True 
"""
# System imports
import numpy as np
from datetime import datetime, timedelta
import copy
import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# Subseasonal forecasting imports
from src.s2s_environment import S2SEnvironment 
from src.utils.eval_util import get_target_dates
from src.utils.experiments_util import get_start_delta
from src.s2s_vis_params import model_alias, alg_naming, style_algs

# PoolD imports
from poold import create
from poold.utils import loss_regret
from poold.utils import visualize

# Set print parameters
np.set_printoptions(precision=6)

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon 

parser.add_argument('--target_dates', '-t', default="std_contest_eval")
parser.add_argument('--expert_models', '-m', default="tuned_catboost,tuned_cfsv2,tuned_doy,llr,multillr,tuned_salient_fri",
                    help="Comma separated list of models e.g., 'doy,cfsv2,llr,multillr'")
parser.add_argument('--alg', '-a', default="dormplus",
                    help="Online learning algorithm. One of: 'dorm', dormplus', 'adahedged', 'dub'")
parser.add_argument('--visualize', '-vis', default=False,
                    help="Visualize online learning output.")
args, opt = parser.parse_known_args()

# Task parameters
gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
horizon = args.pos_vars[1] # "34w" or "56w"    

date_str = args.target_dates # target date object
model_string = args.expert_models # string of expert prediction, comma separated
alg = args.alg # algorithm 
vis = bool(args.visualize)

# Perpare experts, sort model names, and get selected submodel for each
models = model_string.split(',')
models.sort()

# Set alias for online learner
model_alias["online_learner"] = f"{alg_naming[alg]}"

# Forecast targets
targets = get_target_dates(date_str=date_str, horizon=horizon) # forecast target dates
targets_missed = [] # targets we do not make predictions for
period_length = 26 # yearly regret periods/resetting

start_delta = timedelta(days=get_start_delta(horizon, gt_id)) # difference between issuance + target
dates = [t - start_delta for t in targets] # forecast issuance dates
T = len(dates) # algorithm duration 

# Online learning algorithm 
learner = create(alg, model_list=models, groups=None, T=period_length)

# Subseasonal forecasting environment
s2s_env = S2SEnvironment(dates, models, gt_id=gt_id, horizon=horizon)

regret_periods = [] # start and end of 
t_pred = 0 # number of successful predictions made
period_start = 0 # start of regret period

# Iterate through algorithm times
for t in range(T):
    print(" >>> Starting round", t)
    if t % period_length == 0 and t != 0:
        # Get the remainder of the losses
        losses_fb = s2s_env.get_losses(
            t, os_times=learner.get_outstanding(include=False), override=True)
        learner.history.record_losses(losses_fb)
        learner.reset_params(T=period_length)

        # Record that start of a new regret period
        regret_periods.append((period_start, t_pred))
        period_start = t_pred

    # Check expert predictions
    pred = s2s_env.check_pred(t)
    if pred is False:
        print(f"Missing expert predictions for round {t}.")
        targets_missed.append(t)
        learner.increment_time() # increment learner as well
        continue 

    # Get available learner feedback
    os_times = learner.get_outstanding()
    losses_fb = s2s_env.get_losses(t, os_times=os_times)

    # Update learner with hint and feedback 
    w = learner.update_and_play(losses_fb, hint=None)

    # Display metrics
    print(learner.log_params(t))

    # Increment number of successful predictions
    t_pred += 1

# Update the final regret period
regret_periods.append((period_start, t_pred))
targets_pred = copy.deepcopy(targets)
for index in sorted(targets_missed, reverse=True):
    del targets_pred[index]

# Get the remainder of the losses
losses_fb = s2s_env.get_losses(T-1, os_times=learner.get_outstanding(include=False), override=True)
learner.history.record_losses(losses_fb)

exp_string = f"{gt_id}_{horizon}_{date_str}_A{alg}"
fl = open(f"experiments/learner_history_{exp_string}.pickle", "wb")
pickle.dump([targets_pred, learner.history], fl)

if vis:
    # Visualize history
    visualize(learner.history, regret_periods, targets_pred, model_alias, style_algs)
    plt.show()