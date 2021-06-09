# System imports
import numpy as np
from functools import partial
from datetime import datetime, timedelta
import copy
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Subseasonal forecasting imports
from src.s2s_environment import S2SEnvironment 
from src.s2s_hints import S2SHinter
from src.s2s_hint_environment import S2SHintEnvironment
from src.utils.eval_util import get_target_dates
from src.utils.experiments_util import get_start_delta
from vis_params import model_alias, alg_naming, style_algs

# PoolD imports
from poold import create
from poold.utils import loss_regret
from poold.utils import visualize

#TODO: remove this import
import pdb

# Set print parameters
np.set_printoptions(precision=3)

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon 

parser.add_argument('--target_dates', '-t', default="std_contest_eval")
parser.add_argument('--expert_models', '-m', default="tuned_catboost,tuned_cfsv2,tuned_doy,llr,multillr,tuned_salient_fri",
                    help="Comma separated list of models e.g., 'doy,cfsv2,llr,multillr'")
parser.add_argument('--reg', '-r', default="None",
                    help="Regularization type, one of: 'None', 'dub', 'adahedged'")
parser.add_argument('--alg', '-a', default="dormplus",
                    help="Online learning algorithm. One of: 'dorm', dormplus', 'adahedged', 'dub'")
parser.add_argument('--hint_alg', '-ha', default="None",
                    help="Algorithm to use for hint learning. Set to None to not hint learning.")
parser.add_argument('--hint', '-hi', default="None",
                    help="Optimistic hint type. Comma separated list of hint types.")
# parser.add_argument('--delay', '-d', default=0,
#                     help='Delay parameter, number of experts to instantiate. String containting an integer >=0.')      
args, opt = parser.parse_known_args()

# Task parameters
gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
horizon = args.pos_vars[1] # "34w" or "56w"    

date_str = args.target_dates # target date object
model_string = args.expert_models # string of expert prediction, comma separated
reg = args.reg # algorithm regularization 
alg = args.alg # algorithm 
hint_alg = args.hint_alg # algorithm 
hint_type = args.hint # type of optimistic hint
# delay_param = int(args.delay) # delay parameter 

# Perpare experts, sort model names, and get selected submodel for each
models = model_string.split(',')
models.sort()

# Perpare experts, sort model names, and get selected submodel for each
hint_options = hint_type.split(',')
hint_options.sort()

# Subseasonal forecasting hinter
learn_to_hint = (hint_alg != "None")
hz_hints = False
regret_hints = False if alg == "adahedged" else True

# Set alias for online learner
model_alias["online_learner"] = f"{alg_naming[alg]}"

# Forecast targets
targets = get_target_dates(date_str=date_str, horizon=horizon) # forecast target dates
# targets = targets[175:205] # TODO: delete this
# targets = targets[-26:] # TODO: delete this
targets_pred = copy.deepcopy(targets) # targets we successfully make forecasts for 

start_delta = timedelta(days=get_start_delta(horizon, gt_id)) # difference between issuance + target
dates = [t - start_delta for t in targets] # forecast issuance dates
T = len(dates) # algorithm duration 

# Online learning algorithm 
learner = create(alg, model_list=models, groups=None, T=26)
regret_periods = []

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
    hint_groups = [i for i, sublist in enumerate(horizon_hints.values()) \
        for item in sublist]
else:
    horizon_hints = {"default": hint_options}  
    n_hints = [sum(len(x) for x in horizon_hints)]
    hint_models = ["h_" + "".join(item) \
        for i, item in enumerate(horizon_hints["default"])]
    hint_groups = [0 for i, sublist in enumerate(horizon_hints["default"])]

# Set up hinter (produces hints for delay period)
s2s_hinter = S2SHinter(
    hint_types=horizon_hints, gt_id=gt_id, horizon=horizon, learner=learner, 
    environment=s2s_env, hint_groups=hint_groups, regret_hints=regret_hints, 
    hz_hints=hz_hints)

# Set up hint environment (manages losses and ground truth for hinter) 
s2s_hint_env = S2SHintEnvironment(
    dates, hint_models, gt_id=gt_id, horizon=horizon, learner=learner)

# Create hint learner
if learn_to_hint:
    hint_learner = create(hint_alg, model_list=hint_models, groups=hint_groups, T=26)

t_pred = 0 # number of successful predictions made
period_start = 0 # start of regret period

# Iterate through algorithm times
for t in range(T):
    print("------ Starting round", t)
    if t % 26 == 0 and t != 0:
        # Get the remainder of the losses
        losses_fb = s2s_env.get_losses(
            t, os_times=learner.get_outstanding(include=False), override=True)
        learner.history.record_losses(losses_fb)
        learner.reset_params(T=26)

        #  Get the remainder of the hint losses
        s2s_hinter.reset_hint_data()
        if learn_to_hint:
            hint_losses_fb = s2s_hint_env.get_losses(
                t, os_times=hint_learner.get_outstanding(include=False), override=True)
            hint_learner.history.record_losses(hint_losses_fb)
            hint_learner.reset_params(T=26)

        # Record that start of a new regret period
        regret_periods.append((period_start, t_pred))
        period_start = t_pred

    # Check expert predictions
    pred = s2s_env.check_pred(t)
    if pred is False:
        print(f"Missing expert predictions for round {t}.")
        del targets_pred[t]
        learner.t += 1 # increment learner as well
        continue 

    # Get available learner feedback
    os_times = learner.get_outstanding()
    losses_fb = s2s_env.get_losses(t, os_times=os_times)

    # Update hinter
    s2s_hinter.update_hint_data(t, losses_fb)
    hint_os_times = copy.copy(os_times)
    for t_fb, loss_fb in losses_fb:
        hint_os_times.remove(t_fb)

    if learn_to_hint:
        H = s2s_hinter.get_hint_matrix(t, hint_os_times) 

        # Record hint output
        s2s_hint_env.log_hint_matrix(t, H)

        # Get available hinter feedback
        hint_losses_fb = s2s_hint_env.get_losses(t, os_times=os_times)

        # Update hint learner with feedback
        omega = hint_learner.update_and_play(hint_losses_fb, hint=None)

        # Create hint
        hint = (H @ omega).reshape(-1,)
        h = {"grad": lambda w: hint}
    else:
        # Create hint
        h = s2s_hinter.get_hint(t, hint_os_times)

    print("Hint:", h['grad'](learner.w))
    # Update learner with hint and feedback 
    w = learner.update_and_play(losses_fb, hint=h)

    # Display metrics
    print(learner.log_params(t))
    if learn_to_hint:
        print(hint_learner.log_params(t))

    # Increment number of successful predictions
    t_pred += 1

# Update the final regret period
regret_periods.append((period_start, t_pred))

# Get the remainder of the losses
losses_fb = s2s_env.get_losses(T-1, os_times=learner.get_outstanding(include=False), override=True)
learner.history.record_losses(losses_fb)

exp_string = f"{gt_id}_{horizon}_{date_str}_A{alg}_HL{hint_alg if learn_to_hint else ('-').join(hint_options)}"
fl = open(f"learner_history_{exp_string}.pickle", "wb")
pickle.dump([targets_pred, regret_periods, model_alias, learner.history], fl)

visualize(learner.history, regret_periods, targets_pred, model_alias, style_algs)

if learn_to_hint:
    hint_losses_fb = s2s_hint_env.get_losses(T-1, os_times=learner.get_outstanding(include=False), override=True)
    hint_learner.history.record_losses(hint_losses_fb)

    fh = open(f"hinter_history_{exp_string}.pickle", "wb")
    pickle.dump([targets_pred, regret_periods, {}, hint_learner.history], fh)

    # Visualize history
    visualize(hint_learner.history, regret_periods, targets_pred)

plt.show()