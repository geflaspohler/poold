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
# delay_param = int(args.delay) # delay parameter 

# Perpare experts, sort model names, and get selected submodel for each
models = model_string.split(',')
models.sort()

# Set alias for online learner
model_alias["online_learner"] = f"{alg_naming[alg]}"

# Forecast targets
date_str = "std_contest_eval"
targets = get_target_dates(date_str=date_str, horizon=horizon) # forecast target dates
targets_pred = copy.deepcopy(targets) # targets we successfully make forecasts for 

start_delta = timedelta(days=get_start_delta(horizon, gt_id)) # difference between issuance + target
dates = [t - start_delta for t in targets] # forecast issuance dates
T = len(dates) # algorithm duration 

# Online learning algorithm 
learner = create(alg, model_list=models, groups=None, T=26)

# Subseasonal forecasting environment
s2s_env = S2SEnvironment(dates, models, gt_id=gt_id, horizon=horizon)

# Iterate through algorithm times
for t in range(T):
    print("------ Starting round", t)

    if t % 26 == 0 and t != 0:
        # Get the remainder of the losses
        losses_fb = s2s_env.get_losses(
            t, os_times=learner.get_outstanding(include=False), override=True)
        learner.history.record_losses(losses_fb)
        learner.reset_params(T=26)

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

    # Update learner with hint and feedback 
    w = learner.update_and_play(losses_fb, hint=None)

    # Display metrics
    print(learner.log_params(t))


# Get the remainder of the losses
losses_fb = s2s_env.get_losses(T-1, os_times=learner.get_outstanding(include=False), override=True)
learner.history.record_losses(losses_fb)


exp_string = f"{gt_id}_{horizon}_{date_str}_A{alg}"
fl = open(f"experiments/learner_history_{exp_string}.pickle", "wb")
pickle.dump([targets_pred, learner.history], fl)

# Visualize history
visualize(learner.history, targets_pred, model_alias)