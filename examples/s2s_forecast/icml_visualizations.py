"""
Generate visualization figures and metrics for 
Flaspohler et al. Online Learning with Optimism and Delay, 
ICML 2021. 

Usage example:
    python icml_visualizations.py dormplus prev_g hinting --final_year

This will plot final year weights and regret for the 
hinting experiments using dormplus base algorithm.
"""
import pickle
import matplotlib.pyplot as plt
import copy
import os
from itertools import product
import pandas as pd
from argparse import ArgumentParser

# S2S imports
from src.s2s_vis_params import model_alias, alg_naming, style_algs, task_dict

# PoolD imports
from poold import create
from poold.utils import loss_regret
from poold.utils import visualize, visualize_multiple

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars", nargs="*")  # alg, hint, experiment
parser.add_argument('--final_year', '-fy', action='store_true')
args, opt = parser.parse_known_args()

alg = args.pos_vars[0] 
hint = args.pos_vars[1] 
experiment = args.pos_vars[2] 
final_year = args.final_year

def display_table(data_dict, model_list, model_alias={}, task_dict={}, filename="temp"):
    """Displays and saves dataframe after sorting """
    only_learner = False
    df = pd.DataFrame.from_dict(data_dict)
    df = df.rename(task_dict, axis=1)
    if only_learner:
        df = df.drop(model_list)
    df = df.T

    learners = list(set(df.columns).difference(set(model_list)))
    learners.sort()
    model_list.sort()

    df = df.reindex(learners + model_list, axis=1) # Sort alphabetically
    df = df.rename(model_alias, axis=1)
    align = "l" + "r"*len(learners) + "|" + "r"*len(model_list)
    tasks = list(task_dict.values())

    df = df.reindex(tasks) # Sort tasks; might not be stable as a dict

    if not os.path.exists('./eval'):
        os.mkdir('eval')
    fname = f"eval/all_task_losses_{filename}.tex"
    df.to_latex(fname, float_format="%.3f", longtable=False, column_format=align)
    return df

# Experiments home directory 
experiments_home = "experiments"
if not os.path.exists(experiments_home):
    oi.mkdir(experiments_home)

all_task_tables = {}
for gt_id, horizon in product(
        ["contest_tmp2m","contest_precip"], ["34w", "56w"]):

    task = f"{gt_id}_{horizon}"

    if experiment == "zoo":
        learner_list = [
        f"learner_history_{task}_std_contest_eval_Adormplus_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_Adorm_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_Aadahedged_HL{hint}.pickle",
        ]
    elif experiment == "hinting":
        learner_list = [
        f"learner_history_{task}_std_contest_eval_A{alg}_HLavg_recent_g.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLrecent_g.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLmean_g.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLNone.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLdormplus.pickle",
        ]
    elif experiment == "regularization":
        learner_list = [
        f"learner_history_{task}_std_contest_eval_Adub_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_Aadahedged_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HL{hint}.pickle",
        ]
    elif experiment == "rep":
        rep = 3 if horizon == "34w" else 4
        learner_list = [
        f"learner_history_{task}_std_contest_eval_A{alg}_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_R{rep}_A{alg}_HL{hint}.pickle",
        ]
    elif experiment == "multiple":
        learner_list = [
        f"learner_history_{task}_std_contest_eval_A{alg}_HLNone.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLrecent_g_past.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLrecent_g_future.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLrecent_g.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLrecent_g_double.pickle",
        f"learner_history_{task}_std_contest_eval_A{alg}_HLrecent_g_triple.pickle",
        ]

    experiment_list = []
    for f in learner_list:
        fh = open(os.path.join(experiments_home, f), "rb")
        targets, regret_periods, alias, history = pickle.load(fh)
        models = copy.copy(model_alias)
        models["online_learner"] = alias["online_learner"]
        if experiment == "hinting":
            # Rename the learner to it's hint type
            if ":w
            " in f:
                models["online_learner"] = "prev_g"
            elif "mean_g" in f:
                models["online_learner"] = "mean_g"
            elif "recent_g in f:
                models["online_learner"] = "recent_g"
            elif "None" in f:
                models["online_learner"] = "none"
            elif "dormplus" in f:
                models["online_learner"] = "learned"
        elif experiment == "rep":
            if "_R" in f:
                models["online_learner"] = "Replicated DORM+"
        elif experiment == "multiple":
            if "None" in f:
                models["online_learner"] = "0"
            elif "past" in f:
                models["online_learner"] = "1"
            elif "future" in f:
                models["online_learner"] = "D"
            elif "double" in f:
                models["online_learner"] = "2D+1"
            elif "triple" in f:
                models["online_learner"] = "3D+1"
            elif "recent_g" in f:
                models["online_learner"] = "D+1"

        experiment_list.append((targets, regret_periods, models, history))

    filename = f"{experiment}_{hint}_{alg}_{task}"
    if final_year:
        subset_time = (-26, None)
    else:
        subset_time = None
    all_task_tables[task] = visualize_multiple(experiment_list, style_algs, subset_time=subset_time, filename=filename)

tablename = f"{experiment}_{hint}_{alg}"
df = display_table(all_task_tables, experiment_list[0][3].models, experiment_list[0][2], task_dict, filename=tablename)
print(df)

if experiment == "multiple":
    hinters = ["0", "D", "1", "D+1", "2D+1", "3D+1"]
    fig, axs = plt.subplots(1,4, figsize=(25,5), sharey=False)

    lines_precip = []
    for i, task in enumerate(df.index):
        axs[i].scatter(["none", "past", "future", "past+\nfuture", "2D+1", "3D+1"], df.loc[task, hinters], s=100)
        axs[i].set_title(task)
        if i == 0:
            axs[i].set_ylabel(f"Average RMSE", fontsize=25)  

    # Save figure to file 
    if not os.path.exists('figs'):
        os.mkdir('figs')
    plt.savefig(f"figs/multiple_{hint}_{alg}.pdf", bbox_inches='tight')