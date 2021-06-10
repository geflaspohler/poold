
import pickle
import matplotlib.pyplot as plt
import copy
import os
from itertools import product
import pandas as pd

# S2S imports
from src.s2s_vis_params import model_alias, alg_naming, style_algs, task_dict

# PoolD imports
from poold import create
from poold.utils import loss_regret
from poold.utils import visualize, visualize_multiple

task_dict = {
    "contest_precip_34w": "Precip. 3-4w",
    "contest_precip_56w": "Precip. 5-6w",    
    "contest_tmp2m_34w": "Temp. 3-4w",
    "contest_tmp2m_56w": "Temp. 5-6w"
}

def display_table(data_dict, model_list, model_alias={}, task_dict={}, filename="temp"):
    """Displays dataframe after sorting """
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

    fname = f"./eval/all_task_losses_{filename}.tex"
    df.to_latex(fname, float_format="%.3f", longtable=False, column_format=align)
    return df

experiments_home = "./experiments"
experiment = "rep"
hint = "prev_g"
all_task_tables = {}
for gt_id, horizon in product(
        ["contest_tmp2m","contest_precip"], ["34w", "56w"]):

    task = f"{gt_id}_{horizon}"

    if experiment == "zoo":
        learner_list = [
        f"learner_history_{task}_std_contest_eval_Adormplus_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_Adorm_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_Aadahedged_HL{hint}.pickle",
        # f"learner_history_{task}_std_contest_eval_Adub_HL{hint}.pickle",
        ]
    elif experiment == "hinting":
        learner_list = [
        f"learner_history_{task}_std_contest_eval_Adormplus_HLavg_prev_g.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLprev_g.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLmean_g.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLNone.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLdormplus.pickle",
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
        f"learner_history_{task}_std_contest_eval_Adormplus_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_R{rep}_Adormplus_HL{hint}.pickle",
        ]
    elif experiment == "multiple":
        learner_list = [
        f"learner_history_{task}_std_contest_eval_Adormplus_HLNone.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLprev_g_past.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLprev_g_future.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLprev_g.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLprev_g_double.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HLprev_g_triple.pickle",
        ]

    experiment_list = []
    for f in learner_list:
        fh = open(os.path.join(experiments_home, f), "rb")
        targets, regret_periods, alias, history = pickle.load(fh)
        models = copy.copy(model_alias)
        models["online_learner"] = alias["online_learner"]
        if experiment == "hinting":
            # Rename the learner to it's hint type
            if "avg_prev_g" in f:
                models["online_learner"] = "prev_g"
            elif "mean_g" in f:
                models["online_learner"] = "mean_g"
            elif "prev_g" in f:
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
            elif "prev_g" in f:
                models["online_learner"] = "D+1"

        experiment_list.append((targets, regret_periods, models, history))

    if experiment == "zoo" and hint != "prev_g":
        filename = f"{experiment}_{hint}_{task}"
    else:
        filename = f"{experiment}_{task}"

    all_task_tables[task] = visualize_multiple(experiment_list, style_algs, filename=filename)
    # all_task_tables[task] = visualize_multiple(experiment_list, style_algs, subset_time=(-26, None), filename=filename)

if experiment == "zoo" and hint != "prev_g":
   tablename = f"{experiment}_{hint}"
else:
    tablename = f"{experiment}"
df = display_table(all_task_tables, experiment_list[0][3].models, experiment_list[0][2], task_dict, filename=tablename)
print(df)

if experiment == "multiple":
    hinters = ["0", "1", "D", "D+1", "2D+1", "3D+1"]
    fig, axs = plt.subplots(1,4, figsize=(25,5), sharey=False)

    lines_precip = []
    for i, task in enumerate(df.index):
        df.loc[task, hinters].plot(ax=axs[i], label=task, color='b', linewidth=2)
        axs[i].set_xticks(range(6))
        axs[i].set_xticklabels(["0", "1", "D", "D+1", "2D+1", "3D+1"])
        axs[i].set_title(task)
        axs[i].set_xlabel("recent\_g Multiplier $c$", fontsize=15)
        if i == 0:
            axs[i].set_ylabel(f"Average RMSE", fontsize=15)  


    # Save figure to file 
    plt.savefig("figs/multiple.pdf",bbox_inches='tight')
