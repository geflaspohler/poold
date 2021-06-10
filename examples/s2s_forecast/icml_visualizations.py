
import pickle
import matplotlib.pyplot as plt
import copy
import os
from itertools import product
import pandas as pd

from vis_params import model_alias, alg_naming, style_algs

# PoolD imports
from poold import create
from poold.utils import loss_regret
from poold.utils import visualize, visualize_multiple

#TODO: remove this import
import pdb

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
experiment = "regularization"
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
        hint = "avg_prev_g"
        # hint = "prev_g"
        learner_list = [
        f"learner_history_{task}_std_contest_eval_Adub_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_Aadahedged_HL{hint}.pickle",
        f"learner_history_{task}_std_contest_eval_Adormplus_HL{hint}.pickle",
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
                models["online_learner"] = "past_g"
            elif "mean_g" in f:
                models["online_learner"] = "mean_g"
            elif "prev_g" in f:
                models["online_learner"] = "recent_g"
            elif "None" in f:
                models["online_learner"] = "none"
            elif "dormplus" in f:
                models["online_learner"] = "learned"

        experiment_list.append((targets, regret_periods, models, history))

    filename = f"{experiment}_{task}"
    # all_task_tables[task] = visualize_multiple(experiment_list, style_algs, filename=filename)
    all_task_tables[task] = visualize_multiple(experiment_list, style_algs, subset_time=(-26, None), filename=filename)
# visualize_multiple(experiment_list, style_algs, subset_time=(-26, None), filename=filename)

df = display_table(all_task_tables, experiment_list[0][3].models, experiment_list[0][2], task_dict, filename=experiment)
print(df)

# plt.show()
