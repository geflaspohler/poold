
import pickle
import matplotlib.pyplot as plt
import copy

from vis_params import model_alias, alg_naming, style_algs

# PoolD imports
from poold import create
from poold.utils import loss_regret
from poold.utils import visualize, visualize_multiple

#TODO: remove this import
import pdb

fl = [
    "learner_history_contest_tmp2m_34w_std_contest_eval_Adormplus_HLprev_g.pickle",
    "learner_history_contest_tmp2m_34w_std_contest_eval_Adorm_HLprev_g.pickle",
    "learner_history_contest_tmp2m_34w_std_contest_eval_Aadahedged_HLprev_g.pickle",
    "learner_history_contest_tmp2m_34w_std_contest_eval_Adub_HLprev_g.pickle",
    "learner_history_contest_tmp2m_34w_std_contest_eval_Adormplus_HLavg_prev_g.pickle",
    "learner_history_contest_tmp2m_34w_std_contest_eval_Adormplus_HLprev_g_future.pickle"
]

experiment_list = []
for f in fl:
    fh = open(f, "rb")
    targets, regret_periods, model_alias, history = pickle.load(fh)
    experiment_list.append((targets, regret_periods, model_alias, history))
visualize_multiple(experiment_list, style_algs)
# visualize_multiple(experiment_list, style_algs, subset_time=(-26, None))
plt.show()
