""" Visualizations for online learning.

For example:
    import poold

"""
# System imports
import numpy as np
import pandas as pd

# Plotting imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
font = {'family' : 'lmodern',
        'weight' : 'normal',
        'size'   : 50}
text = {'usetex' : True}
matplotlib.rc('font',**font)
matplotlib.rc('text',**text)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import seaborn as sns
sns.set(font_scale=1.6)
sns.set_style("white")

import pdb

def visualize(history, time_labels=None, model_labels={}):
    """ Visualize

    Args:
        history (History): online learning History object
    """
    times = history.get_times()
    if time_labels is None:
        time_labels = range(len(times))
    assert(len(time_labels) == len(times))

    df_losses = pd.DataFrame(columns=history.models+["online_learner"], index=time_labels)
    df_weights = pd.DataFrame(columns=history.models, index=time_labels)

    for t, time in enumerate(times):
        loss_obj, loss_learner, loss_grad  = history.get_loss(time)
        play_learner = history.get_play(time, return_past=False)

        loss_all = loss_obj.get('exp', {})
        loss_all['online_learner'] = loss_learner

        # Assign loss and weight dataframe 
        try:
            df_losses.iloc[t] = loss_all
            df_weights.iloc[t] = dict(zip(history.models, play_learner))
        except:
            pdb.set_trace()
            df_losses.iloc[t] = loss_all
            df_weights.iloc[t] = dict(zip(history.models, play_learner))

    plot_weights(df_weights, model_labels)

def plot_weights(df_w, model_labels):
    num_plots = 1
    fig, ax = plt.subplots(1, num_plots, figsize=(5*num_plots, 4), sharex=False)
    for m in df_w.columns:
        alias = model_labels.get(m, m)
        alias = alias.replace("_", "\_")
        ax.plot(df_w.index, df_w[m], label=alias)

    ax.set_title(model_labels.get("online_learner", "Online Learner").replace("_", "\_") + " weights $\mathbf{w}_t$")
    ax.set_ylim([0.0, 1.0])

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,  prop={'size': 15}, ncol=2, loc='best')

    # Date based formatting
    # if isinstance(df_w.index, pd.DatetimeIndex):
    #     datefmt = mdates.DateFormatter('%m-%d')
    #     ax.xaxis.set_major_formatter(datefmt)

    plt.show()
