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
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

import seaborn as sns
sns.set(font_scale=1.6)
sns.set_style("white")

import pdb

def visualize_multiple(experiment_list, style_algs, subset_time=None):
    num_plots = len(experiment_list)
    fig_w, ax_w = plt.subplots(1, num_plots, figsize=(5*num_plots, 4), sharex=False)
    fig_r, ax_r = plt.subplots(1, 1, figsize=(5, 4), sharex=False)
    fig_p, ax_p = plt.subplots(1, 1, figsize=(5, 4), sharex=False)

    for i, (targets, regret_periods, model_alias, history) in enumerate(experiment_list):
        visualize(history, regret_periods, targets, model_alias, style_algs, ax=[ax_w[i], ax_r, ax_p], subset_time=subset_time, legend=(i==0))

def visualize(history, regret_periods=None, time_labels=None, model_labels={}, 
    style_algs={}, ax=[None, None, None], params=["lam"], subset_time=None, legend=True):
    """ Visualize

    Args:
        history (History): online learning History object
    """
    times = history.get_times()
    if time_labels is None:
        time_labels = range(len(times))

    if subset_time is not None:
        times = times[subset_time[0]:subset_time[1]]
        time_labels = time_labels[subset_time[0]:subset_time[1]]
    
    if regret_periods is None:
        regret_periods = [(0, len(times))]

    if subset_time is not None:
        subset_regret_periods = []
        for s, e in regret_periods:
            if s in times and e in times:
                subset_regret_periods.append((times.index(s), times.index(e)))
            elif s in times:
                subset_regret_periods.append((times.index(s), len(times)))
            elif e in times:
                subset_regret_periods.append((0, times.index(e)))
        regret_periods = subset_regret_periods

    assert(len(time_labels) == len(times))

    df_losses = pd.DataFrame(columns=history.models+["online_learner"], index=time_labels)
    df_weights = pd.DataFrame(columns=history.models, index=time_labels)
    
    param = history.get_params(0)
    if len(param) > 0:
        param_labels = list(set(params).intersection(set(param.keys())))
        df_params = pd.DataFrame(columns=param_labels, index=time_labels)
    else:
        df_params = None

    for t, time in enumerate(times):
        loss_obj, loss_learner, loss_grad  = history.get_loss(time)
        play_learner = history.get_play(time, return_past=False)
        params_learner = history.get_params(time)

        loss_all = loss_obj.get('exp', {})
        loss_all['online_learner'] = loss_learner

        # Assign loss and weight dataframe
        df_losses.iloc[t] = loss_all
        df_weights.iloc[t] = dict(zip(history.models, play_learner))
        if df_params is not None:
            df_params.iloc[t] = params_learner

    plot_weights(df_weights, regret_periods, model_labels, style_algs, ax[0], legend)
    plot_regret(df_losses, regret_periods, model_labels, style_algs, history.models, ax[1], only_learner=True)
    if df_params is not None:
        plot_params(df_params, regret_periods, model_labels["online_learner"], style_algs, ax[2])

def plot_weights(df, regret_periods, model_labels, style_algs, ax=None, legend=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=False)

    for m in df.columns:
        alias = model_labels.get(m, m)
        style = style_algs.get(alias, {})
        alias = alias.replace("_", "\_")
        ax.plot(df.index, df[m], label=alias, **style)

    ax.set_title(model_labels.get("online_learner", "Online Learner").replace("_", "\_") + " weights $\mathbf{w}_t$")
    ax.set_ylim([0.0, 1.0])

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels,  prop={'size': 12}, ncol=2, loc='best')

    # Date based formatting
    # if isinstance(df.index, pd.DatetimeIndex):
    #     datefmt = mdates.DateFormatter('%m-%d')
    #     ax.xaxis.set_major_formatter(datefmt)

    plot_time_seperators(regret_periods, df.index, ax)

def plot_regret(df, regret_periods, model_labels, style_algs, input_models, ax=None, only_learner=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=False)

    # Find best model per regret period
    if only_learner:
        model_list = ["online_learner"]
    else:
        model_list = list(df.columns)

    mean_loss = df.mean(axis=0)
    zeros = pd.Series(0, index=df.index)
    df_regret = pd.DataFrame(columns=model_list, index=df.index)
    for s, e in regret_periods:
        cumsum = df.iloc[s:e].cumsum(axis=0)
        best_model = pd.to_numeric(cumsum[input_models].iloc[-1]).idxmin()
        relative_cumsum = cumsum - cumsum[[best_model]].values
        df_regret.loc[relative_cumsum.index, :] = relative_cumsum[model_list].values

    for m in df_regret.columns:
        alias = model_labels.get(m, m)
        style = style_algs.get(alias, {})
        alias = alias.replace("_", "\_")

        label = f"{alias} \t (RMSE: {mean_loss[m]: .3f})" 
        ax.plot(df_regret.index, df_regret[m], label=label, **style)

    ax.plot(zeros.index, zeros, c='k', linestyle="-")
    ax.set_title("Cumulative regret (RMSE loss)")

    # Sort both labels and handles by labels
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,  prop={'size': 15}, ncol=2, loc='best')

    plot_time_seperators(regret_periods, df.index, ax)

    # Date based formatting
    # if isinstance(df.index, pd.DatetimeIndex):
    #     datefmt = mdates.DateFormatter('%m-%d')
    #     ax.xaxis.set_major_formatter(datefmt)

def plot_params(df, regret_periods, model_alias, style_algs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=False)

    for m in df.columns:
        style = style_algs.get(model_alias, {})
        model_alias = model_alias.replace("_", "\_")
        final_val = df.iloc[-1]["lam"]
        label = f"{model_alias}" + "\t ($\lambda_{T}$ = " + f"{final_val: .3f})"
        ax.plot(df.index, df[m], label=label, **style)

    ax.set_title("Regularization $\lambda_t$")

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,  prop={'size': 15}, ncol=1, loc='best')

    plot_time_seperators(regret_periods, df.index, ax)

    # Date based formatting
    # if isinstance(df.index, pd.DatetimeIndex):
    #     datefmt = mdates.DateFormatter('%b')
    #     ax.xaxis.set_major_formatter(datefmt)

def plot_time_seperators(regret_periods, index, ax):
    ''' Local utiliy function for plotting vertical time seperators '''
    for start, end in regret_periods:
        start_time = index[start]
        if end > len(index):
            pdb.set_trace()
        if end == len(index):
            end -= 1
        end_time = index[end]
        ax.axvline(x=start_time, c='k', linestyle='-.', linewidth=1.0)
        # ax.axvline(x=end_time, c='k', linestyle='-.', linewidth=1.0)