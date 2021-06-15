""" Visualizations for online learning.

For example:
    import poold
    poold.visualize(learner.history)
    plt.show()

"""
# System imports
import numpy as np
import pandas as pd
import copy
import os

# Plotting imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
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

def visualize_multiple(experiment_list, style_algs, subset_time=None, filename="temp"):
    num_plots = len(experiment_list)
    fig_r, ax_r = plt.subplots(1, 1, figsize=(10, 4), sharex=False)
    fig_p, ax_p = plt.subplots(1, 1, figsize=(5, 4), sharex=False)
    fig_w, ax_w = plt.subplots(1, num_plots, figsize=(4*num_plots, 4), sharex=False)

    df_losses = None
    for i, (targets, regret_periods, model_alias, history) in enumerate(experiment_list):
        df = visualize(history, regret_periods, targets, model_alias, style_algs, ax=[ax_w[i], ax_r, ax_p], subset_time=subset_time, legend=(i==0))
        if df_losses is None:
            df_losses = copy.copy(df)
        else:
            df_losses = df_losses.merge(df)
    mean_losses = df_losses.mean(axis=0)

    # Save dataframe in latex table format
    fname = f"./eval/losses_{filename}.tex"
    mean_losses.to_latex(fname, float_format="%.3f", longtable=False)

    fig_w.tight_layout()
    fig_w.subplots_adjust(top=0.95, wspace=0, hspace=0)
    fig_r.tight_layout()
    fig_r.subplots_adjust(top=0.95)
    fig_p.tight_layout()
    fig_p.subplots_adjust(top=0.95)

    if not os.path.exists('figs'):
        os.mkdir('figs')
    filename_w = f"./figs/weights_{filename}.pdf"
    filename_r = f"./figs/regret_{filename}.pdf"
    filename_p = f"./figs/params_{filename}.pdf"
    fig_w.savefig(filename_w, bbox_inches='tight')
    fig_r.savefig(filename_r, bbox_inches='tight')
    fig_p.savefig(filename_p, bbox_inches='tight')

    return mean_losses

def visualize(history, regret_periods=None, time_labels=None, model_labels={}, 
    style_algs={}, ax=[None, None, None], params=["lam"], subset_time=None, legend=True):
    """ Visualize online learning losses, weights, and parameters.

    Args:
        history (History): online learning History object
        regret_periods (list[tuple]): list of tuples specifying the start (inclusive) and end
            points (not inclusive) of regret periods
        time_labels (list): list of labels for the time periods
        model_labels (dict): dictionary of model labels
        style_algs (dict): dictionary of model styles
        ax (list[ax]): list of axis objects for plotting the weights, regret, and parameter
            plots respectively.
        params (list[str]): list of parameters to plot
        subset_time (tuple): plot values from times[subset_time[0]:subset_time[1]]
        legend (bool): if True, plot legend.
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
        loss_learner = loss_obj['fun'](w=play_learner)
        params_learner = history.get_params(time)

        loss_all = loss_obj.get('exp', {})
        loss_all['online_learner'] = loss_learner

        # Assign loss and weight dataframe
        df_losses.iloc[t] = loss_all
        df_weights.iloc[t] = dict(zip(history.models, play_learner))
        if df_params is not None:
            df_params.iloc[t] = params_learner

    plot_weights(df_weights, regret_periods, model_labels, style_algs, ax[0], legend, subset_time)
    if not df_losses[history.models].isna().all(axis=None):
        plot_regret(df_losses, regret_periods, model_labels, style_algs, history.models, ax[1], only_learner=True, subset_time=subset_time)
    if df_params is not None:
        plot_params(df_params, regret_periods, model_labels["online_learner"], style_algs, ax[2], subset_time=subset_time)

    return df_losses.rename({"online_learner": model_labels["online_learner"]}, axis=1)

def plot_weights(df, regret_periods, model_labels, style_algs, ax=None, legend=True, subset_time=None):
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
        ax.legend(handles, labels,  prop={'size': 15}, ncol=1, loc='best')
    else:
        ax.set_yticks([])
    ticks = ax.get_xticks()
    # ax.set_xticks(ticks[0:-1])

    if subset_time is not None:
        # Date based formatting
        if isinstance(df.index, pd.DatetimeIndex):
            datefmt = mdates.DateFormatter('%b')
            ax.xaxis.set_major_formatter(datefmt)

    plot_time_seperators(regret_periods, df.index, ax)

def plot_regret(df, regret_periods, model_labels, style_algs, input_models, ax=None, only_learner=False, subset_time=None):
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

    if subset_time is not None:
        # Date based formatting
        if isinstance(df.index, pd.DatetimeIndex):
            datefmt = mdates.DateFormatter('%b')
            ax.xaxis.set_major_formatter(datefmt)

def plot_params(df, regret_periods, model_alias, style_algs, ax=None, subset_time=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharex=False)

    for m in df.columns:
        style = style_algs.get(model_alias, {})
        model_alias = model_alias.replace("_", "\_")
        final_val = df.iloc[-1]["lam"]
        label = f"{model_alias}" + "\t ($\lambda_{T}$ = " + f"{final_val: .3f})"
        ax.plot(df.index, df[m], label=label, **style)

    ticks = ax.get_xticks()
    # ticks[-1] = ticks[-2]
    ax.set_xticks(ticks)

    ax.set_title("Regularization $\lambda_t$")

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,  prop={'size': 15}, ncol=1, loc='best')


    plot_time_seperators(regret_periods, df.index, ax)

    if subset_time is not None:
        # Date based formatting
        if isinstance(df.index, pd.DatetimeIndex):
            datefmt = mdates.DateFormatter('%b')
            ax.xaxis.set_major_formatter(datefmt)

def plot_time_seperators(regret_periods, index, ax):
    ''' Local utiliy function for plotting vertical time seperators '''
    for start, end in regret_periods:
        start_time = index[start]
        if end == len(index):
            end -= 1
        elif end > len(index):
            raise ValueError("Bad time seperator", start, end)
        end_time = index[end]
        ax.axvline(x=start_time, c='k', linestyle='-.', linewidth=1.0)