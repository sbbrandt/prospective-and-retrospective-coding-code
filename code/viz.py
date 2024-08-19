import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.optimize
import scipy.interpolate
import scipy.signal

import pickle

import logging
from brian2 import ms, mV, exp, exprel, asarray

# from analysis import get_posterior_limits
from utils import (
    zscore,
    augment_parameters,
    eqs_to_dict,
    get_label,
    get_evaluated_parameters,
)
import config


class sine_func:
    """Sinus function for fitting

    Parameters
    ----------
    fixed_f : bool, optional
        Whether to fix frequency, by default False
    f : float, optional
        Frequency used when 'fixed_f=True', by default None
    """

    def __init__(self, fixed_f=False, f=None):
        self.fixed_f = fixed_f
        if fixed_f:
            assert f is not None
            self.f = f

    def __call__(self, x, *params):
        if self.fixed_f:
            dist, amp, phi = params
            f = self.f
        else:
            dist, amp, f, phi = params

        return dist + amp * np.sin(2 * np.pi * f * x + phi)


synonyms = {
    "s": "size",
    "size": "fontsize",
    "c": "color",
    "lw": "linewidth",
    "ls": "linestyle",
}


def pop_plot_kw(parameter, default, kw):
    """Utility function to get get parameter (and synonyms) from plot kwargs

    Parameters
    ----------
    parameter : str
        Short name of parameter, e.g. 'lw' (linewidth)
    default : any
        Value the parameter should default to if not specified in the 'kw'
    kw : dict
        kwargs to get parameter from

    Returns
    -------
    any
        The parameter value in the kwargs dict or the specified default value.
    """
    value = kw.pop(parameter, default)
    if parameter in synonyms.keys():
        value = kw.pop(synonyms[parameter], value)
    return value


def line_between_axs(fig, ax0, x, xlim, subplot=(1, 1, 1), plot_kw={}):
    """Utility function to plot line between to axes

    Parameters
    ----------
    fig : Figure
        Figure that contains the two axes between which the line should be
        drawn.
    ax0 : Axis
        Lower axis from which the line should start to the upper axis.
    x : float
        x-axis value of the line.
    xlim : list of float
        Limits of x-axis of the two axes.
    subplot : int, optional
        Subplot id that corresponds to the subplot containing the two axes
        between which the line should be drawn, by default 111
    plot_kw : dict, optional
        kwargs that are passed to 'plt.plot', by default {}
    """
    if type(subplot) is tuple:
        ax2 = fig.add_subplot(*subplot, frameon=False)
    elif type(subplot) is int:
        ax2 = fig.add_subplot(subplot, frameon=False)
    elif type(subplot) is plt.SubplotSpec:
        ax2 = fig.add_subplot(subplot, frameon=False)
    else:
        raise ValueError
    ax2.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
    # ax2.get_shared_x_axes().join(ax2, ax0)
    ax2._shared_axes["x"].join(ax2, ax0)
    ax2.set_ylim([0, 1])
    ax2.set_xlim(xlim)

    plot_kw = plot_kw.copy()

    ls = pop_plot_kw("ls", "--", plot_kw)
    lw = pop_plot_kw("lw", "1.5", plot_kw)
    c = pop_plot_kw("c", "k", plot_kw)

    ax2.plot([x] * 2, [0, 1], lw=lw, ls=ls, c=c, **plot_kw)

    return ax2


def clean_sin_params(params):
    """Brings sinus parameters in correct form

    - removes negative sign: -sin(w*t)=sin(w*t-pi)
    - sets phase to be within [0, 2*pi]: sin(w*t + 2*pi)=sin(w*t)

    Parameters
    ----------
    params : list of floats
        List containing the sinus parameters in the form '[a0, a1, ..., phi]'.

    Returns
    -------
    list
        List containing the sinus parameters with positive 'a0' and
        '0<=phi< 2*pi'.
    """
    a0, a1, phi = params[[0, 1, -1]]

    if a1 < 0:
        a1 *= -1
        phi -= np.pi

    phi = np.sign(phi) * (abs(phi) % (2 * np.pi))

    params[[0, 1, -1]] = a0, a1, phi

    return tuple(params)


def psth(t, N, hist_kw={}):
    """Function to compute the psth from spike times

    Peristimulus time histogram (psth) shows the average spike response of a
    neuron to a stimulus, taken over multiple trials. In silico, a single trial
    with multiple similar neurons can be simulated instead of repeating single
    neuron experiments. A stimulus can also be the start of the recording.

    Parameters
    ----------
    t : array like
        The spike times
    N : int
        The number of neurons/trials
    hist_kw : dict, optional
        kwargs that are passed to 'numpy.histogram', by default {'bins': 100}

    Returns
    -------
    tuple
        Time and value of the average firing rate
    """
    hist_kw.setdefault("bins", 100)
    h, t_h = np.histogram(t, **hist_kw)
    dth = np.diff(t_h)[0]
    h = h / dth / N
    t_h = t_h[:-1] + dth / 2
    return (t_h, h)


def fit_sine(s, dist=None, amp=None, f=10, phi=0.0, initialize=True, fixed_f=False):
    """fit s to a sine function

    Args:
        s (np.array): signal
        dist (float, optional): Initialization parameter: offset. Defaults to None.
        amp (float, optional): Initialization parameter: amplitude. Defaults to None.
        f (int, optional): Initialization parameter: frequency. Defaults to 10.
        phi (float, optional): Initialization parameter: phase. Defaults to 0..
        initialize (bool, optional): Initialize with signal properties if initialization parameter is not given. Defaults to True.
        fixed_f (bool, optional): Fix the frequency. Defaults to False.

    Returns:
        list: parameters
    """
    if initialize:
        if dist is None:
            dist = s[1].mean()
        if amp is None:
            amp = (s[1].max() - s[1].min()) / 2
    else:
        if dist is None:
            dist = 0.0
        if amp is None:
            amp = 1.0

    if fixed_f:
        # raise NotImplementedError
        # func = sin_func_fixed_f
        p_0 = (dist, amp, phi)
    else:
        # func = sin_func
        p_0 = (dist, amp, f, phi)

    func = sine_func(fixed_f=fixed_f, f=f)
    params, _ = scipy.optimize.curve_fit(func, s[0], s[1], p0=p_0)
    params = clean_sin_params(params)
    return params


def phase_shift(
    s1,
    s2,
    dist_0=None,
    amp_0=None,
    f_0=10.0,
    phi_0=0.0,
    dist_1=None,
    amp_1=None,
    f_1=None,
    phi_1=None,
    initialize=True,
    return_params=False,
    verbose=False,
    fixed_f=False,
    logger=logging.getLogger(__name__),
):
    """Computes the phase shift between two signals

    Fits a sinusoidal for each of the two signals and either returns the phase
    shift based on the phase of the sinusoidal parameters or returns the
    parameters of the fit.

    Parameters
    ----------
    s1 : tuple
        Contains time and value of the first signal
    s2 : tuple
        Contains time and value of the second signal
    dist_0 : float, optional
        Starting point for the fitting of the first signal: offset, by default 0.
    amp_0 : float, optional
        Starting point for the fitting of the first signal: amplitude, by default 1.
    f_0 : float, optional
        Starting point for the fitting of the first signal: frequency, by default 10.
    phi_0 : float, optional
        Starting point for the fitting of the first signal: phase, by default 0.
    dist_1 : float, optional
        Starting point for the fitting of the second signal: offset. Defaults to
        'dist_0' if 'None', by default None
    amp_1 : float, optional
        Starting point for the fitting of the second signal: amplitude. Defaults
        to 'amp_0' if 'None', by default None
    f_1 : float, optional
        Starting point for the fitting of the second signal: frequency. Defaults
        to 'f_0' if 'None', by default None
    phi_1 : float, optional
        Starting point for the fitting of the second signal: phase. Defaults to
        'phi_0' if 'None', by default None
    initialize: bool, optional
        Initialize with signal properties if initialization parameter is not given. Defaults to True.
    return_params : bool, optional
        If 'True', fitted parameters for the two signals and not the phase are
        returned, by default False
    verbose : bool, optional
        Whether to print stats, by default False
    fixed_f : bool, optional
        If 'True', frequency will not be fitted and fixed to starting point
        'f_0' and/or 'f_1', by default False
    logger : logging.logger, optiona
        logging object. Defaults to logging.getLogger(__name__).

    Returns
    -------
    float or tuple of dicts
        Either the phase between the two signals or their fitted parameters.
    """
    assert type(s1) == tuple
    assert type(s2) == tuple
    assert len(s1) == 2
    assert len(s2) == 2
    # x1, y1 = s1
    # x2, y2 = s2

    func = sine_func(fixed_f=fixed_f, f=f_0)
    if f_1 is None:
        f_1 = f_0
    if phi_1 is None:
        phi_1 = phi_0

    params_1 = fit_sine(
        s1,
        dist=dist_0,
        amp=amp_0,
        f=f_0,
        phi=phi_0,
        initialize=initialize,
        fixed_f=fixed_f,
    )

    if fixed_f:
        f1 = f_0
    else:
        f1 = params_1[-2]

    dt_1 = (2 - np.sign(params_1[1])) / 4 / f1 - params_1[-1] / 2 / np.pi / f1
    if verbose:
        if fixed_f:
            logger.info(
                "s1:\n\ta_0={:.2f}\n\ta_1={:.2f}\n\t\phi={:.2f}".format(*params_1)
            )
            print("s1:\n\ta_0={:.2f}\n\ta_1={:.2f}\n\t\phi={:.2f}".format(*params_1))
        else:
            logger.info(
                "s1:\n\ta_0={:.2f}\n\ta_1={:.2f}\n\tf={:.2f}\n\t\phi={:.2f}".format(
                    *params_1
                )
            )
            print(
                "s1:\n\ta_0={:.2f}\n\ta_1={:.2f}\n\tf={:.2f}\n\t\phi={:.2f}".format(
                    *params_1
                )
            )

    params_2 = fit_sine(
        s2,
        dist=dist_1,
        amp=amp_1,
        f=f_1,
        phi=phi_1,
        initialize=initialize,
        fixed_f=fixed_f,
    )

    if fixed_f:
        f2 = f_0
    else:
        f2 = params_2[-2]

    dt_2 = (2 - np.sign(params_2[1])) / 4 / f2 - params_2[-1] / 2 / np.pi / f2

    if fixed_f:
        shift = -params_2[-1] / 2 / np.pi / f_0 + params_1[-1] / 2 / np.pi / f_0

        if shift > 1 / f_0 / 2:
            shift -= 1 / f_0
        if shift < -1 / f_0 / 2:
            shift += 1 / f_0

    else:
        while (dt_2 - dt_1) > 1 / f2 / 2:
            dt_2 -= 1 / f2
        while (dt_1 - dt_2) > 1 / f1 / 2:
            dt_1 -= 1 / f1
        shift = dt_2 - dt_1

    if verbose:
        if fixed_f:
            logger.info(
                "s2:\n\ta_0={:.2f}\n\ta_1={:.2f}\n\t\phi={:.2f}".format(*params_2)
            )
        else:
            logger.info(
                "s2:\n\ta_0={:.2f}\n\ta_1={:.2f}\n\tf={:.2f}\n\t\phi={:.2f}".format(
                    *params_2
                )
            )
        logger.info("phase shift = {:.4f}".format(shift))

    if return_params:
        return params_1, params_2
    else:
        return shift


def plot_phase_shift(
    s1,
    s2,
    fig=None,
    ax1=None,
    ax2=None,
    label1=None,
    label2=None,
    i_max=1,
    params_1=None,
    params_2=None,
    verbose=False,
    fixed_f=False,
    subplot=(1, 1, 1),
    arrow_from="bottom",
    arrow_ax="top",
    arrow_y=None,
    arrow_y_offset=None,
    arrow_min_dx=None,
    arrow_text_kw={},
    arrow_text_x_offset=0,
    arrow_text_y_offset=None,
    arrow_text_x=None,
    arrow_text=r"${\rm dt}=<shift>\,{\rm <unit>}$",
    arrow_patch_kw={},
    plot_kw_1={},
    plot_kw_2={},
    type_1="line",
    type_2="line",
    type_hat_1="line",
    type_hat_2="line",
    fontsize=16,
    plot_kw_hat={},
    plot_kw_hat_2=None,
    plot_kw_line_between={},
    plot_kw_line_between_2=None,
    phase_shift_kw={},
):
    plot_kw_1 = plot_kw_1.copy()
    plot_kw_2 = plot_kw_2.copy()
    plot_kw_hat = plot_kw_hat.copy()
    c_hat = plot_kw_hat.pop("color", "r")
    plot_kw_hat.setdefault("c", c_hat)
    if plot_kw_hat_2 is None:
        plot_kw_hat_2 = plot_kw_hat.copy()
    else:
        plot_kw_hat_2 = plot_kw_hat_2.copy()
    plot_kw_line_between = plot_kw_line_between.copy()
    if plot_kw_line_between_2 is None:
        plot_kw_line_between_2 = plot_kw_line_between.copy()
    else:
        plot_kw_line_between_2 = plot_kw_line_between_2.copy()
    phase_shift_kw = phase_shift_kw.copy()
    arrow_text_kw = arrow_text_kw.copy()
    arrow_patch_kw = arrow_patch_kw.copy()
    arrow_patch_kw.setdefault("arrowstyle", "fancy, head_length=5, head_width=5")
    assert arrow_ax in ["top", "bottom"]
    assert arrow_from in ["top", "bottom"]

    if fig is None:
        assert ax1 is None
        assert ax2 is None
        fig, (ax1, ax2) = plt.subplots(2, figsize=(7.5, 10), sharex=True)

    f = phase_shift_kw.get("f_0", 10)
    func = sine_func(fixed_f=fixed_f, f=f)

    if (params_1 is None) or (params_2 is None):
        assert params_1 is None
        assert params_2 is None
        params_1, params_2 = phase_shift(
            s1,
            s2,
            return_params=True,
            verbose=verbose,
            fixed_f=fixed_f,
            **phase_shift_kw
        )

    assert arrow_from in ["top", "bottom"]

    xlim = [min(s1[0][0], s2[0][0]), max(s1[0][-1], s2[0][-1])]

    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    for i, (ax, s, l, p, kw, t, th, kw_hat) in enumerate(
        zip(
            [ax1, ax2],
            [s1, s2],
            [label1, label2],
            [params_1, params_2],
            [plot_kw_1, plot_kw_2],
            [type_1, type_2],
            [type_hat_1, type_hat_2],
            [plot_kw_hat, plot_kw_hat_2],
        )
    ):
        x, y = s
        if t == "line":
            ax.plot(x, y, label=l, **kw)
        elif t == "bar":
            ax.bar(x, y, width=np.diff(x)[0], bottom=xlim[0], label=l, **kw)
        elif t is not None:
            raise NotImplementedError
        x = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 200)

        y_hat = func(x, *p)
        if th == "line":
            ax.plot(x, y_hat, label="sine fit", **kw_hat)
        elif th is not None:
            raise NotImplementedError
        ax.set_xlim(xlim)

    xmaxs = []
    for i in range(2):
        p = [params_1, params_2][i]
        phi = np.sign(p[-1]) * (abs(p[-1]) % (2 * np.pi))

        if not fixed_f:
            f = p[-2]

        xmax = (2 - np.sign(p[1])) / 4 / f - phi / 2 / np.pi / f
        xmax += (i_max) / f
        if i == 1:
            while xmax > xmaxs[0] + 1 / f / 2:
                xmax -= 1 / f
            while xmax <= xmaxs[0] - 1 / f / 2:
                xmax += 1 / f
        kw_ = [plot_kw_line_between, plot_kw_line_between_2][i]
        ax_between = line_between_axs(
            fig, ax1, xmax, xlim, subplot=subplot, plot_kw=kw_
        )
        x0, y0, w, h = ax2.get_position().bounds
        x1, y1, _, _ = ax1.get_position().bounds

        h = y1 + h - y0
        ax_between.set_position([x0, y0, w, h])
        xmaxs.append(xmax)

    if arrow_ax == "top":
        y_min = s1[1].min()
        y_max = s1[1].max()
        ax_idx_arrow = 0
    elif arrow_ax == "bottom":
        y_min = s2[1].min()
        y_max = s2[1].max()
        ax_idx_arrow = 1

    ax_arrow = [ax1, ax2][ax_idx_arrow]

    xlim = ax_arrow.get_xlim()
    ylim = ax_arrow.get_ylim()

    if arrow_from == "top":
        i_from = 0
        i_to = 1
    elif arrow_from == "bottom":
        i_from = 1
        i_to = 0

    dx = xmaxs[i_to] - xmaxs[i_from]
    sign = np.sign(dx)
    if arrow_min_dx is None:
        min_dx = (xlim[1] - xlim[0]) / 10
    else:
        min_dx = arrow_min_dx
    if abs(dx) < min_dx:
        x1 = xmaxs[i_from]
        x2 = xmaxs[i_from] + sign * min_dx
    else:
        x1 = xmaxs[i_from]
        x2 = xmaxs[i_to]

    # y = ylim[1] + arrow_y_offset  # - .1*(ylim[1] - ylim[0])
    if arrow_y is None:
        if arrow_y_offset is None:
            arrow_y_offset = (ylim[1] - ylim[0]) / 10
        if arrow_text_y_offset is None:
            arrow_text_y_offset = (ylim[1] - ylim[0]) / 10
        y = y_max  # - .1*(ylim[1] - ylim[0])
    else:
        if arrow_y_offset is None:
            arrow_y_offset = 0
        if arrow_text_y_offset is None:
            arrow_text_y_offset = 0
        y = arrow_y
    y += arrow_y_offset

    if arrow_text_x is None:
        # position of arrow
        arrow_text_x = min(x1, x2) + 0.1 * np.sign(dx) * min_dx
    elif arrow_text_x == "right":
        arrow_text_x = max(xmaxs)
    elif arrow_text_x == "left":
        arrow_text_x = min(xmaxs)
    else:
        assert type(1.0 * arrow_text_x) is float

    arrow = mpl.patches.FancyArrowPatch(
        (x1, y),
        (x2, y),
        facecolor="k",
        zorder=5,
        shrinkA=0,
        shrinkB=0,
        **arrow_patch_kw
    )
    ax_arrow.add_patch(arrow)

    # y = y_max + .05*(y_max - y_min) + arrow_y_offset + arrow_text_y_offset
    y += arrow_text_y_offset
    # y = ylim[1] + .05*(ylim[1] - ylim[0]) + arrow_y_offset + arrow_text_y_offset

    size = arrow_text_kw.pop("size", fontsize)
    size = arrow_text_kw.pop("fontsize", size)
    import re

    pattern = r"<(.*?)>"
    variables = re.findall(pattern, arrow_text)
    for v in variables:
        assert v in ["sign*shift", "shift", "unit"]  # '-sign*shift',

    ns = {
        "shift": np.round(abs(xmaxs[1] - xmaxs[0]) * 1e3, 1),
        "sign*shift": sign * np.round(abs(xmaxs[1] - xmaxs[0]) * 1e3, 1),
        "unit": "ms",
    }

    for v in variables:
        arrow_text = arrow_text.replace("<{}>".format(v), str(ns[v]))

    ax_arrow.text(
        arrow_text_x + arrow_text_x_offset, y, arrow_text, size=size, **arrow_text_kw
    )
    ax_arrow.set_ylim(
        [
            ax_arrow.get_ylim()[0],
            ax_arrow.get_ylim()[1] * 1.1 + arrow_y_offset + arrow_text_y_offset,
        ]
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax1.tick_params(labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)

    ax1.set_xticklabels([])
    return params_1, params_2


def plot_variable(
    x,
    y,
    zscore_variable=False,
    scale=1.0,
    offset=0.0,
    every=1,
    x_offset=0.0,
    label=None,
    xlabel=None,
    ylabel=None,
    legend=False,
    ax=None,
    fontsize=16,
    figsize=(15, 7.5),
    plot_kw={},
):
    """Utility function to plot a variable

    Parameters
    ----------
    x : array
        Time or x-value of the variable.
    y : array
        Value of the variable.
    zscore_variable : bool, optional
        Whether to zscore (zero mean, unit variance) the variable, by default False
    scale : str, optional
        Factor by which to scale the variable. Will be applied after zscore,
        by default 1.
    offset : float, optional
        Offset that will be added to the variable. Will be applied after zscore,
        by default 0.
    every : int, optional
        plot 'every'th value, by default 1.
    label : str or None, optional
        Label of the variable, by default None
    xlabel : str or None, optional
        Label of the x-axis, by default None
    ylabel : str or None, optional
        Label of the y-axis, by default None
    legend : bool, optional
        Whether a legend should be included, by default False
    ax : axes, optional
        If provided, variable will be added to axes via 'ax.plot', by default None
    fontsize : int, optional
        Fontsize for all texts, by default 16
    figsize : tuple, optional
        Size of figure, by default (15, 7.5)
    plot_kw : dict, optional
        kwargs that will be passed to plot, by default {}

    Returns:
        Axes: axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    if zscore_variable:
        y = zscore(y)
    y *= scale
    y += offset

    x += x_offset

    ax.plot(x[::every], y[::every], label=label, **plot_kw)
    if legend:
        ax.legend(fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, size=fontsize)
    if xlabel is None:
        xlabel = get_label("t")
    if xlabel:
        ax.set_xlabel(xlabel, size=fontsize)
    ax.tick_params(labelsize=fontsize)

    return ax


def plot_spikes(t, i, ax=None, plot_kw={}):
    """Raster plot of spikes

    Args:
        t (array): spike times
        i (array): spike indices
        ax (axes, optional): Axis to plot on. Defaults to None.
        plot_kw (dict, optional): 'plt.scatter' kwargs. Defaults to {}.

    Returns:
        tuple: plt.Figure, plt.Axes
    """
    plot_kw = plot_kw.copy()
    plot_kw.set_default("c", "k")

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 7.5))
    ax.scatter(t, i, **plot_kw)
    return fig, ax


def plot_posterior_check(
    x_o,
    phases,
    x_range=(-0.02, 0.02),
    sign_flip=True,
    ax=None,
    fontsize=None,
    save_name=None,
    dpi=300,
    hist_kw={},
    plot_kw={},
):
    hist_kw = hist_kw.copy()
    plot_kw = plot_kw.copy()

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(7.5, 5))

    if type(phases) is list:
        phases = np.array(phases)
    if sign_flip:
        phases = np.array(phases)
        phases = phases.copy()
        phases *= -1

    hist_kw.setdefault("bins", 100)
    hist_kw.setdefault("histtype", "step")

    ax.hist(phases, range=x_range, **hist_kw)

    c = pop_plot_kw("c", "r", plot_kw)
    ylim = ax.get_ylim()
    ax.plot([x_o] * 2, ylim, c=c, **plot_kw)
    ax.set_ylim(ylim)

    ax.set_xlim(x_range)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.tick_params(labelsize=fontsize)

    ax.set_xticks(
        [
            i
            for i in np.arange(
                x_range[0], x_range[1] + 1e-3, (x_range[1] - x_range[0]) / 4
            )
        ]
    )
    ax.set_xlabel(r"${\rm shift\,[s]}$", size=fontsize)

    if save_name is not None:
        plt.savefig(save_name, dpi=dpi)


# posterior analysis
def get_posterior_limits(experiment_id):
    experiment_params = config.experiment["sbi"][experiment_id]
    n_theta = len(
        experiment_params["augmented_model_params"] + experiment_params["augmented_eqs"]
    )

    if type(experiment_params["low"]) is float:
        assert type(experiment_params["high"]) is float
        low = [experiment_params["low"]] * n_theta
        high = [experiment_params["high"]] * n_theta
    elif experiment_params["low"] is None:
        low = [-1] * n_theta
        high = [1] * n_theta
    else:
        assert len(experiment_params["low"]) == len(experiment_params["high"])
        low = experiment_params["low"]
        high = experiment_params["high"]
    return low, high


def plot_conditional_posterior(
    experiment_id,
    posterior_path=None,
    posterior=None,
    fig=None,
    colors=None,
    shifts=[-0.01, 0.01],
    resolution=100,
    n_samples=1000,
    limits=None,
    figsize=(10, 10),
    fontsize=16,
    seed=None,
):
    if seed is not None:
        import torch

        torch.manual_seed(seed)
    exp_params = config.experiment["sbi"][experiment_id]
    if posterior is None:
        if posterior_path is None:
            posterior_path = os.path.join(
                exp_params["save_dir"], "sbi", experiment_id, "posterior.p"
            )
        with open(posterior_path, "rb") as tmp:
            posterior = pickle.load(tmp)
    sign_flip = True

    conditions = []
    for shift in shifts:
        import torch

        x_o = shift * torch.ones((1))
        if sign_flip:
            x_o *= -1
        posterior.set_default_x(x_o)
        tmp_condition = posterior.sample((n_samples,), show_progress_bars=True)
        log_prob = posterior.log_prob(tmp_condition)
        conditions.append(tmp_condition[log_prob.argmax()])

    thetas = exp_params["augmented_model_params"] + exp_params["augmented_eqs"]
    n_theta = len(thetas)

    if fig is None:
        gridspec_kw = {"left": 0.01, "right": 0.99, "bottom": 0.075, "top": 0.99}
        fig, axs = plt.subplots(
            n_theta, n_theta, figsize=figsize, gridspec_kw=gridspec_kw
        )
    else:
        axs = fig.subplots(n_theta, n_theta)

    indices = np.tril_indices(n_theta, -1)
    for i, (row, col) in enumerate(zip(indices[0], indices[1])):
        ax = axs[row, col]
        ax.set_axis_off()

    indices = np.triu_indices(n_theta, 1)
    for i, (row, col) in enumerate(zip(indices[0], indices[1])):
        ax = axs[row, col]
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    indices = np.triu_indices(n_theta, 0)
    for i, (row, col) in enumerate(zip(indices[0], indices[1])):
        ax = axs[row, col]
        # delay
        for i_s, shift in enumerate(shifts):
            x_o = shift
            if sign_flip:
                x_o *= -1
            _plot_single_conditional_posterior(
                row,
                col,
                posterior,
                x_o,
                i_color=i_s,
                experiment_id=experiment_id,
                ax=ax,
                colors=colors,
                condition=conditions[i_s],
                resolution=resolution,
                n_samples=n_samples,
                limits=limits,
                seed=seed,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(labelsize=fontsize)
        if row == col:
            label = (
                r"$\lambda_{" + config.variable_dict[thetas[row]]["tex"][1:-1] + "}$"
            )
            ax.set_yticks([])
            ax.set_xlabel(label, size=fontsize + 2)
            ax.spines["left"].set_visible(False)

    return axs


def _plot_single_conditional_posterior(
    row,
    col,
    posterior,
    x_o,
    i_color,
    experiment_id,
    colors=None,
    ax=None,
    im=None,
    resolution=50,
    n_samples=1000,
    limits=None,
    seed=None,
    condition=None,
):
    from sbi.analysis import eval_conditional_density
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    def get_im(
        row,
        col,
        posterior,
        limits,
        x_o=0 * torch.ones(1),
        resolution=50,
        n_samples=1000,
        condition=None,
    ):
        eps_margins = (limits[row, 1] - limits[row, 0]) / 1e5
        posterior.set_default_x(x_o)

        if condition is None:
            tmp_condition = posterior.sample((n_samples,), show_progress_bars=True)
            log_prob = posterior.log_prob(tmp_condition)
            condition = tmp_condition[log_prob.argmax()]

        return eval_conditional_density(
            posterior,
            condition,
            limits,
            row,
            col,
            resolution=resolution,
            eps_margins1=eps_margins,
            eps_margins2=eps_margins,
        )

    if colors is None:
        tab = mpl.colormaps["tab10"]
        colors = [tab(0), tab(1)]
        # c_ = [tab(0), (1, 1, 1), tab(1)]
    c_ = [colors[0], (1, 1, 1), colors[1]]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap_name", c_, N=256)

    if limits is None:
        low, high = get_posterior_limits(experiment_id)
        limits = np.array(list(zip(low, high)))

    x_o = x_o * torch.ones((1,))

    if ax is None:
        assert im is None
        _, ax = plt.subplots(1, figsize=(10, 7.5))

    if col == row:
        p_vector = get_im(
            row,
            row,
            posterior,
            limits,
            x_o=x_o,
            resolution=resolution,
            condition=condition,
        )
        x = np.linspace(
            limits[row, 0],
            limits[row, 1],
            resolution,
        )
        if im is not None:
            assert type(im) is mpl.lines.Line2D  # 1d
            im.set_data(x, p_vector)
        else:
            im = ax.plot(
                np.linspace(
                    limits[row, 0],
                    limits[row, 1],
                    resolution,
                ),
                p_vector,
                c=colors[i_color],
            )[0]
    else:
        p_image = np.array(
            get_im(
                row,
                col,
                posterior,
                limits,
                x_o=x_o,
                resolution=resolution,
                n_samples=n_samples,
                condition=condition,
            )
        )
        x = np.ones_like(p_image)
        ax.imshow(
            (-1) ** (i_color + 1) * x,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            alpha=p_image.T,
            origin="lower",
            extent=(
                limits[col, 0],
                limits[col, 1],
                limits[row, 0],
                limits[row, 1],
            ),
            aspect="auto",
        )
    return im


def conditional_posterior_movie(
    posterior,
    experiment_id,
    save_name,
    fontsize=20,
    phase_range=10,
    resolution=50,
    interval=500,
    figsize=(30, 30),
):
    import torch
    from matplotlib import animation

    experiment_params = config.experiment["sbi"][experiment_id]
    # init
    labels = (
        experiment_params["augmented_model_params"] + experiment_params["augmented_eqs"]
    )
    n_theta = len(labels)
    fig, axs = plt.subplots(n_theta, n_theta, figsize=figsize)

    # init images
    x_o = 0 * torch.ones(1)
    ims = []
    indices = np.triu_indices(n_theta, 0)
    for i, (row, col) in enumerate(zip(indices[0], indices[1])):
        ax = axs[row, col]
        im = _plot_single_conditional_posterior(
            row=row,
            col=col,
            posterior=posterior,
            x_o=x_o,
            experiment_id=experiment_id,
            ax=ax,
            im=None,
            resolution=resolution,
        )
        ims.append(im)

    # make lower axes blank
    indices = np.tril_indices(n_theta, -1)
    for i, (row, col) in enumerate(zip(indices[0], indices[1])):
        ax = axs[row, col]
        ax.set_axis_off()

    for i, label in enumerate(labels):
        if label in config.variable_dict:
            if "tex" in config.variable_dict[label]:
                label = config.variable_dict[label]["tex"]
        ax = axs[i, i]
        ax.set_xlabel("$" + label + "$", size=fontsize)

    def animate(phase, fig, axes, ims):
        fig.suptitle("phase = {}".format(phase), size=fontsize)
        phase = -phase  # adapt for Walter
        x_o = phase / 1000 * torch.ones(1)
        posterior.set_default_x(x_o)
        indices = np.triu_indices(n_theta, 0)
        for i, (row, col) in enumerate(zip(indices[0], indices[1])):
            ax = axes[row, col]
            ims[i] = _plot_single_conditional_posterior(
                row=row,
                col=col,
                posterior=posterior,
                x_o=x_o,
                experiment_id=experiment_id,
                ax=ax,
                im=ims[i],
                resolution=resolution,
            )
        return ims

    fargs = (fig, axs, ims)
    anim = animation.FuncAnimation(
        fig,
        animate,
        fargs=fargs,
        frames=range(-phase_range, phase_range),
        interval=interval,
        blit=True,
    )
    anim.save(save_name)


def phase_histogram(phases, ax=None, save_str=None, hist_kw={}):
    if ax is None:
        fig, ax = plt.subplots(1)
    bins = hist_kw.pop("bins", 100)
    ax.hist(phases, bins=bins)
    ax.tick_params(labelsize=16)
    ax.set_xlabel("shift [ms]", size=16)

    if save_str is not None:
        plt.savefig(save_str, dpi=150)


def plot_final_posterior(
    sbi_id,
    posterior_path=None,
    posterior=None,
    shifts=[0.01, -0.01],
    n_samples=1000,
    limits=None,
    v_range=[-100, 20],
    thetas=None,
    sign_flip=True,
    parameters=None,
    figsize=(6.3, 2),
    plot_std=False,
    get_im_kw={},
    gridspec_kw={},
    fill_between_kw={},
):
    import torch
    from sbi.analysis import eval_conditional_density

    experiment_params = config.experiment["sbi"][sbi_id].copy()
    model_type = experiment_params["model_type"]
    model_name = experiment_params["model_name"]
    model_params = config.model_dynamics[model_type][model_name]["parameters"].copy()
    eqs = config.model_dynamics[model_type][model_name]["eqs"]

    log10 = experiment_params.pop("log10", False)

    gridspec_kw = gridspec_kw.copy()
    gridspec_kw.setdefault("left", 0.25)

    fill_between_kw = fill_between_kw.copy()
    fill_between_kw.setdefault("alpha", 0.2)

    get_im_kw = get_im_kw.copy()
    get_im_kw.setdefault("resolution", 100)
    get_im_kw.setdefault("n_samples", 1000)

    # if fig is not None:
    #     assert axs is not None

    if parameters is None:
        parameters = []
        if "augmented_model_params" in experiment_params:
            parameters += experiment_params["augmented_model_params"]
        if "augmented_eqs" in experiment_params:
            parameters += experiment_params["augmented_eqs"]
        if "augmented_simulation_params" in experiment_params:
            parameters += list(experiment_params["augmented_simulation_params"].keys())

    if thetas is not None:
        assert len(thetas) == len(shifts)
        assert len(thetas[0][0]) == len(parameters)
        n_samples = len(thetas[0])

    n_theta = len(parameters)

    if posterior is None:
        if posterior_path is None:
            posterior_path = os.path.join(
                experiment_params["save_dir"], "sbi", sbi_id, "posterior.p"
            )
        if not os.path.isfile(posterior_path):
            raise FileNotFoundError

        with open(posterior_path, "rb") as tmp:
            posterior = pickle.load(tmp)

    def get_im(
        row,
        col,
        posterior,
        limits,
        x_o=0 * torch.ones(1),
        n_samples=1000,
        resolution=50,
    ):
        eps_margins = (limits[row, 1] - limits[row, 0]) / 1e5

        posterior.set_default_x(x_o)

        tmp_condition = posterior.sample((n_samples,), show_progress_bars=False)
        log_prob = posterior.log_prob(tmp_condition)
        condition = tmp_condition[log_prob.argmax()]

        return eval_conditional_density(
            posterior,
            condition,
            limits,
            row,
            col,
            resolution=resolution,
            eps_margins1=eps_margins,
            eps_margins2=eps_margins,
        )

    eqs = eqs.replace(" ", "")
    equations = eqs.split("\n")
    variables = ["alpha_m", "beta_m", "alpha_h", "beta_h"]
    var_eqs = {}
    for eq in equations:
        eq = eq.split("=")
        if eq[0] in variables:
            var_eqs[eq[0]] = eq[1].split(":")[0]

    v = np.arange(v_range[0], v_range[1], 0.1)

    g = {}
    g["original"] = {
        "tau_m": np.zeros(len(v)),
        "m_inf": np.zeros(len(v)),
        "tau_h": np.zeros(len(v)),
        "h_inf": np.zeros(len(v)),
    }

    def f(v):
        return 1 / (
            eval(var_eqs["alpha_m"], globals(), locals()) + eval(var_eqs["beta_m"])
        )

    g["original"]["tau_m"] = f(v * mV)
    f = lambda v: eval(var_eqs["alpha_m"]) / (
        eval(var_eqs["alpha_m"]) + eval(var_eqs["beta_m"])
    )
    g["original"]["m_inf"] = f(v * mV)
    f = lambda v: 1 / (eval(var_eqs["alpha_h"]) + eval(var_eqs["beta_h"]))
    g["original"]["tau_h"] = f(v * mV)
    f = lambda v: eval(var_eqs["alpha_h"]) / (
        eval(var_eqs["alpha_h"]) + eval(var_eqs["beta_h"])
    )
    g["original"]["h_inf"] = f(v * mV)
    f = lambda v: eval(var_eqs["beta_m"])
    g["original"]["beta_m"] = f(v * mV)
    f = lambda v: eval(var_eqs["alpha_h"])
    g["original"]["alpha_h"] = f(v * mV)

    for shift in shifts:
        g[shift] = {
            "tau_m": np.zeros((n_samples, len(v))),
            "m_inf": np.zeros((n_samples, len(v))),
            "tau_h": np.zeros((n_samples, len(v))),
            "h_inf": np.zeros((n_samples, len(v))),
            "alpha_m": np.zeros((n_samples, len(v))),
            "beta_h": np.zeros((n_samples, len(v))),
            "g_l": np.zeros(n_samples),
        }

        if thetas is None:
            thts = posterior.sample((n_samples,), x=shift)
        else:
            thts = thetas[0]

        if log10:
            thts = 10**thts

        # get rid of torch tensor
        thts = [[float(ti) for ti in t] for t in thts]

        if "g_l" in parameters:
            i_gl = parameters.index("g_l")
        else:
            i_gl = None
            gl = 1
        if "alpha_m" in parameters:
            i_am = parameters.index("alpha_m")
        else:
            i_am = None
            am = "1"
        if "beta_h" in parameters:
            i_bh = parameters.index("beta_h")
        else:
            i_bh = None
            bh = "1"

        for i_t, tht in enumerate(thts):
            if i_gl is not None:
                gl = tht[i_gl]
            if i_am is not None:
                am = str(tht[i_am])
            if i_bh is not None:
                bh = str(tht[i_bh])
            f = lambda v: 1 / (
                eval("*".join([am, var_eqs["alpha_m"]])) + eval(var_eqs["beta_m"])
            )
            g[shift]["tau_m"][i_t] = f(v * mV)
            f = lambda v: eval("*".join([am, var_eqs["alpha_m"]])) / (
                eval("*".join([am, var_eqs["alpha_m"]])) + eval(var_eqs["beta_m"])
            )
            g[shift]["m_inf"][i_t] = f(v * mV)
            f = lambda v: 1 / (
                eval(var_eqs["alpha_h"]) + eval("*".join([bh, var_eqs["beta_h"]]))
            )
            g[shift]["tau_h"][i_t] = f(v * mV)
            f = lambda v: eval(var_eqs["alpha_h"]) / (
                eval(var_eqs["alpha_h"]) + eval("*".join([bh, var_eqs["beta_h"]]))
            )
            g[shift]["h_inf"][i_t] = f(v * mV)
            f = lambda v: eval(var_eqs["beta_m"])
            g[shift]["alpha_m"][i_t] = f(v * mV)
            f = lambda v: eval(var_eqs["alpha_h"])
            g[shift]["beta_h"][i_t] = f(v * mV)
            g[shift]["g_l"][i_t] = gl

    # if fig is None:
    fig = plt.figure(constrained_layout=False, figsize=(figsize))
    subfigs = fig.subfigures(
        2, 4, wspace=0.1, width_ratios=[3, 3, 3, 2], height_ratios=[7, 1]
    )
    axs = []
    plot_kw = {}
    alpha = plot_kw.pop("alpha", 0.7)
    lw = plot_kw.pop("lw", 1)
    ls_original = ":"
    labels = ["advance", "delay"]
    xticks, xticklabels = [-100, -80, -60, -40, -20, 0, 20], [
        "",
        -80,
        "",
        -40,
        "",
        0,
        "",
    ]
    fontsize = 12

    tex_dict = {
        "m_inf": r"$m_\infty$",
        "tau_m": r"$\tau_m$",
        "h_inf": r"$h_\infty$",
        "tau_h": r"$\tau_h$",
        "alpha_m": r"$\alpha_m$",
        "beta_h": r"$\beta_h$",
        "g_l": r"$g_l$",
        "lam_alpha_m": r"$\lambda_{\alpha_m}$",
        "lam_beta_h": r"$\lambda_{\beta_h}$",
        "lam_g_l": r"$\lambda_{g_l}$",
    }
    lines = []

    for i_g, gating in enumerate(["m", "h"]):
        axs_ = subfigs[0, i_g + 1].subplots(2, 1, sharex=True, gridspec_kw=gridspec_kw)
        for i_v, var in enumerate(["tau_" + gating, gating + "_inf"]):
            ax = axs_[i_v]
            for i_s, shift in enumerate(shifts):
                y = g[shift][var].mean(0)
                if labels is None:
                    if sign_flip:
                        label = -shift
                    else:
                        label = shift
                else:
                    label = labels[i_s]
                if i_v == 0:
                    label = None
                ax.plot(v, y, label=label, alpha=alpha, lw=lw)
                if plot_std:
                    std = g[shift][var].std(0)
                    ax.fill_between(v, y - std, y + std, **fill_between_kw)
            if i_v == 0:
                label = None
            else:
                label = "original"
            ax.plot(
                v,
                g["original"][var],
                label=label,
                alpha=alpha,
                c="k",
                ls=ls_original,
                lw=lw,
            )
            ax.set_xticks(xticks, xticklabels)
            if var.split("_")[0] == "tau":
                yticks = ax.get_yticks()
                if yticks[-1] * 1000 > 2:
                    ax.set_yticklabels(["{}".format(int(yt * 1000)) for yt in yticks])
                else:
                    yticks = [round(yt, 4) for yt in yticks[1:-1]]
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(["{:.1f}".format(yt * 1000) for yt in yticks])
            elif var.split("_")[1] == "inf":
                ax.set_yticks([0, 0.5, 1])

            if i_v == 1:
                ax.set_xlabel(r"$V_{\rm m}\,{\rm [mV]}$", size=fontsize)

            ax.set_ylabel(tex_dict[var], size=fontsize)
        axs.append(axs_)

    ##########
    axs_ = subfigs[0, 0].subplots(n_theta, n_theta)
    tab = mpl.colormaps["tab10"]
    colors = [tab(0), (1, 1, 1), tab(1)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap_name", colors, N=256)

    if limits is None:
        low = experiment_params["low"]
        high = experiment_params["high"]
        if type(low) is list:
            assert type(high) is list
            assert len(low) == len(high)
        else:
            low = [low] * n_theta
            high = [high] * n_theta
        limits = np.array([low, high]).T

    indices = np.tril_indices(n_theta, -1)
    for i, (row, col) in enumerate(zip(indices[0], indices[1])):
        ax = axs_[row, col]
        ax.set_axis_off()

    indices = np.triu_indices(n_theta, 1)
    for i, (row, col) in enumerate(zip(indices[0], indices[1])):
        ax = axs_[row, col]
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    indices = np.triu_indices(n_theta, 1)
    for i, (row, col) in enumerate(zip(indices[0], indices[1])):
        ax = axs_[row, col]
        # delay
        for i_s, shift in enumerate(shifts):
            x_o = shift * torch.ones(1)
            # if sign_flip:
            #     x_o *= -1
            p_image = np.array(
                get_im(row, col, posterior, limits, x_o=x_o, **get_im_kw)
            )
            x = np.ones_like(p_image)
            ax.imshow(
                (-1) ** (i_s + 1) * x,
                cmap=cmap,
                vmin=-1,
                vmax=1,
                alpha=p_image.T,
                origin="lower",
                extent=(
                    limits[col, 0],
                    limits[col, 1],
                    limits[row, 0],
                    limits[row, 1],
                ),
                aspect="auto",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i in range(n_theta):
        col = row = i
        ax = axs_[row, col]
        ones = np.linspace(
            limits[row, 0],
            limits[row, 1],
            get_im_kw["resolution"],
        )
        for i_s, shift in enumerate(shifts):
            x_o = shift * torch.ones(1)
            # if sign_flip:
            #     x_o *= -1
            p_image = np.array(
                get_im(row, col, posterior, limits, x_o=x_o, **get_im_kw)
            )
            ax.plot(ones, p_image, c=tab(i_s), alpha=alpha, lw=lw)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=fontsize)
        ax.set_yticks([])
        label = r"$\lambda_{" + config.variable_dict[parameters[i]]["tex"][1:-1] + "}$"
        # ax.set_xlabel(tex_dict['lam_' + parameters[i]], fontsize=fontsize+2)
        ax.set_xlabel(label, fontsize=fontsize + 2)

        # if log10:
        #     xticks = ax.get_xticks()
        #     print(xticks)
        #     ax.set_xticklabels('$10^{}$'.format(int(xt)) for xt in xticks)
    axs.append(axs_)
    subfigs[0, 2].legend(bbox_to_anchor=(1, 0.5), loc="center left", fontsize=fontsize)
    return fig, subfigs, axs


def get_gating_variable(
    v,
    variable,
    model_type="HH_model",
    model_name=None,
    eqs=None,
    model_parameters=None,
    augmented_eqs={},
    augmented_model_params={},
):
    if model_name is not None:
        assert model_parameters is None
        assert eqs is None
        assert model_name in config.model_dynamics[model_type].keys()
        model_parameters, eqs, _ = augment_parameters(
            model_type,
            model_name,
            augmented_model_params=augmented_model_params,
            augmented_eqs=augmented_eqs,
        )
    else:
        assert model_parameters is not None
        assert eqs is not None

    mp = get_evaluated_parameters(model_parameters=model_parameters.copy(), unit=True)
    var_eqs = eqs_to_dict(eqs)

    if variable in var_eqs.keys():
        # var = lambda v: eval(var_eqs[variable], globals(), dict(mp, **{'v': v*mV}))
        var = eval(var_eqs[variable], globals(), dict(mp, **{"v": v * mV}))
    elif variable.split("_")[1] == "inf":
        v0 = variable.split("_")[0]
        assert "alpha_{}".format(v0) in var_eqs
        assert "beta_{}".format(v0) in var_eqs

        a = lambda v: eval(
            var_eqs["alpha_{}".format(v0)], globals(), dict(mp, **{"v": v * mV})
        )
        b = lambda v: eval(
            var_eqs["beta_{}".format(v0)], globals(), dict(mp, **{"v": v * mV})
        )
        fa = augmented_model_params.get("f_{}_inf_a{}".format(v0, v0), 1)
        fb = augmented_model_params.get("f_{}_inf_b{}".format(v0, v0), 1)
        var = fa * a(v) / (fa * a(v) + fb * b(v))
    elif variable.split("_")[0] == "tau":
        v0 = variable.split("_")[1]
        assert "alpha_{}".format(v0) in var_eqs
        assert "beta_{}".format(v0) in var_eqs

        a = lambda v: eval(
            var_eqs["alpha_{}".format(v0)], globals(), dict(mp, **{"v": v * mV})
        )
        b = lambda v: eval(
            var_eqs["beta_{}".format(v0)], globals(), dict(mp, **{"v": v * mV})
        )
        f_tau = mp.get("f_tau_{}".format(v0), 1)
        fa = augmented_model_params.get("f_tau_{}_a{}".format(v0, v0), 1)
        fb = augmented_model_params.get("f_tau_{}_b{}".format(v0, v0), 1)
        var = 1e3 * f_tau / (fa * a(v) + fb * b(v))
    else:
        raise NotImplementedError
    return asarray(var)


def plot_gating_variables(
    variables=["m_inf", "tau_m", "h_inf", "tau_h"],
    v_range=[-100, 20],
    dv=1,
    model_name=None,
    model_type="HH_model",
    eqs=None,
    model_parameters=None,
    augmented_model_params={},
    augmented_eqs={},
    c=None,
    fontsize=16,
    ax=None,
    axt=None,
    lines=None,
    legend=True,
    plot_kw={},
):
    if type(variables) is not list:
        variables = [variables]
    if c is None:
        c = [plt.colormaps["tab10"](i) for i in range(len(variables))]

    # voltage range
    v_lin = np.arange(v_range[0], v_range[1], dv)

    # get gating variables
    get_gating_kw = {
        "v": v_lin,
        "model_type": model_type,
        "model_name": model_name,
        "eqs": eqs,
        "model_parameters": model_parameters,
        "augmented_eqs": augmented_eqs,
        "augmented_model_params": augmented_model_params,
    }
    inf = tau = g_na = False
    inf_labels = tau_labels = []
    gating = {}
    for i_var, var in enumerate(variables):
        if var.split("_")[1] == "inf":
            inf = True
            inf_labels.append(get_label(var))
            gating[var] = get_gating_variable(variable=var, **get_gating_kw)
        elif var.split("_")[0] == "tau":
            tau = True
            tau_labels.append(get_label(var))
            gating[var] = get_gating_variable(variable=var, **get_gating_kw)
        elif var == "g_na_inf":
            g_na = True
            gating[var] = (
                1e9
                * get_evaluated_parameters(
                    "g_na", model_type=model_type, model_name=model_name
                )
                * get_gating_variable(variable="m_inf", **get_gating_kw) ** 3
                * get_gating_variable(variable="h_inf", **get_gating_kw)
            )
        elif var == "m_inf^3":
            inf = True
            inf_labels.append(r"$m_\infty^3$")
            gating[var] = get_gating_variable(variable="m_inf", **get_gating_kw) ** 3

    # only one on twinx; assume x_inf is plotted
    assert not (tau and g_na)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(15, 10))
    if ((inf and tau) or g_na) and (axt is None):
        axt = ax.twinx()
        axt.tick_params(labelsize=fontsize)

    # allow to concatenate earlier lines
    if lines is None:
        lines = []
    for i_var, var in enumerate(variables):
        if var in config.variable_dict.keys():
            label = config.variable_dict[var]["tex"]
        else:
            label = var
        if inf and ((var.split("_")[0] == "tau") or (var == "g_na_inf")):
            l = axt.plot(v_lin, gating[var], label=label, c=c[i_var], **plot_kw)[0]
        else:
            l = ax.plot(v_lin, gating[var], label=label, c=c[i_var], **plot_kw)[0]
        lines.append(l)

    # legend
    labs = [l.get_label() for l in lines]
    if legend:
        ax.legend(lines, labs, fontsize=fontsize)

    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(get_label("v"), size=fontsize)

    # labels
    if inf and tau:
        if len(tau_labels) > 1:
            axt.set_ylabel(r"$\tau_x\,{\rm [ms]}$", size=fontsize)
        else:
            axt.set_ylabel(tau_labels[0], size=fontsize)
    elif inf and g_na:
        axt.set_ylabel(r"$g_{\rm Na}^\infty$", size=fontsize)
    elif tau:
        if len(tau_labels) > 1:
            ax.set_ylabel(r"$\tau_x\,{\rm [ms]}$", size=fontsize)
        else:
            ax.set_ylabel(tau_labels[0], size=fontsize)
    if inf:
        if len(inf_labels) > 1:
            ax.set_ylabel(r"$x_\infty$", size=fontsize)
        else:
            ax.set_ylabel(inf_labels[0], size=fontsize)
    return lines
