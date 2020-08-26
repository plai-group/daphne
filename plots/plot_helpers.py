import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def plot_ensemble(xs, ys, label, color, lstyle="-", nstd=1):
    q = erf(nstd / np.sqrt(2))

    #plt.plot(xs, np.median(ys, axis=0), label=label, c=color, ls=lstyle)
    plt.plot(xs, np.mean(ys, axis=0), label=label, c=color, ls=lstyle)
    print("Quantiles:", (1 + q) / 2, (1 - q) / 2)
    plt.fill_between(
        xs, np.quantile(ys, (1 + q) / 2, axis=0), np.quantile(ys, (1 - q) / 2, axis=0),
        color=color, alpha=.2)


# TODO needed for below, maybe factor out
aistats_textwidth = 487.8225
aistats_columnwidth = 234.8775


# Taken from https://jwalton.info/Embed-Publication-Matplotlib-Latex/ :
def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def setup_matplotlib():
    # Using seaborn's style
    #plt.style.use('seaborn')

    nice_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
    }

    mpl.rcParams.update(nice_fonts)

