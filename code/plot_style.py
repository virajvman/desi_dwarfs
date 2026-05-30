"""Shared matplotlib style and layout helpers for catalog-paper figures."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

MARGIN_LABEL  = 0.9   # edge with axis label + tick labels
MARGIN_TICKS  = 0.45  # edge with only tick labels (no axis label)
MARGIN_PAD    = 0.2   # edge with nothing
MARGIN_SHARED = 0.1   # gap between panels sharing an axis
MARGIN_SPLIT  = 0.7   # gap between panels with independent axes
MARGIN_CBAR   = 1.1   # edge with a colorbar


def apply_paper_style(usetex=False):
    """Matplotlib rcParams for catalog-paper figures (explore_fastspec style)."""
    params = {
        'text.usetex':                 usetex,
        'font.family':                 'serif',
        'font.serif':                  'cmr10',
        'font.size':                   11,
        'mathtext.fontset':            'cm',
        'axes.unicode_minus':          False,
        'axes.formatter.use_mathtext': True,
        'axes.labelsize':              12,
        'axes.titlesize':              12,
        'xtick.labelsize':             10,
        'ytick.labelsize':             10,
        'legend.fontsize':             11,
        'axes.linewidth':              1.0,
        'axes.xmargin':                0.05,
        'xtick.direction':             'in',
        'xtick.top':                   True,
        'xtick.minor.visible':         True,
        'xtick.major.top':             True,
        'xtick.minor.top':             True,
        'xtick.major.size':            3.5,
        'xtick.minor.size':            2.0,
        'xtick.major.width':           1.0,
        'xtick.minor.width':           0.8,
        'xtick.major.pad':             4,
        'xtick.minor.pad':             4,
        'ytick.direction':             'in',
        'ytick.right':                 True,
        'ytick.minor.visible':         True,
        'ytick.major.right':           True,
        'ytick.minor.right':           True,
        'ytick.major.size':            3.5,
        'ytick.minor.size':            2.0,
        'ytick.major.width':           1.0,
        'ytick.minor.width':           0.8,
        'ytick.major.pad':             4,
        'ytick.minor.pad':             4,
        'legend.frameon':              False,
        'legend.edgecolor':            'k',
        'legend.framealpha':           0,
        'figure.facecolor':            'w',
        'image.aspect':                'auto',
        'image.cmap':                  'magma',
        'savefig.format':              'pdf',
        'savefig.bbox':                'standard',
        'savefig.dpi':                 300,
        'savefig.transparent':         False,
        'errorbar.capsize':            0,
        'lines.markersize':            5,
        'hist.bins':                   'auto',
    }

    if usetex:
        params['text.latex.preamble'] = r'\usepackage{xcolor}'

    matplotlib.rcParams.update(params)

def reshape_axes(flat_axes, nrow, ncol):
    """Convert flat list from make_subplots (bottom-left, row-major, bottom-up)
    into a 2D array indexed as [row_from_top, col], matching plt.subplots."""
    arr = np.empty((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            arr[nrow - 1 - i, j] = flat_axes[i * ncol + j]
    return arr


def make_subplots(ncol=3, nrow=1, row_spacing=1.1, col_spacing=1.1, plot_size=2,
                  direction="horizontal", return_fig=False):
    if np.isscalar(row_spacing):
        row_spacing = [row_spacing] * (nrow + 1)
    if np.isscalar(col_spacing):
        col_spacing = [col_spacing] * (ncol + 1)
    assert len(row_spacing) == nrow + 1, f"need {nrow+1} row spacings, got {len(row_spacing)}"
    assert len(col_spacing) == ncol + 1, f"need {ncol+1} col spacings, got {len(col_spacing)}"
    tot_len    = plot_size * ncol + sum(col_spacing)
    tot_height = plot_size * nrow + sum(row_spacing)
    fig = plt.figure(figsize=(tot_len, tot_height))
    h = []
    for j in range(ncol):
        h.append(Size.Fixed(col_spacing[j]))
        h.append(Size.Fixed(plot_size))
    h.append(Size.Fixed(col_spacing[-1]))
    v = []
    for i in range(nrow):
        v.append(Size.Fixed(row_spacing[i]))
        v.append(Size.Fixed(plot_size))
    v.append(Size.Fixed(row_spacing[-1]))
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    all_axes = []
    for i in range(nrow):
        for j in range(ncol):
            axi = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=2*j + 1, ny=2*i + 1))
            all_axes.append(axi)
    if return_fig:
        return fig, all_axes
    return all_axes
