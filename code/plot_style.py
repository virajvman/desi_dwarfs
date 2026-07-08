"""Shared matplotlib style and layout helpers for catalog-paper figures."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import scipy.optimize as opt
from scipy.signal import wiener

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


# ---------------------------------------------------------------------------
# 2D distribution / contour + alternating-line helpers.
#
# Moved here (verbatim) from desi_lowz_funcs.py so the catalog-paper figure
# scripts can import them without pulling in the full DESI stack (desi_lowz_funcs
# imports desispec at module top, which is not available in a local
# astropy+matplotlib environment). desi_lowz_funcs re-exports these to preserve
# existing `from desi_lowz_funcs import plot_2d_dist, make_alternating_plot`
# call sites.
# ---------------------------------------------------------------------------

def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level


def plot_2d_dist(x,y, nxbins, nybins,
                cmin=1.e-4, cmax=1.0, smooth=None, clevs=None,ax=None, bounds=None,plot_pcol=False,
                color = "k",cmap=None,filled=False,label="1",cmap_alpha=0.5,lw_scale=1,alternating_contours=False):
    """
    construct and plot a binned, 2d distribution in the x-y plane
    using nxbins and nybins in x- and y- direction, respectively

    log = specifies whether logged quantities are passed to be plotted on log-scale outside this routine
    """

    weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(bounds[0], bounds[1], nxbins),np.linspace(bounds[2], bounds[3], nybins)))

    H = np.rot90(H); H = np.flipud(H);

    X,Y = np.meshgrid(xbins[:-1],ybins[:-1])

    if smooth != None:
        H = wiener(H, mysize=smooth)

    H = H/np.sum(H)
    Hmask = np.ma.masked_where(H==0,H)

    if plot_pcol:
        pcol = ax.pcolormesh(X, Y,(Hmask),cmap=plt.cm.BuPu, norm = LogNorm(vmin=cmin*np.max(Hmask), vmax=cmax*np.max(Hmask) ), linewidth=0., rasterized=True)
        pcol.set_edgecolor('face')

    # plot contours if contour levels are specified in clevs
    contour_paths = []
    if clevs is not None:
        lvls = []
        for cld in clevs:
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )
            lvls.append(sig)

        if filled:
            cs = ax.contourf(X, Y, H, linewidths=np.array([1.0,0.75, 0.5, 0.25])[::-1]*lw_scale, cmap=cmap, levels = sorted(lvls),
                    norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]],alpha=cmap_alpha, rasterized=True)
        else:
            cs = ax.contour(X, Y, H, linewidths=3*np.array([1.0,0.75, 0.5, 0.25])[::-1]*lw_scale, colors=color, levels = sorted(lvls),
                    norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]], linestyles="--")

            if alternating_contours:
                cs_bg = ax.contour(X, Y, H,
                       linewidths=3*np.array([1.0, 0.75, 0.5, 0.25])[::-1]*lw_scale,
                       colors='black',
                       levels=sorted(lvls),
                       norm=LogNorm(),
                       extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                       rasterized=True,
                        linestyles="-")

                # Draw white line on top
                cs = ax.contour(X, Y, H,
                                linewidths=3*np.array([1.0, 0.75, 0.5, 0.25])[::-1]*lw_scale,
                                colors='white',
                                levels=sorted(lvls),
                                norm=LogNorm(),
                                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                                rasterized=True,
                               linestyles="--")

            # dashed linestyle is applied via `linestyles="--"` at contour
            # creation above -- works on all matplotlib versions, unlike
            # QuadContourSet.set_linestyle() which only exists on mpl >= 3.8.

        # Save contour paths
        # contour_paths = [c for c in cs.collections]

        if hasattr(cs, "collections"):          # matplotlib < 3.10
            contour_paths = list(cs.collections)
        else:                                    # matplotlib >= 3.10
            contour_paths = cs.get_paths()

     # At end of function
    if not filled:
        ax.plot([-5,-10], [-5,-10], color=color, linewidth=2.5, label=label, alpha=cmap_alpha)
    else:
        from matplotlib.patches import Patch
        ax.plot([-5,-10], [-5,-10], color=cmap(0.6), linewidth=8, label=label)  # proxy line

    return contour_paths


def make_alternating_plot(ax,x,y,dash_len=5,color_1="yellowgreen",color_2="k",lw=1,alpha=1):
    '''
    dash_len is in number of points
    '''

    # Create segments of the line
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Alternate colors for each segment
    colors = []
    for i in range(len(segments)):
        # Alternate every `dash_len` segments
        if (i // dash_len) % 2 == 0:
            colors.append(color_1)
        else:
            colors.append(color_2)

    # Create the line collection
    lc = LineCollection(segments, colors=colors, linewidth=lw,alpha=alpha)

    ax.add_collection(lc)

    return
