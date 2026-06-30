"""
plots.py -- optional per-object diagnostic panel (gated by cfg.save_plots).

One figure, four panels: grz DATA | full SCARLET MODEL | RESIDUAL (data-model) |
initial dwarf RECONSTRUCTION. Uses the same sdss_rgb stretch as recon_vi so the
panels are directly comparable. Plotting failures never abort the fit (the
caller wraps this in try/except), but the function is itself defensive.
"""

import os
import sys

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# sdss_rgb scales matching recon_vi's renderer
_SCALES = dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2))
_M = 0.03


def _rgb(cube):
    from desi_lowz_funcs import sdss_rgb
    cube = np.nan_to_num(np.asarray(cube, dtype=np.float64), nan=0.0)
    return sdss_rgb([cube[0], cube[1], cube[2]], ["g", "r", "z"],
                    scales=_SCALES, m=_M)


def save_diagnostic_panel(path, data, full_model, residual, dwarf_obs,
                          title=None):
    """Write the 4-panel diagnostic figure to `path`. Returns the path or None
    on failure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(_rgb(data), origin="lower")
        axes[0].set_title("grz data")
        axes[1].imshow(_rgb(full_model), origin="lower")
        axes[1].set_title("full scarlet model")

        # residual: r-band, symmetric diverging scale
        r = np.nan_to_num(np.asarray(residual, dtype=np.float64)[1], nan=0.0)
        lim = np.nanpercentile(np.abs(r), 99) if r.size else 1.0
        lim = lim if lim > 0 else 1.0
        axes[2].imshow(r, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[2].set_title("residual (r-band)")

        axes[3].imshow(_rgb(dwarf_obs), origin="lower")
        axes[3].set_title("initial dwarf reconstruction")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        if title:
            fig.suptitle(str(title))
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=110)
        plt.close(fig)
        return path
    except Exception:                                       # noqa: BLE001
        try:
            plt.close("all")
        except Exception:                                   # noqa: BLE001
            pass
        return None
