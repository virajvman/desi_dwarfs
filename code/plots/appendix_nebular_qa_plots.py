"""
appendix_nebular_qa_plots.py
============================

Two catalog-paper appendix figures for "Details on Nebular Property
Estimation" (QA of the direct-Te pipeline against the population-level
prescriptions):

  1. direct_vs_strongline_zgas.pdf
     Left : direct-Te 12+log(O/H) (TE_12_LOG_OH) vs strong-line R23+N2
            (Z_GAS_R23_N2) as a 2D density, with the 1:1 line.
     Right: residual (direct - strong-line) vs strong-line, with white
            running medians (16-84th percentile bars), annotated with the
            median offset and NMAD.

  2. av_direct_vs_param.pdf
     Left : per-object direct-fit A_V (TE_AV, joint {ne, Te, Av, O+, O++}
            posterior median) vs log M*, as a 2D density, with the
            mass-based parametrization overlaid: BD = model_hahb(logM*)
            (logistic fit to stacked Halpha/Hbeta, 2.86 low-mass asymptote)
            dereddened against the intrinsic 2.79 through the CCM89 law.
            A vertical bar shows the median TE_AV_ERR of the sample.
     Right: residual TE_AV - A_V,model(logM*) vs log M*, with white running
            medians (16-84th percentile bars).

Both read only catalog columns (MAIN + SPEC_DERIVED), so they run anywhere
astropy + matplotlib are available (e.g. locally on a downloaded catalog).
Sample: DWARF_MASKBIT == 0 plus finite values of the plotted columns; the
direct-Te subset is auroral-line selected (see the paper appendix caveat).

Usage
-----
    python appendix_nebular_qa_plots.py CATALOG.fits --outdir /path/to/figures
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.table import Table

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
for _p in (_CODE_DIR, os.path.join(_CODE_DIR, "nebular_stuff")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_style import (
    apply_paper_style, make_subplots,
    MARGIN_LABEL, MARGIN_PAD, MARGIN_SPLIT, MARGIN_SHARED
)

#todo: import the make_alternating_line function here and update tehe cmap to be consistent with others
#once plot_style as this make_alternating_function, we might need to update the import statement elsewhere.


from cardelli_attenuation import k_ccm89, model_hahb, BALMER_INTRINSIC


def nmad(x):
    """Normalised median absolute deviation (robust sigma)."""
    x = np.asarray(x, dtype=float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def av_from_bd(bd):
    """A_V implied by an observed Balmer decrement through the pipeline
    convention: CCM89 (R_V=3.1), dereddened against the intrinsic 2.79.
    E(B-V) = 2.5 log10(BD/2.79) / (k_Hb - k_Ha);  A_V = E(B-V) * k(5500)."""
    ebv = 2.5 * np.log10(np.asarray(bd, dtype=float) / BALMER_INTRINSIC) \
        / (k_ccm89(4861.0) - k_ccm89(6563.0))
    return np.clip(ebv, 0, np.inf) * k_ccm89(5500.0)


def running_median(x, y, edges, min_n=20):
    """Median + 16/84th percentiles of y in bins of x (skip bins with < min_n)."""
    cen, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x < b)
        if sel.sum() < min_n:
            continue
        cen.append(0.5 * (a + b))
        med.append(np.median(y[sel]))
        lo.append(np.percentile(y[sel], 16))
        hi.append(np.percentile(y[sel], 84))
    cen, med, lo, hi = map(np.asarray, (cen, med, lo, hi))
    return cen, med, med - lo, hi - med


def add_median_points(ax, x, y, edges, min_n=20):
    cen, med, elo, ehi = running_median(x, y, edges, min_n=min_n)
    ax.errorbar(cen, med, yerr=[elo, ehi], fmt="o", color="w",
                mec="k", mew=0.8, ecolor="k", elinewidth=1.0,
                capsize=0, ms=5, zorder=5)


def two_panel_axes():
    """Standard two-panel layout used by the other comparison figures."""
    fig, axes = make_subplots(
        ncol=2, nrow=1, plot_size=2.25, return_fig=True,
        row_spacing=[MARGIN_LABEL - 0.4, MARGIN_PAD],
        col_spacing=[MARGIN_LABEL - 0.2, MARGIN_SPLIT, MARGIN_PAD],
    )
    return fig, axes


def plot_direct_vs_strongline(main, der, outpath):
    ok = (
        (np.asarray(main["DWARF_MASKBIT"]) == 0)
        & np.isfinite(np.asarray(der["TE_12_LOG_OH"]))
        & np.isfinite(np.asarray(der["Z_GAS_R23_N2"]))
    )
    
    strong = np.asarray(der["Z_GAS_R23_N2"])[ok]
    direct = np.asarray(der["TE_12_LOG_OH"])[ok]
    resid = direct - strong

    med_off, sig = np.median(resid), nmad(resid)
    print(f"  direct-vs-strongline: N={ok.sum()}, "
          f"median offset={med_off:+.3f} dex, NMAD={sig:.3f} dex")

    fig, ax = make_subplots(
        ncol=1, nrow=2, plot_size=2.25, return_fig=True,
        row_spacing=[MARGIN_LABEL - 0.4, MARGIN_SHARED, MARGIN_PAD],
        col_spacing=[MARGIN_LABEL - 0.2, MARGIN_PAD],
    )

    lim = (7.0, 8.75)
    ax[1].hist2d(direct, strong, bins=[60, 60],
               range=[lim, lim], norm=LogNorm(), cmap="magma", rasterized=True)

    ax[1].set_ylabel(r"$12 + \log_{10}(\mathrm{O/H})$  [$R_{23}$+$N2$]")
    ax[1].set_xticklabels([])

    ax[0].set_ylabel(r"$12 + \log_{10}(\mathrm{O/H})$  [direct $T_e$, lit.]")
    ax[0].set_xlabel(r"$12 + \log_{10}(\mathrm{O/H})$  [direct $T_e$]")

    for axi in ax:
        axi.set_yticks([7,7.5,8,8.5,9])
        axi.set_xticks([7,7.5,8,8.5,9])
        axi.set_xlim(lim); axi.set_ylim(lim)
        axi.plot(lim, lim, ls="--", color="k", lw=1.0, zorder=4)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  saved {outpath}")

def plot_av_direct_vs_param(main, der, outpath):
    ok = (
        (np.asarray(main["DWARF_MASKBIT"]) == 0)
        & np.isfinite(np.asarray(der["TE_AV"]))
        & (np.asarray(der["TE_AV_ERR"]) < 0.2)
        & (der["TE_FIT_SUCCESS"] == True )
        & np.isfinite(np.asarray(main["LOG_MSTAR_M24"]))
        & (np.asarray(der["TE_CHI2_AV_ML"] < 5))
    )
    logm = np.asarray(main["LOG_MSTAR_M24"])[ok]
    av = np.asarray(der["TE_AV"])[ok]
    av_err = np.asarray(der["TE_AV_ERR"])[ok]
    av_model = av_from_bd(model_hahb(logm))
    resid = av - av_model
    med_err = np.nanmedian(av_err)

    #print the median te_chi2_av
    print("Median AV CHI2 = ", np.median( np.asarray(der["TE_CHI2_AV_ML"])[ok] )  )
    print("16per AV CHI2 = ", np.percentile( np.asarray(der["TE_CHI2_AV_ML"])[ok], 16 )  )
    print("84per AV CHI2 = ", np.percentile( np.asarray(der["TE_CHI2_AV_ML"])[ok], 84 )  )

    print(f"  av-direct-vs-param: N={ok.sum()}, median TE_AV_ERR={med_err:.3f}, "
          f"median resid={np.median(resid):+.3f}, NMAD={nmad(resid):.3f}")

    fig, ax = make_subplots(
        ncol=1, nrow=1, plot_size=2.25, return_fig=True,
        row_spacing=[MARGIN_LABEL - 0.4, MARGIN_PAD],
        col_spacing=[MARGIN_LABEL - 0.2, MARGIN_PAD],
    )

    mlim = (6.5, 9.25)
    alim = (0.0, 1.6)
    ax[0].hist2d(logm, av, bins=[60, 60],
               range=[mlim, alim], norm=LogNorm(), cmap="magma", rasterized=True)
    mgrid = np.linspace(*mlim, 200)
    ax[0].plot(mgrid, av_from_bd(model_hahb(mgrid)), color="deepskyblue", lw=2.0,
             zorder=4, label=r"mass-based model")
    # median per-object A_V uncertainty, shown as a single vertical bar
    ax[0].errorbar([mlim[0] + 0.25], [alim[1] - 0.35], yerr=[med_err],
                 fmt="none", ecolor="k", elinewidth=1.2, capsize=2.5, zorder=5)
    ax[0].text(mlim[0] + 0.33, alim[1] - 0.35, "median\nerror",
             fontsize=8, ha="left", va="center")
    ax[0].legend(loc="upper right", handlelength=1.2)
    ax[0].set_xlim(mlim); ax[0].set_ylim(alim)
    ax[0].set_xlabel(r"$\log_{10}\,M_\star/M_\odot$")
    ax[0].set_ylabel(r"$A_V$ (nebular)")

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  saved {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cat_path", help="Multi-extension dwarf catalog FITS file.")
    ap.add_argument("--outdir", default=".", help="Directory for the output PDFs.")
    args = ap.parse_args()

    apply_paper_style()

    print(f"Reading {args.cat_path}")
    main_tab = Table.read(args.cat_path, hdu="MAIN")
    der_tab = Table.read(args.cat_path, hdu="SPEC_DERIVED")

    os.makedirs(args.outdir, exist_ok=True)
    plot_direct_vs_strongline(
        main_tab, der_tab,
        os.path.join(args.outdir, "direct_vs_strongline_zgas.pdf"))

    #TODO: cross match our catalog with other catalogs and see if we have are measuring consistent O/H
    #we are seeing this systematic offset in this very low metallicity direct pop, but maybe that is expected ..?
    #draw comparison samples from Sui et al. and other literature like CLASSY, SDSS etc. 

    plot_av_direct_vs_param(
        main_tab, der_tab,
        os.path.join(args.outdir, "av_direct_vs_param.pdf"))


if __name__ == "__main__":
    main()
