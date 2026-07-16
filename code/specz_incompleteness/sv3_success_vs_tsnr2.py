'''
Empirical z-success vs TSNR2_BGS split by spectral type (color), from the SV3
single-exposure vs deep-truth table -- the direct test of TSNR2-only
redshift-failure weights (PROVABGS-style w_ZF = 1/f(rfib, TSNR2)).

Message: within the BGS depth range, success barely moves along the TSNR2
axis but splits wide open by type at fixed TSNR2 -- the covariate that
matters is missing from depth-only weights.

CAVEAT printed on the figure: this truth table is selected to
EFFTIME 160-200s (nominal BGS depth), so the TSNR2 lever arm is narrow by
construction; the point is the type gap at fixed depth, not the flatness.

Figures -> ~/DESI2_LOWZ/quenched_fracs_nbs/incompleteness_plots/
'''

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plot_style import (apply_paper_style, make_subplots,
                        MARGIN_LABEL, MARGIN_SPLIT, MARGIN_PAD)
from sv3_type_dependence_plot import (load_sample, boot_stat, _wmean,
                                      RED_CUT, BLUE_CUT, C_RED, C_BLUE)

OUT_STEM = ("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/"
            "incompleteness_plots/sv3_success_vs_tsnr2")

TSNR2_BINS = np.array([1100, 1200, 1280, 1360, 1450])   # narrow by selection
RFIB_FAINT = 21.0    # faint slice where failures actually happen


def load_tsnr2():
    from astropy.table import Table
    t = Table.read("/global/cfs/cdirs/desi/users/chahah/bgs-cmxsv/sv-paper/"
                   "sv3.bgs_exps.efftime160_200.zsuccess.fuji.fits")
    return t


def binned_rate_tsnr2(d, tsnr2, class_mask, succ_key, bins):
    succ = d[succ_key].astype(float)
    def fn(w):
        out = np.full(len(bins) - 1, np.nan)
        for i in range(len(bins) - 1):
            inb = class_mask & (tsnr2 >= bins[i]) & (tsnr2 < bins[i + 1])
            if (inb & (w > 0)).sum() >= 25:
                out[i] = _wmean(succ[inb], w[inb])
        return out
    return boot_stat(fn, d)


if __name__ == "__main__":
    apply_paper_style()
    d = load_sample()

    # re-read TSNR2 for the same masked rows (load_sample applies mask `m`
    # internally; rebuild it identically)
    t = load_tsnr2()
    ztrue = np.asarray(t["Z_TRUE"])
    rfib_all = np.asarray(t["FIBER_RMAG_DRED"])
    gr_all = np.asarray(t["GMAG_DRED"]) - np.asarray(t["RMAG_DRED"])
    m = ((np.asarray(t["SV3_BGS_TARGET"]) & 2) != 0)
    m &= np.asarray(t["DEEP_TRUE"]).astype(bool)
    m &= (ztrue > 0.001) & (ztrue < 0.15)
    m &= np.isfinite(rfib_all) & np.isfinite(gr_all)
    tsnr2 = np.asarray(t["TSNR2_BGS"], dtype=float)[m]
    assert len(tsnr2) == len(d["rfib"])

    red = d["gr"] > RED_CUT
    blue = d["gr"] < BLUE_CUT
    faint = d["rfib"] > RFIB_FAINT
    cen = 0.5 * (TSNR2_BINS[1:] + TSNR2_BINS[:-1])

    fig, flat = make_subplots(ncol=2, nrow=1, plot_size=3.1,
                              col_spacing=[MARGIN_LABEL, MARGIN_SPLIT, MARGIN_PAD],
                              row_spacing=[MARGIN_LABEL, 0.55],
                              return_fig=True)

    for ax, fm, name in [(flat[0], ~faint, rf"$r_{{\rm fib}} < {RFIB_FAINT}$"),
                         (flat[1], faint, rf"$r_{{\rm fib}} > {RFIB_FAINT}$")]:
        for cls, color, label in [(red & fm, C_RED, f"red ($g-r>{RED_CUT}$)"),
                                  (blue & fm, C_BLUE, f"blue ($g-r<{BLUE_CUT}$)")]:
            v, lo, hi = binned_rate_tsnr2(d, tsnr2, cls, "succ40", TSNR2_BINS)
            ax.errorbar(cen, v, yerr=[v - lo, hi - v], color=color, ls="-",
                        marker="o", ms=4, lw=1.5,
                        label=label + r", ZWARN=0 & $\Delta\chi^2>40$")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(r"TSNR2$_{\rm BGS}$ ($\propto$ EFFTIME$_{\rm BRIGHT}$)")
        ax.set_ylim(0.35, 1.04)
        ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    flat[0].set_ylabel("redshift success rate (vs deep truth)")
    flat[0].legend(fontsize=8, loc="lower left")
    flat[1].set_yticklabels([])

    fig.text(0.5, 0.955,
             "at fixed TSNR2 (the LSS weighting variable), success splits by type "
             "-- SV3 deep-truth, BGS Bright, $z<0.15$",
             ha="center", fontsize=11)
    flat[1].text(0.04, 0.06,
                 "TSNR2 range narrow by sample construction\n"
                 "(single exposures at nominal depth)",
                 transform=flat[1].transAxes, fontsize=7.5, color="0.45")

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=250, bbox_inches="tight")
        print(f"wrote {OUT_STEM}.{ext}")
