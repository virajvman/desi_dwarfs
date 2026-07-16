'''
Real-data demonstration that DESI redshift success is strongly spectral-type
dependent at fixed fiber magnitude and depth -- and that a hard DELTACHI2
quality cut therefore biases red/quenched fractions low.

Uses ONLY ChangHoon Hahn's SV3 single-exposure vs deep-coadd truth table
(no simulations, no crossmatch):
    sv3.bgs_exps.efftime160_200.zsuccess.fuji.fits
Each row = one ~nominal-depth (160-200s efftime) single exposure of a BGS
target whose true redshift comes from the SV3 deep coadd.

Definitions match the specz_incompleteness pipeline (config.py):
  correct       |z_1exp - z_deep| / (1 + z_deep) < 0.0033
  success       RR_ZWARN == 0 & correct           ("ZWARN only")
  success+cut   additionally RR_DELTACHI2 > 40    (standard catalog cut)

Panel A: success rate vs fiber mag, split by g-r color (red / blue).
Panel B: red fraction a cut catalog would measure, relative to truth.

Error bars: bootstrap over unique TARGETIDs (repeat exposures of the same
galaxy are correlated; per-exposure binomial errors would be optimistic).

Run:  source desi_environment.sh main; python sv3_type_dependence_plot.py
Figures -> ~/DESI2_LOWZ/quenched_fracs_nbs/incompleteness_plots/sv3_type_dependence.{png,pdf}
'''

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plot_style import (apply_paper_style, make_subplots,
                        MARGIN_LABEL, MARGIN_SPLIT, MARGIN_PAD)

SV3_FILE = ("/global/cfs/cdirs/desi/users/chahah/bgs-cmxsv/sv-paper/"
            "sv3.bgs_exps.efftime160_200.zsuccess.fuji.fits")
OUT_STEM = ("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/"
            "incompleteness_plots/sv3_type_dependence")

CATASTROPHIC = 0.0033
RED_CUT, BLUE_CUT = 0.75, 0.60          # g-r split (middle omitted for contrast)
ZMAX = 0.15
BINS = np.array([18.0, 19.0, 19.5, 20.0, 20.4, 20.8, 21.2, 21.6, 22.4])
N_BOOT = 500
RNG = np.random.default_rng(42)

C_RED, C_BLUE = "#c23b22", "#2a6fbb"


def load_sample():
    t = Table.read(SV3_FILE)
    ztrue = np.asarray(t["Z_TRUE"])
    rfib = np.asarray(t["FIBER_RMAG_DRED"])
    gr = np.asarray(t["GMAG_DRED"]) - np.asarray(t["RMAG_DRED"])
    m = ((np.asarray(t["SV3_BGS_TARGET"]) & 2) != 0)      # BGS_BRIGHT
    m &= np.asarray(t["DEEP_TRUE"]).astype(bool)          # reliable deep truth
    m &= (ztrue > 0.001) & (ztrue < ZMAX)
    m &= np.isfinite(rfib) & np.isfinite(gr)

    correct = np.abs(np.asarray(t["RR_Z"]) - ztrue) / (1 + ztrue) < CATASTROPHIC
    succ0 = (np.asarray(t["RR_ZWARN"]) == 0) & correct
    succ40 = succ0 & (np.asarray(t["RR_DELTACHI2"]) > 40)

    out = dict(rfib=rfib[m], gr=gr[m], succ0=succ0[m], succ40=succ40[m],
               tid=np.asarray(t["TARGETID"])[m])
    print(f"sample: {m.sum()} exposures, {len(np.unique(out['tid']))} targets")
    return out


def boot_stat(stat_fn, d, n_boot=N_BOOT):
    """Bootstrap over unique targets (multiplicity-weighted).

    stat_fn takes per-exposure weights (float) and returns binned stats.
    Returns (value, lo, hi) with a 68% bootstrap interval.
    """
    utid, inv = np.unique(d["tid"], return_inverse=True)
    val = stat_fn(np.ones(len(d["tid"])))
    boots = np.empty((n_boot, np.size(val)))
    for b in range(n_boot):
        take = RNG.integers(0, len(utid), len(utid))
        counts = np.bincount(take, minlength=len(utid))
        boots[b] = stat_fn(counts[inv].astype(float))
    lo, hi = np.nanpercentile(boots, [16, 84], axis=0)
    return np.atleast_1d(val), np.atleast_1d(lo), np.atleast_1d(hi)


def _wmean(values, w):
    tot = w.sum()
    return (values * w).sum() / tot if tot > 0 else np.nan


def binned_rate(d, class_mask, succ_key):
    """Success rate per rfib bin for one color class, with bootstrap errors."""
    succ = d[succ_key].astype(float)
    def fn(w):
        out = np.full(len(BINS) - 1, np.nan)
        for i in range(len(BINS) - 1):
            inb = class_mask & (d["rfib"] >= BINS[i]) & (d["rfib"] < BINS[i + 1])
            if (inb & (w > 0)).sum() >= 15:
                out[i] = _wmean(succ[inb], w[inb])
        return out
    return boot_stat(fn, d)


def binned_redfrac_ratio(d, red, blue, succ_key):
    """(red fraction among kept) / (red fraction in truth) per rfib bin."""
    both = red | blue
    redf = red.astype(float)
    kept_all = d[succ_key]
    def fn(w):
        out = np.full(len(BINS) - 1, np.nan)
        for i in range(len(BINS) - 1):
            inb = both & (d["rfib"] >= BINS[i]) & (d["rfib"] < BINS[i + 1])
            kept = inb & kept_all
            if (inb & (w > 0)).sum() >= 30 and (red & inb & (w > 0)).sum() >= 8:
                rf_true = _wmean(redf[inb], w[inb])
                rf_kept = _wmean(redf[kept], w[kept])
                out[i] = rf_kept / rf_true if rf_true > 0 else np.nan
        return out
    return boot_stat(fn, d)


if __name__ == "__main__":
    apply_paper_style()
    d = load_sample()
    red = d["gr"] > RED_CUT
    blue = d["gr"] < BLUE_CUT
    cen = 0.5 * (BINS[1:] + BINS[:-1])

    fig, axes = make_subplots(ncol=2, nrow=1, plot_size=3.1,
                              col_spacing=[MARGIN_LABEL, MARGIN_SPLIT, MARGIN_PAD],
                              row_spacing=[MARGIN_LABEL, 0.55],
                              return_fig=True)

    # ------------------------------------------------------------- Panel A
    ax = axes[0]
    for cls, color, name in [(red, C_RED, f"red ($g-r > {RED_CUT}$)"),
                             (blue, C_BLUE, f"blue ($g-r < {BLUE_CUT}$)")]:
        for key, ls, extra in [("succ0", "-", "ZWARN=0"),
                               ("succ40", "--", r"+ $\Delta\chi^2>40$")]:
            v, lo, hi = binned_rate(d, cls, key)
            ax.errorbar(cen, v, yerr=[v - lo, hi - v], color=color, ls=ls,
                        marker="o", ms=3.5, lw=1.4,
                        label=f"{name}, {extra}")
    ax.set_xlabel(r"$r$ fiber magnitude")
    ax.set_ylabel("redshift success rate (vs deep truth)")
    ax.set_ylim(0.25, 1.04)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("same depth + fiber mag: success\ndepends on spectral type", fontsize=11)
    ax.text(0.04, 0.62,
            f"catastrophic: $|\\Delta z|/(1+z) > {CATASTROPHIC}$\n"
            "errors: bootstrap over targets",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="0.4")

    # ------------------------------------------------------------- Panel B
    ax = axes[1]
    for key, color, ls, name in [
            ("succ40", "k", "-", r"ZWARN=0 & $\Delta\chi^2>40$ (standard cut)"),
            ("succ0", "0.45", "--", "ZWARN=0 only")]:
        v, lo, hi = binned_redfrac_ratio(d, red, blue, key)
        ax.errorbar(cen, v, yerr=[v - lo, hi - v], color=color, ls=ls,
                    marker="s", ms=3.5, lw=1.4, label=name)
    ax.axhline(1.0, color=C_RED, lw=0.8, ls=":")
    ax.set_xlabel(r"$r$ fiber magnitude")
    ax.set_ylabel("measured / true red fraction")
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=8, loc="lower left")
    ax.annotate("DESI dwarfs:\n" + r"$r_{\rm fib}$ out to $\sim$23.5 $\rightarrow$",
                xy=(0.93, 0.42), xycoords="axes fraction", ha="right",
                fontsize=10, color=C_RED, fontstyle="italic")
    ax.set_title("hard quality cut biases the\nmeasured quenched fraction low", fontsize=11)

    fig.text(0.5, 0.965,
             "DESI SV3: single exposures (160-200s efftime) vs deep-coadd truth "
             f"-- BGS Bright, $z_{{\\rm true}} < {ZMAX}$",
             ha="center", fontsize=11.5)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=250, bbox_inches="tight")
        print(f"wrote {OUT_STEM}.{ext}")
