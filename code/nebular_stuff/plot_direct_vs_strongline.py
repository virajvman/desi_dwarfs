"""
QA: direct-Te vs strong-line gas-phase metallicity, as a 2D density plot.

Compares the two oxygen-abundance estimates written to the SPEC_DERIVED HDU by
sfr_and_metallicity.build_spec_derived_hdu:

  * TE_12_LOG_OH  -- direct-Te (PyNeb + UltraNest, Plan A); 12+log(O/H)
  * Z_GAS_R23_N2  -- strong-line R23+N2 (Scholte/Dirk calibration); 12+log(O/H)

Both are on the 12+log(O/H) scale, so they are directly comparable. The
expected behaviour is that they TRACK, with the well-known systematic that the
direct (Te) abundances run LOWER than strong-line ones, growing toward high
metallicity (the "abundance discrepancy"; typically ~0.1-0.2 dex). A tight
locus with a modest offset is healthy; large scatter, an anti-correlation, or a
bimodal cloud points to a problem upstream (line fluxes, masks, or the fit).

No fitting / PyNeb / UltraNest needed -- this just reads derived columns, so it
runs anywhere astropy + matplotlib are available (e.g. locally on a downloaded
catalog).

Left panel:  direct (y) vs strong-line (x) density, with the 1:1 line.
Right panel: residual (direct - strong-line) vs strong-line density, with 0 line.
Annotated with the median offset and the NMAD scatter.

Example
-------
    python plot_direct_vs_strongline.py \
        /pscratch/.../desi_dr1_dwarf_catalog.fits --outpath direct_vs_strongline.pdf
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.table import Table
import astropy.io.fits as fits

DIRECT_COL = "TE_12_LOG_OH"
STRONG_COL = "Z_GAS_R23_N2"


def nmad(x):
    """Normalised median absolute deviation (robust sigma)."""
    x = np.asarray(x, dtype=float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("cat_path", help="Multi-extension dwarf catalog FITS file.")
    ap.add_argument("--hdu", default="SPEC_DERIVED",
                    help="HDU holding TE_12_LOG_OH and Z_GAS_R23_N2 "
                         "(default SPEC_DERIVED).")
    ap.add_argument("--min-n-ratios", type=int, default=0,
                    help="Require TE_N_RATIOS >= this for the direct point "
                         "(default 0 = no cut).")
    ap.add_argument("--require-success", action="store_true",
                    help="Require TE_FIT_SUCCESS == True (recommended).")
    ap.add_argument("--oh-lim", type=float, nargs=2, default=(7.0, 9.0),
                    help="Axis limits for 12+log(O/H) (default 7.0 9.0).")
    ap.add_argument("--gridsize", type=int, default=35,
                    help="hexbin gridsize (default 35).")
    ap.add_argument("--outpath", default="direct_vs_strongline.pdf")
    args = ap.parse_args()

    cat = Table(fits.getdata(args.cat_path, args.hdu))
    for col in (DIRECT_COL, STRONG_COL):
        if col not in cat.colnames:
            raise SystemExit(
                f"Column {col!r} not in HDU {args.hdu!r}. "
                f"Has build_spec_derived_hdu been run? Columns present: "
                f"{[c for c in cat.colnames if 'OH' in c or 'Z_GAS' in c]}"
            )

    direct = np.asarray(cat[DIRECT_COL], dtype=float)
    strong = np.asarray(cat[STRONG_COL], dtype=float)

    mask = np.isfinite(direct) & np.isfinite(strong)
    if args.require_success and "TE_FIT_SUCCESS" in cat.colnames:
        mask &= np.asarray(cat["TE_FIT_SUCCESS"], dtype=bool)
    if args.min_n_ratios > 0 and "TE_N_RATIOS" in cat.colnames:
        mask &= np.asarray(cat["TE_N_RATIOS"], dtype=int) >= args.min_n_ratios

    n = int(mask.sum())
    if n == 0:
        raise SystemExit("No rows have BOTH a finite direct and strong-line O/H "
                         "(after cuts). Nothing to plot.")
    d = direct[mask]
    s = strong[mask]
    resid = d - s

    offset = np.median(resid)
    scatter = nmad(resid)
    print(f"Overlap sample: N = {n}")
    print(f"median(direct - strong) = {offset:+.3f} dex")
    print(f"NMAD scatter            = {scatter:.3f} dex")

    lo, hi = args.oh_lim
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))

    # --- left: direct vs strong-line density + 1:1 ---
    hb = axL.hexbin(s, d, gridsize=args.gridsize, cmap="viridis",
                    mincnt=1, norm=LogNorm(), extent=(lo, hi, lo, hi))
    axL.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1")
    axL.set_xlim(lo, hi)
    axL.set_ylim(lo, hi)
    axL.set_aspect("equal")
    axL.set_xlabel(r"strong-line  $12+\log(\mathrm{O/H})$  (R23+N2)")
    axL.set_ylabel(r"direct $T_e$  $12+\log(\mathrm{O/H})$")
    axL.legend(loc="upper left", frameon=False)
    axL.text(0.97, 0.06,
             f"$N={n}$\n"
             f"median offset $={offset:+.2f}$ dex\n"
             f"NMAD $={scatter:.2f}$ dex",
             transform=axL.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    fig.colorbar(hb, ax=axL, label="count")

    # --- right: residual vs strong-line density + 0 line + running median ---
    hb2 = axR.hexbin(s, resid, gridsize=args.gridsize, cmap="magma",
                     mincnt=1, norm=LogNorm())
    axR.axhline(0.0, color="k", ls="--", lw=1.2)
    axR.axhline(offset, color="cyan", ls=":", lw=1.2,
                label=f"median {offset:+.2f}")
    # running median in strong-line bins
    bins = np.linspace(lo, hi, 13)
    centers = 0.5 * (bins[:-1] + bins[1:])
    run_med = np.full(centers.size, np.nan)
    for i in range(centers.size):
        sel = (s >= bins[i]) & (s < bins[i + 1])
        if sel.sum() >= 5:
            run_med[i] = np.median(resid[sel])
    good = np.isfinite(run_med)
    axR.plot(centers[good], run_med[good], "o-", color="white", mec="k",
             ms=5, lw=1.5, label="running median")
    axR.set_xlim(lo, hi)
    axR.set_xlabel(r"strong-line  $12+\log(\mathrm{O/H})$  (R23+N2)")
    axR.set_ylabel(r"direct $-$ strong-line  [dex]")
    axR.legend(loc="upper right", frameon=False, fontsize=8)
    fig.colorbar(hb2, ax=axR, label="count")

    fig.suptitle("Direct-$T_e$ vs strong-line gas-phase metallicity", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    outdir = os.path.dirname(os.path.abspath(args.outpath))
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(args.outpath, dpi=150)
    print(f"Wrote {args.outpath}")


if __name__ == "__main__":
    main()
