"""
Diagnostic: plot example direct-method (Plan A) posterior corner plots.

Purpose
-------
Sanity-check the single-stage joint 5-parameter UltraNest fit
(pn_functions._fit_row_ultranest) by eyeballing the full posterior for a few
representative objects:

  * the HIGHEST-SNR object in the te_mask (should be tight, unimodal), and
  * a few objects sitting right at the SNR boundary (min per-line SNR ~ 5),
    where ne-Te or Te-Av "banana" degeneracies are most likely to appear and
    leak into 12+log(O/H).

This is exactly the production fit (same r_model, priors, likelihood); we just
ask _fit_row_ultranest to also hand back the equal-weight posterior samples
(return_samples=True), so the corner plots reflect the real catalog posterior
rather than a re-implementation.

Run this WHERE the fit normally runs (NERSC / wherever ultranest, the PyNeb
custom atomic data, and the catalog all live) -- importing pn_functions builds
the PyNeb interpolation grids from the atomic-data path hard-coded at the top of
that module.

Examples
--------
    python plot_te_posteriors.py \
        /pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits \
        --line-flux-type FLUX --n-boundary 3 --outdir te_posterior_qa

    # specific objects:
    python plot_te_posteriors.py CAT.fits --targetids 39627... 39628...
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

# Make sibling modules importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astropy.table import Table
import astropy.io.fits as fits

# pn_functions builds PyNeb grids at import time (needs the atomic-data path).
from pn_functions import (
    _build_ratios,
    _fit_row_ultranest,
    PARAM_NAMES,
    RATIO_SPECS,
)

# The 7 lines required by the production te_mask (line_snr_mask in
# sfr_and_metallicity.build_spec_derived_hdu); ALL must exceed te_snr_val.
TE_LINE_NAMES = ("HALPHA", "HBETA", "HGAMMA",
                 "OIII_4363", "OIII_5007", "OII_3726", "OII_3729")

# Pretty labels for the 5 fit params + the derived metallicity.
PRETTY = {
    "ne_oii":       r"$n_e\ [\mathrm{cm^{-3}}]$",
    "te_oiii":      r"$T_e(\mathrm{[O\,III]})\ [\mathrm{K}]$",
    "Av":           r"$A_V\ [\mathrm{mag}]$",
    "log_O2_abund": r"$\log(\mathrm{O^+/H^+})$",
    "log_O3_abund": r"$\log(\mathrm{O^{++}/H^+})$",
}
OH_LABEL = r"$12+\log(\mathrm{O/H})$"


def _line_snr(cat, line, line_flux_type, min_flux=1.0):
    """Per-row SNR = flux*sqrt(ivar) for one line, with finite/ivar guards.
    Returns (snr, ok) where ok also enforces flux > min_flux."""
    fcol = f"{line}_{line_flux_type}"
    icol = f"{line}_{line_flux_type}_IVAR"
    f = np.asarray(cat[fcol], dtype=np.float64)
    iv = np.asarray(cat[icol], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        snr = f * np.sqrt(iv)
    ok = np.isfinite(f) & np.isfinite(iv) & (iv > 0) & (f > min_flux) & (snr > 0)
    snr = np.where(ok, snr, np.nan)
    return snr, ok


def te_mask_and_minsnr(cat, line_flux_type, snr_val=5.0, min_flux=1.0):
    """Reproduce the production te_mask (all 7 lines pass) and return the
    per-row MINIMUM per-line SNR (the binding constraint, since te_min_lines=7).
    For the SNR-gated lines OII_3726/3729 the production fit uses _FLUX, but the
    te_mask itself is built on line_flux_type -- mirror that here."""
    n = len(cat)
    min_snr = np.full(n, np.inf)
    all_pass = np.ones(n, dtype=bool)
    for line in TE_LINE_NAMES:
        snr, ok = _line_snr(cat, line, line_flux_type, min_flux=min_flux)
        all_pass &= ok & (snr > snr_val)
        with np.errstate(invalid="ignore"):
            min_snr = np.minimum(min_snr, np.where(np.isfinite(snr), snr, np.inf))
    min_snr = np.where(all_pass, min_snr, np.nan)
    return all_pass, min_snr


def plot_one(row, label, line_flux_type, density_diagnostic,
             min_num_live_points, outdir, tag):
    """Fit one row (Plan A, with samples) and write a corner plot. Returns the
    fit result dict (or None if the fit failed / had no samples)."""
    r, r_err, mask = _build_ratios(row, density_diagnostic, line_flux_type)
    fit = _fit_row_ultranest(
        r, r_err, mask,
        min_num_live_points=min_num_live_points,
        density_diagnostic=density_diagnostic,
        return_samples=True,
    )
    if not fit.get("success") or "samples" not in fit:
        print(f"  [{tag}] fit failed / no samples (n_ratios={fit.get('n_ratios')})")
        return None

    samples = np.asarray(fit["samples"])              # (N, 5)
    oh = np.asarray(fit["twelve_log_OH_samples"])      # (N,)
    data = np.column_stack([samples, oh])              # (N, 6)
    labels = [PRETTY[p] for p in PARAM_NAMES] + [OH_LABEL]

    oh_med, oh_err = fit["twelve_log_OH"], fit["twelve_log_OH_err"]
    title = (
        f"{label}\n"
        f"12+log(O/H) = {oh_med:.3f} +/- {oh_err:.3f}  |  "
        f"n_ratios={fit['n_ratios']}  ESS={fit['ess']:.0f}  "
        f"chi2_Av={fit['chi2_av']:.2f}  "
        f"logZ={fit['logz']:.1f}+/-{fit['logzerr']:.1f}"
    )
    outpath = os.path.join(outdir, f"posterior_{tag}.pdf")

    med = np.median(data, axis=0)
    figc = corner.corner(
        data, labels=labels, truths=med, truth_color="firebrick",
        show_titles=True, title_fmt=".3g",
        quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 8},
        label_kwargs={"fontsize": 9},
    )
    figc.suptitle(title, fontsize=10)
    figc.savefig(outpath)
    plt.close(figc)

    print(f"  [{tag}] {label}: wrote {outpath}")
    return fit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cat_path", help="Multi-extension dwarf catalog FITS file.")
    ap.add_argument("--hdu", default="FASTSPEC",
                    help="HDU with the FastSpec line fluxes (default FASTSPEC).")
    ap.add_argument("--line-flux-type", default="FLUX", choices=["FLUX", "BOXFLUX"])
    ap.add_argument("--density-diagnostic", default="OII", choices=["OII", "SII"])
    ap.add_argument("--min-live-points", type=int, default=400)
    ap.add_argument("--snr-val", type=float, default=5.0,
                    help="Per-line SNR threshold defining the te_mask (default 5).")
    ap.add_argument("--n-boundary", type=int, default=3,
                    help="How many near-SNR-boundary objects to plot (default 3).")
    ap.add_argument("--targetids", type=int, nargs="*", default=None,
                    help="Explicit TARGETIDs to plot (overrides auto-selection).")
    ap.add_argument("--outdir", default="te_posterior_qa")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cat = Table(fits.getdata(args.cat_path, args.hdu))
    has_tid = "TARGETID" in cat.colnames
    tids = np.asarray(cat["TARGETID"]) if has_tid else np.arange(len(cat))

    mask, min_snr = te_mask_and_minsnr(
        cat, args.line_flux_type, snr_val=args.snr_val,
    )
    n_te = int(mask.sum())
    print(f"te_mask: {n_te}/{len(cat)} rows pass all {len(TE_LINE_NAMES)} "
          f"lines at SNR > {args.snr_val:g}")
    if n_te == 0:
        print("No rows pass te_mask; nothing to plot.")
        return

    # --- choose which rows to fit ---
    if args.targetids:
        if not has_tid:
            raise SystemExit("--targetids given but catalog has no TARGETID column.")
        picks = []
        for t in args.targetids:
            w = np.flatnonzero(tids == t)
            if w.size == 0:
                print(f"  WARNING: TARGETID {t} not in catalog; skipping.")
                continue
            picks.append((w[0], f"TARGETID {t} (min-SNR={min_snr[w[0]]:.1f})", f"tid{t}"))
    else:
        idx_te = np.flatnonzero(mask)
        order = idx_te[np.argsort(min_snr[idx_te])]   # ascending min-SNR
        picks = []
        # highest-SNR object
        hi = order[-1]
        picks.append((hi, f"highest SNR  (min-SNR={min_snr[hi]:.1f}, "
                          f"TARGETID={tids[hi]})", "highSNR"))
        # median-SNR object (context)
        mid = order[len(order) // 2]
        picks.append((mid, f"median SNR  (min-SNR={min_snr[mid]:.1f}, "
                           f"TARGETID={tids[mid]})", "medSNR"))
        # n_boundary lowest-SNR (just above the threshold)
        for rank, ii in enumerate(order[:args.n_boundary]):
            picks.append((ii, f"near boundary #{rank+1}  (min-SNR={min_snr[ii]:.1f}, "
                              f"TARGETID={tids[ii]})", f"boundary{rank+1}"))

    print(f"Plotting {len(picks)} objects -> {args.outdir}/")
    for i, label, tag in picks:
        plot_one(cat[i], label, args.line_flux_type, args.density_diagnostic,
                 args.min_live_points, args.outdir, tag)


if __name__ == "__main__":
    main()
