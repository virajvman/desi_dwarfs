"""
stack_direct_metallicity.py
===========================

Direct-method (T_e) nebular abundances for the M* x H-alpha-EW stacked
spectra produced by `stack_mstar_haew_3bin.py` and fit with custom
FastSpecFit (`job_scripts/run_stack_fastspec_haew_3bin.sh`).

For each (log M*, H-alpha EW) bin there is one FastSpecFit stack output
`fastspec_stack_ALL_mstar_{mlo}_{mhi}_{ewtoken}.fits` whose emission-line
table (hdu=3) has:
    row 0     -> the mean stack
    rows 1..N -> the bootstrap realizations of that stack

This script:
  1. Globs the per-bin FastSpecFit stack outputs.
  2. Keeps only bins where the MEAN stack detects all seven lines required
     by the direct method
     (HALPHA, HBETA, HGAMMA, OIII_4363, OIII_5007, OII_3726, OII_3729)
     at SNR > 3 with flux > 1 (FastSpec units), reusing
     `sfr_and_metallicity.line_snr_mask` with the same gating as
     `add_nebular_props.py`.
  3. Runs the UltraNest direct-method fit
     (`pn_functions.compute_direct_metallicities`) on the mean stack AND
     its bootstrap rows.
  4. Reports per bin:
       - central value  = the mean-stack posterior median, and
       - uncertainty    = the 16/84 spread of the per-bootstrap posterior
                          medians (sample variance; option B).
  5. Writes ONE results row per candidate bin (kept and dropped alike, with
     a DETECTED_7LINE flag) to a FITS + ECSV table so the M*/EW trends can
     be plotted later.

Outputs (written to STACK_PATH):
  - direct_metallicity_results.fits
  - direct_metallicity_results.ecsv
  - plots/oh_av_vs_mstar.png   (quick sanity check)

Usage:
    python stack_direct_metallicity.py --line-flux-type BOXFLUX
    python stack_direct_metallicity.py --line-flux-type FLUX
"""

import argparse
import os
import sys
import glob
import re

# ``code/nebular_stuff/`` is a flat folder of scripts (no __init__.py) and
# the project uses cwd-style imports (e.g. ``from sfr_and_metallicity import
# ...``). This file already lives in nebular_stuff/, but make the folder (and
# code/) importable so it runs regardless of the caller's cwd / PYTHONPATH.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _CODE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from astropy.table import Table

from sfr_and_metallicity import line_snr_mask


# =============================================================================
# CONFIG
# =============================================================================

# Directory holding the FastSpecFit stack outputs (same dir the stacking
# runner writes to).
STACK_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_haew_3bin/"

# FastSpecFit stack outputs (one per bin). The runner prepends "fastspec_"
# to each "stack_ALL_mstar_..._ew....fits" input.
INPUT_GLOB = "fastspec_stack_ALL_mstar_*.fits"

# HDU index of the FastSpec emission-line table (HALPHA_FLUX, etc.). The
# existing pipeline reads the stack fastspec output as Table.read(f, hdu=3).
FASTSPEC_HDU = 3

# EW binning, kept identical to stack_mstar_haew_3bin.py so the ew_token in
# the filename maps back to numeric edges.
EW_EDGES = [0.0, 30.0, 300.0, np.inf]
EW_TOKENS = ["ew_lt30", "ew_30_300", "ew_gt300"]

# Seven lines required by the direct method, and the detection gate. These
# match the TE_* knobs in add_nebular_props.py.
TE_LINE_NAMES = ["HALPHA", "HBETA", "HGAMMA",
                 "OIII_4363", "OIII_5007",
                 "OII_3726", "OII_3729"]
SNR_VAL = 3
MIN_LINES = 7
MIN_FLUX = 1.0   # FastSpec units (1e-17 erg/cm2/s)

# UltraNest fit settings (same as the catalog driver).
N_JOBS = 128
MIN_NUM_LIVE_POINTS = 400
SAMPLER_KWARGS = {"frac_remain": 0.01, "max_iters": 40000, "max_ncalls": int(1e5)}
USE_INFORMATIVE_PRIORS = False

# Parameters returned by compute_direct_metallicities (PARAM_NAMES +
# twelve_log_OH). Left-hand name is the lower-case column in the result
# Table; right-hand name is the UPPER_CASE column we write out.
PARAM_MAP = [
    ("ne_oii",        "NE_OII"),
    ("te_oiii",       "TE_OIII"),
    ("Av",            "AV"),
    ("log_O2_abund",  "LOG_O2_ABUND"),
    ("log_O3_abund",  "LOG_O3_ABUND"),
    ("twelve_log_OH", "TWELVE_LOG_OH"),
]

OUT_BASENAME = "direct_metallicity_results"


# =============================================================================
# HELPERS
# =============================================================================

# fastspec_stack_ALL_mstar_{mlo}_{mhi}_{ewtoken}.fits
_FNAME_RE = re.compile(
    r"fastspec_stack_ALL_mstar_"
    r"(?P<mlo>[-\d.]+)_(?P<mhi>[-\d.]+)_"
    r"(?P<ewtoken>ew_\w+)\.fits$"
)


def parse_bin_from_filename(path):
    """Return (mstar_min, mstar_max, ew_token, ew_min, ew_max) or None."""
    m = _FNAME_RE.search(os.path.basename(path))
    if m is None:
        return None
    mlo = float(m.group("mlo"))
    mhi = float(m.group("mhi"))
    token = m.group("ewtoken")
    if token not in EW_TOKENS:
        return None
    j = EW_TOKENS.index(token)
    ew_min, ew_max = EW_EDGES[j], EW_EDGES[j + 1]
    return mlo, mhi, token, ew_min, ew_max


def read_nobj_from_input_stack(mlo, mhi, token):
    """Best-effort NOBJ for a bin from the matching input stack FITS.

    The stacking script writes NOBJ into the STACKINFO table of
    stack_ALL_mstar_{mlo}_{mhi}_{token}.fits. FastSpecFit may not carry it
    into its output, so we read it from the input file if present.
    """
    in_path = os.path.join(
        STACK_PATH, f"stack_ALL_mstar_{mlo:.2f}_{mhi:.2f}_{token}.fits"
    )
    if not os.path.exists(in_path):
        return -1
    try:
        info = Table.read(in_path, hdu="STACKINFO")
        if "NOBJ" in info.colnames and len(info) > 0:
            return int(info["NOBJ"][0])
    except Exception:
        pass
    return -1


def boot_spread(values):
    """16/50/84 spread of finite values -> (err, err_lo, err_hi, n_used).

    err     = 0.5 * (p84 - p16)
    err_lo  = p50 - p16
    err_hi  = p84 - p50
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, 0
    p16, p50, p84 = np.nanpercentile(arr, [16, 50, 84])
    return 0.5 * (p84 - p16), p50 - p16, p84 - p50, int(arr.size)


# =============================================================================
# MAIN
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Direct-method nebular abundances for M* x H-alpha-EW stacked "
            "spectra from FastSpecFit stack outputs."
        ),
    )
    parser.add_argument(
        "--line-flux-type",
        required=True,
        choices=("FLUX", "BOXFLUX"),
        help=(
            "FastSpec line-flux family for line detection and direct-method "
            "fits (FLUX Gaussian or BOXFLUX boxcar). Required."
        ),
    )
    args = parser.parse_args(argv)

    import sfr_and_metallicity as sam
    sam.line_flux_type = args.line_flux_type

    plot_dir = os.path.join(STACK_PATH, "plots")

    # Lazy import: pn_functions builds PyNeb interpolation grids at import
    # time (~seconds), so only pay that cost when we actually fit.
    from pn_functions import compute_direct_metallicities

    pattern = os.path.join(STACK_PATH, INPUT_GLOB)
    files = sorted(glob.glob(pattern))
    # Skip any accidental double-processed outputs.
    files = [f for f in files if "fastspec_fastspec_" not in os.path.basename(f)]
    print(f"[1] Found {len(files)} FastSpecFit stack outputs in {STACK_PATH}")
    if len(files) == 0:
        print(f"    Nothing matches {pattern}; exiting.")
        return 1

    rows = []

    for fpath in files:
        parsed = parse_bin_from_filename(fpath)
        if parsed is None:
            print(f"  ! Skipping unparsable filename: {os.path.basename(fpath)}")
            continue
        mlo, mhi, token, ew_min, ew_max = parsed
        nobj = read_nobj_from_input_stack(mlo, mhi, token)

        print(f"\n--- log M*=[{mlo:.2f},{mhi:.2f}] | {token} "
              f"(EW in ({ew_min:.0f},{ew_max:.0f}]) | NOBJ={nobj} ---")

        try:
            t = Table.read(fpath, hdu=FASTSPEC_HDU)
        except Exception as e:
            print(f"    Could not read hdu={FASTSPEC_HDU}: {e}; skipping.")
            continue

        if len(t) == 0:
            print("    Empty fastspec table; skipping.")
            continue

        # Gate the bin on the MEAN stack (row 0).
        detected = bool(line_snr_mask(
            t[[0]], line_names=TE_LINE_NAMES,
            snr_val=SNR_VAL, min_lines=MIN_LINES, min_flux=MIN_FLUX,
            line_flux_type=args.line_flux_type,
        )[0])

        # Base record (filled with NaNs; populated below if detected+fit).
        rec = {
            "MSTAR_MIN": mlo, "MSTAR_MAX": mhi,
            "EW_MIN": ew_min,
            # +inf is not FITS/ECSV-friendly; store -1 as the "open top" sentinel.
            "EW_MAX": ew_max if np.isfinite(ew_max) else -1.0,
            "EW_TOKEN": token,
            "NOBJ": nobj,
            "DETECTED_7LINE": detected,
            "MEAN_FIT_SUCCESS": False,
            "N_BOOT_FIT": 0,
            "N_RATIOS": 0,
            "LOGZ": np.nan,
        }
        for _, up in PARAM_MAP:
            rec[up] = np.nan
            rec[f"{up}_ERR"] = np.nan
            rec[f"{up}_ERR_LO"] = np.nan
            rec[f"{up}_ERR_HI"] = np.nan
            rec[f"{up}_MEANFIT_LO"] = np.nan
            rec[f"{up}_MEANFIT_HI"] = np.nan

        if not detected:
            print(f"    Mean stack does NOT detect all {MIN_LINES} lines "
                  f"(SNR>{SNR_VAL}, flux>{MIN_FLUX:g}); recording as undetected.")
            rows.append(rec)
            continue

        # Fit the mean stack (row 0) + all bootstrap rows (1..N).
        print(f"    Detected. Fitting {len(t)} rows "
              f"(1 mean + {len(t) - 1} bootstraps) with UltraNest ...")
        res = compute_direct_metallicities(
            t,
            args.line_flux_type,
            n_jobs=N_JOBS,
            min_num_live_points=MIN_NUM_LIVE_POINTS,
            sampler_kwargs=SAMPLER_KWARGS,
            use_informative_priors=USE_INFORMATIVE_PRIORS,
            verbose=False,
            verbose_sampler=False,
        )

        mean_fit = res[0]
        rec["MEAN_FIT_SUCCESS"] = bool(mean_fit["fit_success"])
        rec["N_RATIOS"] = int(mean_fit["n_ratios"])
        rec["LOGZ"] = float(mean_fit["logz"])

        # Bootstrap rows used for the spread: successful fits with finite
        # 12+log(OH).
        if len(res) > 1:
            boot = res[1:]
            ok = np.asarray(boot["fit_success"], dtype=bool) & np.isfinite(
                np.asarray(boot["twelve_log_OH"], dtype=float)
            )
            boot_ok = boot[ok]
        else:
            boot_ok = res[:0]
        rec["N_BOOT_FIT"] = int(len(boot_ok))

        for lo_name, up in PARAM_MAP:
            rec[up] = float(mean_fit[lo_name])
            rec[f"{up}_MEANFIT_LO"] = float(mean_fit[f"{lo_name}_lo"])
            rec[f"{up}_MEANFIT_HI"] = float(mean_fit[f"{lo_name}_hi"])
            if len(boot_ok) > 0:
                err, err_lo, err_hi, _ = boot_spread(boot_ok[lo_name])
                rec[f"{up}_ERR"] = err
                rec[f"{up}_ERR_LO"] = err_lo
                rec[f"{up}_ERR_HI"] = err_hi

        print(f"    -> 12+log(OH) = {rec['TWELVE_LOG_OH']:.3f} "
              f"(+{rec['TWELVE_LOG_OH_ERR_HI']:.3f}/-{rec['TWELVE_LOG_OH_ERR_LO']:.3f}), "
              f"A_V = {rec['AV']:.3f} "
              f"(+{rec['AV_ERR_HI']:.3f}/-{rec['AV_ERR_LO']:.3f}); "
              f"N_boot_fit={rec['N_BOOT_FIT']}")
        rows.append(rec)

    if len(rows) == 0:
        print("\n[2] No bins processed; nothing to write.")
        return 1

    # -------------------------------------------------------------------------
    # Assemble + write the results table
    # -------------------------------------------------------------------------
    out_tab = Table(rows)
    # Stable, readable column ordering.
    col_order = [
        "MSTAR_MIN", "MSTAR_MAX", "EW_MIN", "EW_MAX", "EW_TOKEN", "NOBJ",
        "DETECTED_7LINE", "MEAN_FIT_SUCCESS", "N_BOOT_FIT", "N_RATIOS", "LOGZ",
    ]
    for _, up in PARAM_MAP:
        col_order += [up, f"{up}_ERR", f"{up}_ERR_LO", f"{up}_ERR_HI",
                      f"{up}_MEANFIT_LO", f"{up}_MEANFIT_HI"]
    out_tab = out_tab[[c for c in col_order if c in out_tab.colnames]]

    out_fits = os.path.join(STACK_PATH, f"{OUT_BASENAME}.fits")
    out_ecsv = os.path.join(STACK_PATH, f"{OUT_BASENAME}.ecsv")
    out_tab.write(out_fits, overwrite=True)
    out_tab.write(out_ecsv, format="ascii.ecsv", overwrite=True)

    n_det = int(np.sum(out_tab["DETECTED_7LINE"]))
    n_fit = int(np.sum(out_tab["MEAN_FIT_SUCCESS"]))
    print(f"\n[2] Wrote {len(out_tab)} bin rows "
          f"({n_det} detected, {n_fit} with successful mean-stack fit):")
    print(f"    {out_fits}")
    print(f"    {out_ecsv}")

    # -------------------------------------------------------------------------
    # Quick sanity figure
    # -------------------------------------------------------------------------
    try:
        make_sanity_plot(out_tab, plot_dir)
    except Exception as e:
        print(f"    (sanity plot skipped: {e})")

    print("\n[3] Done.")
    return 0


def make_sanity_plot(out_tab, plot_dir):
    """12+log(OH) and A_V vs M* (bin midpoint), one series per EW bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    good = out_tab[(out_tab["DETECTED_7LINE"]) & (out_tab["MEAN_FIT_SUCCESS"])]
    if len(good) == 0:
        print("    (no successful fits; skipping sanity plot)")
        return

    mid = 0.5 * (np.asarray(good["MSTAR_MIN"]) + np.asarray(good["MSTAR_MAX"]))
    colors = {"ew_lt30": "#1f77b4", "ew_30_300": "#ff7f0e", "ew_gt300": "#d62728"}
    labels = {"ew_lt30": r"EW $\leq$ 30",
              "ew_30_300": r"30 < EW $\leq$ 300",
              "ew_gt300": "EW > 300"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (col, ax, ylab) in [
        ("TWELVE_LOG_OH", axes[0], "12 + log(O/H)"),
        ("AV", axes[1], r"$A_V$ [mag]"),
    ]:
        for token in EW_TOKENS:
            sel = np.asarray(good["EW_TOKEN"]) == token
            if not np.any(sel):
                continue
            ax.errorbar(
                mid[sel], np.asarray(good[col])[sel],
                yerr=[np.asarray(good[f"{col}_ERR_LO"])[sel],
                      np.asarray(good[f"{col}_ERR_HI"])[sel]],
                fmt="o-", color=colors[token], label=labels[token],
                capsize=3, lw=1.2,
            )
        ax.set_xlabel(r"log $M_\star$ [$M_\odot$]")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Direct-method stacks: trends vs stellar mass")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(plot_dir, "oh_av_vs_mstar.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


if __name__ == "__main__":
    sys.exit(main())
