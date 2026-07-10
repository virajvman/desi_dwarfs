'''
Scan the number of NNMF templates and measure reconstruction quality.

For each n_templates in a grid, this script:
  1. Trains NNMF templates on the production *training* half (IS_VALIDATION==0)
     using the exact warm-start + joint-optimize recipe from nnmf_analysis.py.
  2. Fits those templates (non-negative least squares) to the held-out
     *validation* half (IS_VALIDATION==1) and, separately, to the training half.
  3. Computes a per-object reduced chi^2,
         chi2_obj   = sum_pix ivar * (flux - fit)^2          (= rnorm^2 from nnls)
         redchi2    = chi2_obj / (N_good_pix - n_templates)
     and summarizes its distribution (median, 16/84, tail percentiles, ...).

Because nearly-NMF is NOT nested (the k-template solution is not the first k of
the (k+1)-template solution), every grid point is trained from scratch. The
held-out validation curve is the quantity to watch: training reconstruction
error falls monotonically with more templates, so only the validation curve
reveals where adding templates stops buying real (shared) spectral structure.

Inputs (reused from the production run, so the split matches exactly):
    desi_dr1_dwarf_catalog_nnmf_<flag>.h5  ->  FLUX_NORM, FLUX_IVAR_NORM,
                                               IS_VALIDATION, WAVE_REST, TARGETID

Outputs (in RESULTS_DIR):
    templates_ntemp{n}.npy            trained templates for each n  (n_pix, n)
    redchi2_valid_ntemp{n}.npy        per-object reduced chi^2, validation set
    redchi2_train_ntemp{n}.npy        per-object reduced chi^2, training set
    ntemplate_scan_summary.csv        one row per n with all summary stats
    ntemplate_scan_summary.npz        same stats, numpy-friendly
    ntemplate_scan.pdf                diagnostic plots
'''

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import time
from multiprocessing import Pool
from scipy.optimize import nnls
from astropy.table import Table
import cupy as cp
from nearly_nmf import nmf

from desi_lowz_funcs import print_stage

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
FLAG = "NEW"

# Extended grid. n=1 is the pure-average baseline: after NMF-1 the residual PCA
# is ~standard PCA on mean-subtracted flux, so the n_nmf=1 edge of the
# combined (n_nmf x n_pca) grid is the "pure PCA" reference. Upper end raised to
# 20 so the diminishing-returns plateau is actually visible past the production
# choice of 10.
N_TEMPLATE_GRID = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Training recipe -- kept identical to nnmf_analysis.py so the scan is faithful.
N_ITER_WARM = 50      # per-template warm-start iterations (sequential build-up)
N_ITER_JOINT = 1000   # final joint optimization over all templates

SEED = 42
N_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", 128))  # for the nnls fits

SPEC_DIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files"
NNMF_H5 = f"{SPEC_DIR}/desi_dr1_dwarf_catalog_nnmf_{FLAG}.h5"

RESULTS_DIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/ntemplate_scan"

# Reduced-chi^2 thresholds used to report an "outlier fraction" (poorly-fit objects)
OUTLIER_THRESHOLDS = [2.0, 5.0]

# Percentiles to record for the reduced-chi^2 distribution
PERCENTILES = [2.5, 16, 50, 84, 90, 97.5, 99]

# Emission lines for the line-window chi^2 (VACUUM wavelengths in Angstrom, since
# DESI works in vacuum). Standard dwarf set: [OII] doublet, Hbeta, [OIII] doublet,
# Halpha, [SII] doublet. chi^2 is summed over +/- LINE_HALFWIDTH around each.
LINE_CENTERS = [3727.09, 3729.88,   # [OII]
                4862.68,            # Hbeta
                4960.30, 5008.24,   # [OIII]
                6564.61,            # Halpha
                6718.29, 6732.68]   # [SII]
LINE_HALFWIDTH = 10.0   # Angstrom, rest-frame

# Fraction of objects (by per-pixel SNR) kept for the "high-SNR subset" metric.
# The figures use the MEAN reduced chi^2 of this subset (top 10%). Kept in sync
# with scan_nnmf_pca_grid.py. (Panel 1's hi-S/N curve is actually read from the
# combined grid's n_pca=0 slice, so a Stage-1 rerun is not required to change
# this fraction -- but keep them equal so the CSV columns stay consistent.)
HISNR_TOP_FRACTION = 0.10


# ----------------------------------------------------------------------------
# NNMF training (mirrors nnmf_analysis.py PART 2)
# ----------------------------------------------------------------------------
def train_nnmf_templates(flux_cp, ivar_cp, n_templates, rng):
    """Train `n_templates` nearly-NMF templates on GPU-resident flux/ivar.

    flux_cp, ivar_cp : cupy arrays of shape (n_pix, n_train).
    Returns the templates W as a numpy array of shape (n_pix, n_templates).
    """
    n_pix, n_train = flux_cp.shape

    H_start = cp.array(rng.uniform(0, 1, (n_templates, n_train)))
    W_start = cp.array(np.ones((n_pix, n_templates)))

    H_nearly = cp.array(H_start, copy=True)
    W_nearly = cp.array(W_start, copy=True)

    # Sequential warm-start: bring template i up to speed before adding i+1.
    for i in range(n_templates):
        H_itr, W_itr = nmf.nearly_NMF(
            flux_cp, ivar_cp,
            H_nearly[:(i + 1), :], W_nearly[:, :(i + 1)],
            n_iter=N_ITER_WARM,
        )
        H_nearly[:(i + 1), :] = H_itr
        W_nearly[:, :(i + 1)] = W_itr

    # Joint optimization over all templates together.
    H_nearly, W_nearly, _ = nmf.nearly_NMF(
        flux_cp, ivar_cp, H_nearly, W_nearly,
        n_iter=N_ITER_JOINT, return_chi_2=True,
    )

    return cp.asnumpy(W_nearly)


# ----------------------------------------------------------------------------
# NNLS fitting of templates to spectra (mirrors nnmf_analysis.py PART 3)
# ----------------------------------------------------------------------------
# Module-level globals populated per-worker via the Pool initializer, so the
# (large) flux/ivar/template arrays are not pickled on every task.
_W = None         # templates (n_pix, n_templates)
_FLUX = None      # (n_pix, n_obj)
_IVAR = None      # (n_pix, n_obj)
_LINE_MASK = None # (n_pix,) bool: pixels inside emission-line windows


def _init_fit_worker(W, flux, ivar, line_mask):
    global _W, _FLUX, _IVAR, _LINE_MASK
    _W, _FLUX, _IVAR, _LINE_MASK = W, flux, ivar, line_mask


def _fit_one(i):
    """Fit object i (min_x>=0 ||sqrt(ivar)(W x - flux)||^2) and return
    (global chi^2, line-window chi^2, nnls coefficients)."""
    sqrt_ivar = np.sqrt(_IVAR[:, i])
    A = sqrt_ivar[:, None] * _W
    b = sqrt_ivar * _FLUX[:, i]
    coeffs, rnorm = nnls(A, b)
    chi2_global = rnorm * rnorm                       # = sum ivar*(flux-fit)^2
    fit = _W @ coeffs
    chi2_pix = _IVAR[:, i] * (_FLUX[:, i] - fit) ** 2
    chi2_lines = float(chi2_pix[_LINE_MASK].sum())
    return chi2_global, chi2_lines, coeffs


def fit_chi2(W, flux, ivar, line_mask, n_proc):
    """Per-object global chi^2, line-window chi^2, and NNLS coefficients for all
    columns of flux. Coefficients are returned as (n_templates, n_obj) so the
    combined-grid scan can rebuild the NMF model (W @ H) and its residual
    without repeating the (expensive) non-negative fit."""
    n_obj = flux.shape[1]
    with Pool(processes=n_proc, initializer=_init_fit_worker,
              initargs=(W, flux, ivar, line_mask)) as pool:
        out = list(pool.imap(_fit_one, range(n_obj), chunksize=256))
    chi2_global = np.array([o[0] for o in out], dtype="f8")
    chi2_lines = np.array([o[1] for o in out], dtype="f8")
    coeffs = np.array([o[2] for o in out], dtype="f4").T   # (n_templates, n_obj)
    return chi2_global, chi2_lines, coeffs


def build_line_mask(wave_rest, centers, halfwidth):
    """Boolean mask of pixels within +/- halfwidth of any line center."""
    mask = np.zeros(wave_rest.shape, dtype=bool)
    for c in centers:
        mask |= np.abs(wave_rest - c) <= halfwidth
    return mask


def reduced_chi2(chi2, n_good, n_templates):
    """chi^2 per degree of freedom; dof = (good pixels) - (n templates)."""
    dof = np.maximum(n_good - n_templates, 1)
    return chi2 / dof


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print_stage("Loading normalized spectra + production split")
    with h5py.File(NNMF_H5, "r") as f:
        wave_rest = f["WAVE_REST"][:]
        flux = f["FLUX_NORM"][:]            # (n_pix, n_gal)
        ivar = f["FLUX_IVAR_NORM"][:]       # (n_pix, n_gal)
        is_valid = f["IS_VALIDATION"][:].astype(bool)

    print(f"flux shape       = {flux.shape}")
    print(f"ivar shape       = {ivar.shape}")
    print(f"n train (==0)    = {int((~is_valid).sum())}")
    print(f"n valid (==1)    = {int(is_valid.sum())}")

    train_mask = ~is_valid
    valid_mask = is_valid

    flux_train = np.ascontiguousarray(flux[:, train_mask])
    ivar_train = np.ascontiguousarray(ivar[:, train_mask])
    flux_valid = np.ascontiguousarray(flux[:, valid_mask])
    ivar_valid = np.ascontiguousarray(ivar[:, valid_mask])

    # Good-pixel counts per object (ivar > 0), for the dof in reduced chi^2.
    ngood_train = (ivar_train > 0).sum(axis=0)
    ngood_valid = (ivar_valid > 0).sum(axis=0)

    # Emission-line window mask + good line-pixel counts (normalize the line chi^2).
    line_mask = build_line_mask(wave_rest, LINE_CENTERS, LINE_HALFWIDTH)
    print(f"line-window pixels: {int(line_mask.sum())} of {line_mask.size}")
    ngood_line_train = ((ivar_train > 0) & line_mask[:, None]).sum(axis=0)
    ngood_line_valid = ((ivar_valid > 0) & line_mask[:, None]).sum(axis=0)

    # Per-object SNR proxy (median per-pixel S/N over good pixels) -> high-SNR cut.
    snr_pix = flux_valid * np.sqrt(ivar_valid)
    snr_pix[ivar_valid <= 0] = np.nan
    snr_valid = np.nanmedian(snr_pix, axis=0)
    del snr_pix
    hi_mask = snr_valid >= np.nanpercentile(snr_valid, 100 * (1 - HISNR_TOP_FRACTION))
    print(f"high-SNR subset: {int(hi_mask.sum())} objects "
          f"(top {HISNR_TOP_FRACTION:.0%} by median S/N)")

    # Move the training data to the GPU once and reuse across the whole grid.
    print_stage("Moving training data to GPU")
    flux_train_cp = cp.array(flux_train)
    ivar_train_cp = cp.array(ivar_train)

    rows = []                 # summary table rows
    per_object = {}           # full reduced-chi^2 arrays, saved to npz

    for n_templates in N_TEMPLATE_GRID:
        print_stage(f"n_templates = {n_templates}")

        t0 = time.time()
        W = train_nnmf_templates(flux_train_cp, ivar_train_cp, n_templates, rng)
        t_train = time.time() - t0
        np.save(f"{RESULTS_DIR}/templates_ntemp{n_templates}.npy", W)
        print(f"  trained in {t_train:.1f} s -> templates {W.shape}")

        # Fit to validation and training sets.
        t0 = time.time()
        chi2_valid, chi2line_valid, H_valid = fit_chi2(W, flux_valid, ivar_valid, line_mask, N_PROC)
        chi2_train, chi2line_train, H_train = fit_chi2(W, flux_train, ivar_train, line_mask, N_PROC)
        t_fit = time.time() - t0
        print(f"  fit (valid+train) in {t_fit:.1f} s")

        # Save the NMF coefficients so the combined (n_nmf x n_pca) grid scan
        # (scan_nnmf_pca_grid.py) can reconstruct the NMF model W @ H and its
        # residual without re-running the non-negative least squares fit.
        np.save(f"{RESULTS_DIR}/hcoeffs_valid_ntemp{n_templates}.npy", H_valid)
        np.save(f"{RESULTS_DIR}/hcoeffs_train_ntemp{n_templates}.npy", H_train)

        rc_valid = reduced_chi2(chi2_valid, ngood_valid, n_templates)
        rc_train = reduced_chi2(chi2_train, ngood_train, n_templates)
        # Line-region chi^2 per good line-pixel (model already fixed, so no -k).
        rcline_valid = chi2line_valid / np.maximum(ngood_line_valid, 1)
        rcline_train = chi2line_train / np.maximum(ngood_line_train, 1)

        for tag, arr in (("redchi2_valid", rc_valid), ("redchi2_train", rc_train),
                         ("redchi2line_valid", rcline_valid),
                         ("redchi2line_train", rcline_train)):
            per_object[f"{tag}_ntemp{n_templates}"] = arr.astype("f4")
            np.save(f"{RESULTS_DIR}/{tag}_ntemp{n_templates}.npy", arr.astype("f4"))

        # (name, reduced-chi^2 array, raw chi^2 array for the total) for each metric.
        metrics = [
            ("valid",       rc_valid,            chi2_valid),
            ("train",       rc_train,            chi2_train),
            ("valid_line",  rcline_valid,        chi2line_valid),
            ("train_line",  rcline_train,        chi2line_train),
            ("valid_hisnr", rc_valid[hi_mask],   chi2_valid[hi_mask]),
        ]
        row = {"n_templates": n_templates,
               "train_seconds": t_train, "fit_seconds": t_fit}
        for name, rc, chi2 in metrics:
            pcts = np.percentile(rc, PERCENTILES)
            for p, v in zip(PERCENTILES, pcts):
                row[f"{name}_p{p}"] = v
            row[f"{name}_mean"] = float(np.mean(rc))
            row[f"{name}_total_chi2"] = float(np.sum(chi2))   # un-reduced catalog sum
            for thr in OUTLIER_THRESHOLDS:
                row[f"{name}_frac_gt{thr}"] = float(np.mean(rc > thr))
        rows.append(row)

        print(f"  VALID global redchi2: median={row['valid_p50']:.3f}  "
              f"[16,84]=[{row['valid_p16']:.3f}, {row['valid_p84']:.3f}]  "
              f"p90={row['valid_p90']:.3f}  p99={row['valid_p99']:.3f}  "
              f"frac>2={row['valid_frac_gt2.0']:.4f}")
        print(f"  VALID line   redchi2: median={row['valid_line_p50']:.3f}  "
              f"p90={row['valid_line_p90']:.3f}  p99={row['valid_line_p99']:.3f}")
        print(f"  VALID hi-SNR redchi2: median={row['valid_hisnr_p50']:.3f}  "
              f"p90={row['valid_hisnr_p90']:.3f}  p99={row['valid_hisnr_p99']:.3f}")

    # ------------------------------------------------------------------
    # Save summary table
    # ------------------------------------------------------------------
    print_stage("Saving summary")
    tab = Table(rows)
    tab.write(f"{RESULTS_DIR}/ntemplate_scan_summary.csv", overwrite=True)
    np.savez(f"{RESULTS_DIR}/ntemplate_scan_summary.npz",
             **{c: np.array(tab[c]) for c in tab.colnames},
             **per_object)
    print(tab)

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    n_arr = np.array(tab["n_templates"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout="constrained")

    # Panel 1: global median + 16-84 band, validation vs training.
    ax = axes[0, 0]
    ax.fill_between(n_arr, tab["valid_p16"], tab["valid_p84"],
                    color="C0", alpha=0.25, label="validation 16-84%")
    ax.plot(n_arr, tab["valid_p50"], "-o", color="C0", label="validation median")
    ax.plot(n_arr, tab["train_p50"], "--s", color="C3", label="training median")
    ax.axhline(1.0, color="k", ls=":", lw=1, label=r"$\chi^2_\nu = 1$")
    ax.set_xlabel("number of NNMF templates")
    ax.set_ylabel(r"reduced $\chi^2$ (per object)")
    ax.set_title("Global reconstruction quality")
    ax.legend()

    # Panel 2: global tail (validation).
    ax = axes[0, 1]
    ax.plot(n_arr, tab["valid_p90"], "-o", label="90th pct")
    ax.plot(n_arr, tab["valid_p97.5"], "-s", label="97.5th pct")
    ax.plot(n_arr, tab["valid_p99"], "-^", label="99th pct")
    ax.set_xlabel("number of NNMF templates")
    ax.set_ylabel(r"reduced $\chi^2$ (validation tail)")
    ax.set_title("Global poorly-fit tail")
    ax.legend()

    # Panel 3: emission-line-window chi^2 (validation), median + band + tail.
    ax = axes[1, 0]
    ax.fill_between(n_arr, tab["valid_line_p16"], tab["valid_line_p84"],
                    color="C2", alpha=0.25, label="validation 16-84%")
    ax.plot(n_arr, tab["valid_line_p50"], "-o", color="C2", label="median")
    ax.plot(n_arr, tab["valid_line_p90"], "-^", color="C2", alpha=0.6, label="90th pct")
    ax.plot(n_arr, tab["valid_line_p99"], ":d", color="C2", alpha=0.6, label="99th pct")
    ax.set_xlabel("number of NNMF templates")
    ax.set_ylabel(r"line-window $\chi^2$ / line pixel")
    ax.set_title("Emission-line region fit quality")
    ax.legend()

    # Panel 4: high-SNR subset vs all objects (validation), median + tail.
    ax = axes[1, 1]
    ax.plot(n_arr, tab["valid_p50"], "--o", color="0.5", label="all: median")
    ax.plot(n_arr, tab["valid_hisnr_p50"], "-o", color="C1", label="hi-SNR: median")
    ax.plot(n_arr, tab["valid_hisnr_p90"], "-^", color="C1", alpha=0.7, label="hi-SNR: 90th")
    ax.plot(n_arr, tab["valid_hisnr_p99"], ":d", color="C1", alpha=0.7, label="hi-SNR: 99th")
    ax.set_xlabel("number of NNMF templates")
    ax.set_ylabel(r"reduced $\chi^2$ (per object)")
    ax.set_title(f"High-SNR subset (top {HISNR_TOP_FRACTION:.0%})")
    ax.legend()

    plt.savefig(f"{RESULTS_DIR}/ntemplate_scan.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nDone. Results in {RESULTS_DIR}")
