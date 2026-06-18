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

N_TEMPLATE_GRID = [2, 4, 6, 8, 10, 12, 14]

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
_W = None        # templates (n_pix, n_templates)
_FLUX = None     # (n_pix, n_obj)
_IVAR = None     # (n_pix, n_obj)


def _init_fit_worker(W, flux, ivar):
    global _W, _FLUX, _IVAR
    _W, _FLUX, _IVAR = W, flux, ivar


def _fit_one(i):
    """Return rnorm^2 (= chi^2) for object i: min_x>=0 ||sqrt(ivar)(W x - flux)||^2."""
    sqrt_ivar = np.sqrt(_IVAR[:, i])
    A = sqrt_ivar[:, None] * _W
    b = sqrt_ivar * _FLUX[:, i]
    _, rnorm = nnls(A, b)
    return rnorm * rnorm


def fit_chi2(W, flux, ivar, n_proc):
    """Per-object chi^2 (= sum ivar*(flux-fit)^2) for all columns of flux."""
    n_obj = flux.shape[1]
    with Pool(processes=n_proc, initializer=_init_fit_worker,
              initargs=(W, flux, ivar)) as pool:
        chi2 = list(pool.imap(_fit_one, range(n_obj), chunksize=256))
    return np.asarray(chi2)


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
        chi2_valid = fit_chi2(W, flux_valid, ivar_valid, N_PROC)
        chi2_train = fit_chi2(W, flux_train, ivar_train, N_PROC)
        t_fit = time.time() - t0
        print(f"  fit (valid+train) in {t_fit:.1f} s")

        rc_valid = reduced_chi2(chi2_valid, ngood_valid, n_templates)
        rc_train = reduced_chi2(chi2_train, ngood_train, n_templates)

        per_object[f"redchi2_valid_ntemp{n_templates}"] = rc_valid.astype("f4")
        per_object[f"redchi2_train_ntemp{n_templates}"] = rc_train.astype("f4")
        np.save(f"{RESULTS_DIR}/redchi2_valid_ntemp{n_templates}.npy", rc_valid.astype("f4"))
        np.save(f"{RESULTS_DIR}/redchi2_train_ntemp{n_templates}.npy", rc_train.astype("f4"))

        row = {"n_templates": n_templates,
               "train_seconds": t_train, "fit_seconds": t_fit}
        for name, rc in (("valid", rc_valid), ("train", rc_train)):
            pcts = np.percentile(rc, PERCENTILES)
            for p, v in zip(PERCENTILES, pcts):
                row[f"{name}_p{p}"] = v
            row[f"{name}_mean"] = float(np.mean(rc))
            # total catalog chi^2 (un-reduced) -- the sum the user mentioned
            row[f"{name}_total_chi2"] = float(np.sum(chi2_valid if name == "valid" else chi2_train))
            for thr in OUTLIER_THRESHOLDS:
                row[f"{name}_frac_gt{thr}"] = float(np.mean(rc > thr))
        rows.append(row)

        med = row["valid_p50"]
        lo, hi = row["valid_p16"], row["valid_p84"]
        print(f"  VALID reduced chi2: median={med:.3f}  [16,84]=[{lo:.3f}, {hi:.3f}]")

    # ------------------------------------------------------------------
    # Save summary table
    # ------------------------------------------------------------------
    print_stage("Saving summary")
    tab = Table(rows)
    tab.write(f"{RESULTS_DIR}/ntemplate_scan_summary.csv", overwrite=True)
    np.savez(f"{RESULTS_DIR}/ntemplate_scan_summary.npz",
             n_templates=np.array(N_TEMPLATE_GRID),
             **{c: np.array(tab[c]) for c in tab.colnames},
             **per_object)
    print(tab)

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    n_arr = np.array(tab["n_templates"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")

    # Panel 1: median + 16-84 band, validation vs training.
    ax = axes[0]
    ax.fill_between(n_arr, tab["valid_p16"], tab["valid_p84"],
                    color="C0", alpha=0.25, label="validation 16-84%")
    ax.plot(n_arr, tab["valid_p50"], "-o", color="C0", label="validation median")
    ax.plot(n_arr, tab["train_p50"], "--s", color="C3", label="training median")
    ax.axhline(1.0, color="k", ls=":", lw=1, label=r"$\chi^2_\nu = 1$")
    ax.set_xlabel("number of NNMF templates")
    ax.set_ylabel(r"reduced $\chi^2$ (per object)")
    ax.set_title("Reconstruction quality vs. template count")
    ax.legend()

    # Panel 2: tail of the distribution + outlier fractions (validation).
    ax = axes[1]
    ax.plot(n_arr, tab["valid_p90"], "-o", label="90th pct")
    ax.plot(n_arr, tab["valid_p97.5"], "-s", label="97.5th pct")
    ax.plot(n_arr, tab["valid_p99"], "-^", label="99th pct")
    ax.set_xlabel("number of NNMF templates")
    ax.set_ylabel(r"reduced $\chi^2$ (validation tail)")
    ax.set_title("Poorly-fit tail vs. template count")
    ax.legend()

    plt.savefig(f"{RESULTS_DIR}/ntemplate_scan.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nDone. Results in {RESULTS_DIR}")
