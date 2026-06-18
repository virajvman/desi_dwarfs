'''
Scan the number of PCA components used on the noise-normalized NNMF residuals,
and quantify how many components are actually worth keeping.

Unlike the NNMF scan, PCA is *nested*: a single SVD of the training residuals
yields every truncation k at once, so there is no grid of expensive retrains.
The work is all in how we judge k:

  1. Scree / explained variance.  From the singular values S of the centered
     training residuals, explained-variance ratio_i = S_i^2 / sum(S^2), and its
     cumulative sum.  (spectra_anomaly_plots.py throws S away -- here we keep it.)

  2. Held-out reconstruction.  Fit PCA on the training half (IS_VALIDATION==0),
     project the validation half (==1) onto the training components, and measure
     how much validation variance the top-k components capture, plus the
     reconstruction MSE.  Training explained-variance falls forever as k grows;
     the validation curve plateaus once the components stop describing shared
     structure -- that plateau / elbow is the answer.

  3. Noise floor (parallel analysis).  The PCA input is residual/noise, so a
     perfect model + Gaussian noise would give entries ~ unit variance and an
     eigenvalue spectrum set purely by random-matrix statistics.  We build a null
     matrix with the SAME per-pixel variance but no cross-pixel correlation
     (column-wise Gaussian), SVD it, and overlay its singular-value spectrum.
     Real components whose singular values rise above this null floor encode
     genuine correlated structure; the rest are fitting noise.  The crossing
     point is a threshold-free estimate of the optimal k.

Inputs:
    norm_residuals_dwarfs_<flag>.npy            (n_gal, n_pix) noise-normalized residuals
    desi_dr1_dwarf_catalog_nnmf_<flag>.h5       IS_VALIDATION (same object order)

Outputs (in RESULTS_DIR):
    pca_scan_summary.csv / .npz      per-k cumulative explained variance (train &
                                     valid), reconstruction MSE, # comps above null
    pca_scan_per_component.npz       full per-component arrays (S, ratios, null)
    pca_scan.pdf                     scree + cumulative + reconstruction diagnostics
'''

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import torch
from astropy.table import Table

from desi_lowz_funcs import print_stage

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
FLAG = "NEW"

# Largest component count to characterize. PCA is nested, so this single number
# sets the whole scan -- everything <= K_MAX comes for free from one SVD.
K_MAX = 25

# Checkpoints written to the summary table (cumulative quantities at these k).
K_GRID = [1, 2, 5, 10, 15, 20, 25]

SEED = 42

SPEC_DIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files"
RESID_NPY = f"{SPEC_DIR}/norm_residuals_dwarfs_{FLAG}.npy"
NNMF_H5 = f"{SPEC_DIR}/desi_dr1_dwarf_catalog_nnmf_{FLAG}.h5"

RESULTS_DIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/ncomponent_scan"


def svd_components(Z):
    """Return (S, Vt) of a centered matrix Z (n, p) via reduced SVD on `device`."""
    _, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    return S, Vh


if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    print_stage("Loading noise-normalized residuals + production split")
    resids = np.load(RESID_NPY)                     # (n_gal, n_pix)
    resids = np.nan_to_num(resids, nan=0.0, posinf=0.0, neginf=0.0)
    with h5py.File(NNMF_H5, "r") as f:
        is_valid = f["IS_VALIDATION"][:].astype(bool)

    assert resids.shape[0] == is_valid.shape[0], \
        f"residual/IS_VALIDATION length mismatch: {resids.shape[0]} vs {is_valid.shape[0]}"

    n_gal, n_pix = resids.shape
    print(f"residuals shape = {resids.shape}")
    print(f"n train = {int((~is_valid).sum())},  n valid = {int(is_valid.sum())}")

    X_train = torch.tensor(resids[~is_valid], dtype=torch.float32, device=device)
    X_valid = torch.tensor(resids[is_valid], dtype=torch.float32, device=device)
    del resids

    # --- Center on the training mean (PCA convention; same mean for valid) ----
    mean_ = X_train.mean(0, keepdim=True)
    Z_train = X_train - mean_
    Z_valid = X_valid - mean_

    # ------------------------------------------------------------------
    # (1) SVD of training residuals  ->  scree / explained variance
    # ------------------------------------------------------------------
    print_stage("SVD of training residuals")
    S, Vt = svd_components(Z_train)                  # S: (r,), Vt: (r, n_pix)
    S2 = S.pow(2)
    total_var_train = float((Z_train.pow(2)).sum())
    evr_train = (S2 / total_var_train).cpu().numpy()         # per-component
    cum_evr_train = np.cumsum(evr_train)

    # ------------------------------------------------------------------
    # (2) Held-out projection  ->  validation explained variance + recon MSE
    # ------------------------------------------------------------------
    print_stage("Projecting validation residuals onto training components")
    T = Z_valid @ Vt.t()                             # (n_valid, r)
    valid_var_per_comp = (T.pow(2)).sum(0).cpu().numpy()
    total_var_valid = float((Z_valid.pow(2)).sum())
    evr_valid = valid_var_per_comp / total_var_valid
    cum_evr_valid = np.cumsum(evr_valid)

    n_valid = Z_valid.shape[0]
    # reconstruction MSE per element when keeping the top-k components
    resid_var_valid = total_var_valid - np.cumsum(valid_var_per_comp)
    recon_mse_valid = resid_var_valid / (n_valid * n_pix)

    # ------------------------------------------------------------------
    # (3) Noise floor: column-wise Gaussian null with matched per-pixel variance
    # ------------------------------------------------------------------
    print_stage("Building noise null (parallel analysis) and its SVD")
    col_std = Z_train.std(0, keepdim=True)           # per-pixel std (1, n_pix)
    Z_null = torch.randn(Z_train.shape, device=device, dtype=torch.float32) * col_std
    Z_null = Z_null - Z_null.mean(0, keepdim=True)
    S_null, _ = svd_components(Z_null)
    S_null2 = S_null.pow(2)

    S_np = S.cpu().numpy()
    S_null_np = S_null.cpu().numpy()
    # number of leading components whose singular value beats the null floor
    above_null = S_np > S_null_np
    # first index where real drops below null = estimated # of signal components
    first_below = int(np.argmax(~above_null)) if (~above_null).any() else len(S_np)
    print(f"Estimated # signal components (real S > null S): {first_below}")

    # ------------------------------------------------------------------
    # Summary table at the K_GRID checkpoints
    # ------------------------------------------------------------------
    print_stage("Saving summary")
    rows = []
    for k in K_GRID:
        if k > len(S_np):
            continue
        rows.append({
            "k": k,
            "cum_evr_train": float(cum_evr_train[k - 1]),
            "cum_evr_valid": float(cum_evr_valid[k - 1]),
            "recon_mse_valid": float(recon_mse_valid[k - 1]),
            "evr_train_k": float(evr_train[k - 1]),
            "evr_valid_k": float(evr_valid[k - 1]),
            "n_above_null": int(np.sum(above_null[:k])),
        })
    tab = Table(rows)
    tab.write(f"{RESULTS_DIR}/pca_scan_summary.csv", overwrite=True)
    print(tab)
    print(f"\nFirst component below the noise floor: k = {first_below}")

    np.savez(f"{RESULTS_DIR}/pca_scan_summary.npz",
             k=np.array([r["k"] for r in rows]),
             **{c: np.array(tab[c]) for c in tab.colnames},
             first_below_null=first_below)

    np.savez(f"{RESULTS_DIR}/pca_scan_per_component.npz",
             S=S_np, S_null=S_null_np,
             evr_train=evr_train, cum_evr_train=cum_evr_train,
             evr_valid=evr_valid, cum_evr_valid=cum_evr_valid,
             recon_mse_valid=recon_mse_valid,
             first_below_null=first_below)

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    kx = np.arange(1, K_MAX + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), layout="constrained")

    # Panel 1: scree (singular values), real train vs noise null.
    ax = axes[0]
    ax.plot(kx, S_np[:K_MAX], "-o", ms=4, label="training data")
    ax.plot(kx, S_null_np[:K_MAX], "--", color="gray", label="noise null")
    if first_below <= K_MAX:
        ax.axvline(first_below, color="C3", ls=":", label=f"floor crossing (k={first_below})")
    ax.set_xlabel("component index")
    ax.set_ylabel("singular value")
    ax.set_yscale("log")
    ax.set_title("Scree vs. noise floor")
    ax.legend()

    # Panel 2: per-component explained variance, training vs validation.
    ax = axes[1]
    ax.plot(kx, evr_train[:K_MAX], "-o", ms=4, label="training")
    ax.plot(kx, evr_valid[:K_MAX], "-s", ms=4, label="validation (held out)")
    ax.set_xlabel("component index")
    ax.set_ylabel("explained variance ratio (per component)")
    ax.set_yscale("log")
    ax.set_title("Per-component explained variance")
    ax.legend()

    # Panel 3: cumulative explained variance + held-out reconstruction MSE.
    ax = axes[2]
    ax.plot(kx, cum_evr_train[:K_MAX], "-o", ms=4, label="cum. EVR (train)")
    ax.plot(kx, cum_evr_valid[:K_MAX], "-s", ms=4, label="cum. EVR (valid)")
    for lvl in (0.90, 0.95, 0.99):
        ax.axhline(lvl, color="k", ls=":", lw=0.8)
    ax.set_xlabel("number of components k")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("Cumulative explained variance")
    ax.legend(loc="lower right")

    plt.savefig(f"{RESULTS_DIR}/pca_scan.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nDone. Results in {RESULTS_DIR}")
