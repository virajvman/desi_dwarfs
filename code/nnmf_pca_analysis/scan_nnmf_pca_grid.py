'''
Combined NNMF + residual-PCA reconstruction grid.

For every NNMF template count n_nmf produced by scan_nnmf_ntemplates.py, this
script sweeps the number of residual-PCA components n_pca and measures the
reconstruction quality of the *combined* model

        model(lambda) = sum_i a_i NMF_i(lambda)  +  sum_j b_j PCA_j(lambda)

against the held-out (IS_VALIDATION==1) spectra. The a_i are the NNMF
coefficients (from scan_nnmf_ntemplates.py); the b_j are the projections of the
noise-normalized NNMF residual onto the training-residual PCA components. The
output is a 2-D grid, over (n_nmf, n_pca), of three reduced-chi^2 summaries that
feed the hist2d panels in template_investigations.ipynb:

    global      mean over validation objects of the per-object reduced chi^2
    line        mean over objects of chi^2 per good line-window pixel
    hisnr_p99   99th-percentile (worst-fit tail) of the reduced chi^2 among the
                top-HISNR_TOP_FRACTION objects by median per-pixel S/N

Why this is cheap: PCA is nested, so one SVD of the training residuals at each
n_nmf yields *every* n_pca truncation at once. The only per-n_nmf work is one
SVD plus a projection; the (expensive) NNMF training and non-negative fit were
already done by scan_nnmf_ntemplates.py, and we reuse its saved templates and
coefficients (templates_ntemp{n}.npy, hcoeffs_{valid,train}_ntemp{n}.npy).

Reading the grid: the combined reduced chi^2 falls monotonically along BOTH axes
(more NNMF templates, more PCA components), and dips well below 1 because the
data are noise-dominated (see scan_nnmf_ntemplates.py). So the grid is read via
iso-chi^2 contours and diminishing returns, NOT by hunting a minimum. We also
record, per n_nmf, the parallel-analysis noise-floor crossing (first_below_null)
-- beyond it, extra PCA components are fitting (partly correlated) noise rather
than genuine shared structure, and that line should be overlaid on the PCA axis.

Conventions (kept deliberately simple; refine in the notebook if needed):
  * PCA is centered on the training-residual mean (production convention). The
    n_pca=0 column is therefore "NNMF + mean residual", which differs from the
    pure-NNMF chi^2 (Panel 1 / ntemplate_scan_summary.csv) only by the small
    mean-residual term. We save the pure-NNMF anchor (nmf_only_*) to verify this.
  * combined dof = (good pixels) - n_nmf - n_pca. The subtraction is negligible
    (tens out of ~4000 pixels) but kept for consistency with Panel 1.
  * Masked pixels (ivar<=0) are set to 0 in the noise-normalized residual, so
    they contribute 0 to both the SVD and the chi^2 -- matching
    scan_pca_ncomponents.py.

Inputs (from scan_nnmf_ntemplates.py RESULTS_DIR):
    templates_ntemp{n}.npy            (n_pix, n)      NNMF templates
    hcoeffs_valid_ntemp{n}.npy        (n, n_valid)    NNMF coeffs, validation
    hcoeffs_train_ntemp{n}.npy        (n, n_train)    NNMF coeffs, training
    ../spectra_files/desi_dr1_dwarf_catalog_nnmf_{FLAG}.h5
                                      FLUX_NORM, FLUX_IVAR_NORM, WAVE_REST,
                                      IS_VALIDATION

Outputs (in RESULTS_DIR):
    nnmf_pca_grid_summary.npz         all 2-D metric grids + axes + null crossing
    nnmf_pca_grid.pdf                 quick-look hist2d of the three metrics
'''

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import torch


def print_stage(line2print, ch='-', end_space=True):
    """Banner line. Inlined so this script needs no DESI stack and runs under
    the standalone NERSC `pytorch` module (see scan_pca_ncomponents.py)."""
    nl = len(line2print)
    print(ch * nl)
    print(line2print)
    print(ch * nl)
    if end_space:
        print(' ')


# ----------------------------------------------------------------------------
# Configuration -- MUST match scan_nnmf_ntemplates.py where they overlap.
# ----------------------------------------------------------------------------
FLAG = "NEW"

# NNMF template counts to read. Must be a subset of what scan_nnmf_ntemplates.py
# actually produced (it saves templates_ntemp{n}.npy + hcoeffs_*_ntemp{n}.npy).
N_TEMPLATE_GRID = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# PCA components swept per n_nmf. n_pca=0 is the (near) pure-NNMF baseline edge.
K_MAX = 25

SEED = 42

# Percentile of the per-object reduced chi^2 distribution reported for the
# high-S/N subset (the worst-fit tail -- where genuine structure the templates
# miss shows up; the mean/median of this subset is noise-floor flat).
HISNR_PERCENTILE = 99.0
HISNR_TOP_FRACTION = 0.25   # top fraction by median per-pixel S/N

# Emission-line windows for the line-region chi^2 (VACUUM Angstrom, +/- 10 A
# rest-frame). Identical set to scan_nnmf_ntemplates.py.
LINE_CENTERS = [3727.09, 3729.88,   # [OII]
                4862.68,            # Hbeta
                4960.30, 5008.24,   # [OIII]
                6564.61,            # Halpha
                6718.29, 6732.68]   # [SII]
LINE_HALFWIDTH = 10.0

SPEC_DIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files"
NNMF_H5 = f"{SPEC_DIR}/desi_dr1_dwarf_catalog_nnmf_{FLAG}.h5"

RESULTS_DIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/ntemplate_scan"


def build_line_mask(wave_rest, centers, halfwidth):
    mask = np.zeros(wave_rest.shape, dtype=bool)
    for c in centers:
        mask |= np.abs(wave_rest - c) <= halfwidth
    return mask


if __name__ == "__main__":

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ------------------------------------------------------------------
    # Load spectra + production split (same object order as the coeffs).
    # ------------------------------------------------------------------
    print_stage("Loading normalized spectra + production split")
    with h5py.File(NNMF_H5, "r") as f:
        wave_rest = f["WAVE_REST"][:]
        flux = f["FLUX_NORM"][:]            # (n_pix, n_gal)
        ivar = f["FLUX_IVAR_NORM"][:]       # (n_pix, n_gal)
        is_valid = f["IS_VALIDATION"][:].astype(bool)

    n_pix, n_gal = flux.shape
    train_mask = ~is_valid
    valid_mask = is_valid
    print(f"flux shape = {flux.shape};  n_train = {int(train_mask.sum())};  "
          f"n_valid = {int(valid_mask.sum())}")

    flux_train = np.ascontiguousarray(flux[:, train_mask])
    ivar_train = np.ascontiguousarray(ivar[:, train_mask])
    flux_valid = np.ascontiguousarray(flux[:, valid_mask])
    ivar_valid = np.ascontiguousarray(ivar[:, valid_mask])
    del flux, ivar

    good_train = ivar_train > 0
    good_valid = ivar_valid > 0
    ngood_valid = good_valid.sum(axis=0)                       # (n_valid,)

    # Emission-line window mask + good line-pixel counts per object.
    line_mask = build_line_mask(wave_rest, LINE_CENTERS, LINE_HALFWIDTH)
    n_line = int(line_mask.sum())
    print(f"line-window pixels: {n_line} of {line_mask.size}")
    ngood_line_valid = (good_valid & line_mask[:, None]).sum(axis=0)  # (n_valid,)

    # High-S/N subset (top fraction by median per-pixel S/N over good pixels),
    # exactly as in scan_nnmf_ntemplates.py.
    snr_pix = flux_valid * np.sqrt(ivar_valid)
    snr_pix[~good_valid] = np.nan
    snr_valid = np.nanmedian(snr_pix, axis=0)
    del snr_pix
    hi_mask = snr_valid >= np.nanpercentile(snr_valid, 100 * (1 - HISNR_TOP_FRACTION))
    print(f"high-S/N subset: {int(hi_mask.sum())} objects "
          f"(top {HISNR_TOP_FRACTION:.0%})")

    sqrt_ivar_train = np.sqrt(np.clip(ivar_train, 0, None))
    sqrt_ivar_valid = np.sqrt(np.clip(ivar_valid, 0, None))

    # Torch tensors reused across the grid.
    line_idx = torch.tensor(np.where(line_mask)[0], device=device)
    ngood_valid_t = torch.tensor(ngood_valid.astype("f4"), device=device)
    ngood_line_valid_t = torch.tensor(np.maximum(ngood_line_valid, 1).astype("f4"),
                                      device=device)
    hi_mask_t = torch.tensor(hi_mask, device=device)
    kx = np.arange(0, K_MAX + 1)                               # PCA axis, incl 0

    # Output grids, shape (n_nmf, K_MAX+1).
    G = len(N_TEMPLATE_GRID)
    grid_shape = (G, K_MAX + 1)
    out = {
        "global_redchi2_mean":   np.full(grid_shape, np.nan, "f4"),
        "global_redchi2_median": np.full(grid_shape, np.nan, "f4"),
        "line_chi2_mean":        np.full(grid_shape, np.nan, "f4"),
        "hisnr_redchi2_p99":     np.full(grid_shape, np.nan, "f4"),
        "hisnr_redchi2_mean":    np.full(grid_shape, np.nan, "f4"),
        "cum_evr_valid":         np.full(grid_shape, np.nan, "f4"),
    }
    first_below_null = np.zeros(G, dtype="i4")
    nmf_only_global_mean = np.full(G, np.nan, "f4")   # pure-NNMF anchor (Panel 1)

    for gi, n_nmf in enumerate(N_TEMPLATE_GRID):
        print_stage(f"n_nmf = {n_nmf}")

        W = np.load(f"{RESULTS_DIR}/templates_ntemp{n_nmf}.npy")            # (n_pix, n)
        H_valid = np.load(f"{RESULTS_DIR}/hcoeffs_valid_ntemp{n_nmf}.npy")  # (n, n_valid)
        H_train = np.load(f"{RESULTS_DIR}/hcoeffs_train_ntemp{n_nmf}.npy")  # (n, n_train)

        # Noise-normalized NNMF residuals; masked pixels -> 0. Shape (n_obj, n_pix).
        res_train = (sqrt_ivar_train * (flux_train - W @ H_train)).T.astype("f4")
        res_valid = (sqrt_ivar_valid * (flux_valid - W @ H_valid)).T.astype("f4")

        X_train = torch.tensor(res_train, device=device)
        X_valid = torch.tensor(res_valid, device=device)
        del res_train, res_valid

        # Pure-NNMF chi^2 anchor: sum of squared noise-normalized residual over
        # all pixels == Stage-1 chi2_global (masked pixels are 0). Reduced by dof.
        ss_valid = (X_valid ** 2).sum(1)                                   # (n_valid,)
        red_nmf_only = ss_valid / torch.clamp(ngood_valid_t - n_nmf, min=1)
        nmf_only_global_mean[gi] = float(red_nmf_only.mean().cpu())

        # --- PCA on training residuals (centered on training mean) ------------
        mean_ = X_train.mean(0, keepdim=True)
        Z_train = X_train - mean_
        Z_valid = X_valid - mean_
        del X_train, X_valid

        _, S, Vt = torch.linalg.svd(Z_train, full_matrices=False)          # Vt: (r, n_pix)
        r = S.shape[0]
        k_use = min(K_MAX, r)

        # --- Held-out projections + cumulative explained variance ------------
        T = Z_valid @ Vt[:k_use].t()                                       # (n_valid, k_use)
        T2 = T ** 2
        cumT2 = torch.cumsum(T2, dim=1)                                    # (n_valid, k_use)
        total_var_valid = float((Z_valid ** 2).sum())

        # Global combined chi^2 for k = 0..K_MAX (orthonormal identity):
        #   ||Z_i||^2 - sum_{j<=k} T_ij^2 .
        normsqZ = (Z_valid ** 2).sum(1, keepdim=True)                      # (n_valid,1)
        chi2_glob = torch.empty((Z_valid.shape[0], K_MAX + 1), device=device)
        chi2_glob[:, 0] = normsqZ[:, 0]
        if k_use > 0:
            chi2_glob[:, 1:k_use + 1] = normsqZ - cumT2
        if k_use < K_MAX:   # ran out of components: hold flat
            chi2_glob[:, k_use + 1:] = chi2_glob[:, k_use:k_use + 1]

        # Line-region combined chi^2 for k = 0..K_MAX (explicit reconstruction
        # over line pixels; the orthonormal identity does not restrict to a
        # pixel subset). Built incrementally to reuse the running reconstruction.
        Z_line = Z_valid[:, line_idx]                                      # (n_valid, n_line)
        Vt_line = Vt[:k_use][:, line_idx]                                  # (k_use, n_line)
        recon_line = torch.zeros_like(Z_line)
        chi2_line = torch.empty((Z_valid.shape[0], K_MAX + 1), device=device)
        chi2_line[:, 0] = (Z_line ** 2).sum(1)
        for k in range(1, K_MAX + 1):
            if k <= k_use:
                recon_line = recon_line + T[:, k - 1:k] * Vt_line[k - 1:k, :]
            chi2_line[:, k] = ((Z_line - recon_line) ** 2).sum(1)

        # --- Reduce + summarize each metric over the PCA axis -----------------
        kvec = torch.tensor(kx.astype("f4"), device=device)               # (K+1,)
        dof = torch.clamp(ngood_valid_t[:, None] - n_nmf - kvec[None, :], min=1)
        red_glob = chi2_glob / dof                                        # (n_valid, K+1)
        red_line = chi2_line / ngood_line_valid_t[:, None]

        out["global_redchi2_mean"][gi]   = red_glob.mean(0).cpu().numpy()
        out["global_redchi2_median"][gi] = red_glob.median(0).values.cpu().numpy()
        out["line_chi2_mean"][gi]        = red_line.mean(0).cpu().numpy()

        red_glob_hi = red_glob[hi_mask_t]
        out["hisnr_redchi2_mean"][gi] = red_glob_hi.mean(0).cpu().numpy()
        out["hisnr_redchi2_p99"][gi]  = torch.quantile(
            red_glob_hi, HISNR_PERCENTILE / 100.0, dim=0).cpu().numpy()

        # Cumulative explained variance of the held-out residual (per k).
        cev = np.zeros(K_MAX + 1, "f4")
        cev[1:k_use + 1] = (torch.cumsum(T2.sum(0), 0) / total_var_valid).cpu().numpy()
        if k_use < K_MAX:
            cev[k_use + 1:] = cev[k_use]
        out["cum_evr_valid"][gi] = cev

        # --- Parallel-analysis noise floor -----------------------------------
        col_std = Z_train.std(0, keepdim=True)
        Z_null = torch.randn(Z_train.shape, device=device) * col_std
        Z_null = Z_null - Z_null.mean(0, keepdim=True)
        _, S_null, _ = torch.linalg.svd(Z_null, full_matrices=False)
        above = (S > S_null).cpu().numpy()
        fbn = int(np.argmax(~above)) if (~above).any() else int(S.shape[0])
        first_below_null[gi] = fbn

        print(f"  n_nmf={n_nmf}: pure-NNMF redchi2 mean={nmf_only_global_mean[gi]:.4f}; "
              f"global(k=0)={out['global_redchi2_mean'][gi,0]:.4f} -> "
              f"(k={K_MAX})={out['global_redchi2_mean'][gi,K_MAX]:.4f}; "
              f"hisnr_p99(k=0)={out['hisnr_redchi2_p99'][gi,0]:.3f} -> "
              f"(k={K_MAX})={out['hisnr_redchi2_p99'][gi,K_MAX]:.3f}; "
              f"null crossing k={fbn}")

        del Z_train, Z_valid, T, T2, cumT2, chi2_glob, chi2_line, recon_line
        del Vt, S, S_null, Z_null
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print_stage("Saving combined grid")
    np.savez(f"{RESULTS_DIR}/nnmf_pca_grid_summary.npz",
             n_nmf_grid=np.array(N_TEMPLATE_GRID),
             k_grid=kx,
             first_below_null=first_below_null,
             nmf_only_global_mean=nmf_only_global_mean,
             **out)
    print(f"saved -> {RESULTS_DIR}/nnmf_pca_grid_summary.npz")

    # ------------------------------------------------------------------
    # Quick-look diagnostic (the paper figure is assembled in the notebook)
    # ------------------------------------------------------------------
    n_arr = np.array(N_TEMPLATE_GRID)
    extent = [kx[0] - 0.5, kx[-1] + 0.5, 0, len(n_arr)]
    panels = [("global_redchi2_mean", r"global mean reduced $\chi^2$"),
              ("line_chi2_mean",      r"line-region $\chi^2$ / line pixel"),
              ("hisnr_redchi2_p99",   r"hi-S/N p99 reduced $\chi^2$")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), layout="constrained")
    for ax, (key, label) in zip(axes, panels):
        Zg = out[key]
        im = ax.imshow(Zg, origin="lower", aspect="auto", extent=extent, cmap="magma")
        cs = ax.contour(kx, np.arange(len(n_arr)) + 0.5, Zg,
                        colors="w", linewidths=0.7, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        # null-floor crossing per n_nmf
        ax.plot(first_below_null, np.arange(len(n_arr)) + 0.5, "c.-", lw=1,
                ms=6, label="noise-floor crossing")
        ax.set_yticks(np.arange(len(n_arr)) + 0.5)
        ax.set_yticklabels(n_arr)
        ax.set_xlabel("# residual PCA components")
        ax.set_ylabel("# NNMF templates")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].legend(loc="upper right", fontsize=8)
    plt.savefig(f"{RESULTS_DIR}/nnmf_pca_grid.pdf", bbox_inches="tight")
    plt.close()
    print(f"saved -> {RESULTS_DIR}/nnmf_pca_grid.pdf\nDone.")
