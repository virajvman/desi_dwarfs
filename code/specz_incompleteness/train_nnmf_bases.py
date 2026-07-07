'''
Stage 2: Train the two NNMF bases and build the bootstrap coefficient pool.

  W_gen  (N_TEMPLATES_GEN, ~10): rich basis for GENERATING realistic spectra.
  W_feat (N_TEMPLATES_FEAT, ~2-3): coarse basis for MEASURING spectral type
         as NN features.

Both are trained on the same broad reference sample (Stage 1) with the exact
warm-start + joint-optimize recipe from code/nnmf_pca_analysis/nnmf_analysis.py.
De-redshifting uses the FLUX-CONSERVING rebin (ivar-weighted resampling
suppresses emission-line cores, which would weaken lines in every generated
spectrum).

The bootstrap pool = W_gen coefficients of the BRIGHT subset (IS_POOL), with
an emission-strength score + stratum per galaxy computed on the W_gen
reconstruction (the mocks *are* reconstructions, so stratification must see
what generation sees). A pool-coverage diagnostic figure is written --
inspect it before generating mocks: bootstrap diversity is set by the number
of DISTINCT pool vectors per stratum, not by the number of mock draws.

To choose N_TEMPLATES_FEAT empirically, run
code/nnmf_pca_analysis/scan_nnmf_ntemplates.py pointed at the rest-frame file
this script writes.

Run on a NERSC GPU node (cupy + nearly_nmf).
Outputs: config.GEN_BASIS_NPY, config.FEAT_BASIS_NPY, config.POOL_H5
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

import config
import specz_utils

REST_H5 = f"{config.REFERENCE_DIR}/reference_sample_restframe.h5"
N_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", 64))


# ----------------------------------------------------------------------------
# De-redshifting (cached)
# ----------------------------------------------------------------------------
_ARGS = None


def _dered_worker(i):
    wave, flux, ivar, zreds, wave_out = _ARGS
    return specz_utils.deredshift_resample_one(wave, flux[i], ivar[i], zreds[i],
                                               wave_out)


def _init_dered(args):
    global _ARGS
    _ARGS = args


def build_restframe_file():
    if os.path.exists(REST_H5):
        print(f"Using cached rest-frame file {REST_H5}")
        return

    print("De-redshifting reference spectra (flux-conserving) ...")
    with h5py.File(config.REFERENCE_SPECTRA_H5, "r") as f:
        wave = f["WAVE"][:]
        flux = f["FLUX"][:]
        ivar = f["FLUX_IVAR"][:]
        zreds = f["Z"][:]
        tgids = f["TARGETID"][:]
        rfib = f["RFIB_SYNTH"][:]
        is_pool = f["IS_POOL"][:]

    wave_rest = specz_utils.get_rest_wave()
    with Pool(processes=N_PROC, initializer=_init_dered,
              initargs=((wave, flux, ivar, zreds, wave_rest),)) as pool:
        results = pool.map(_dered_worker, range(len(flux)), chunksize=64)
    results = np.array(results)  # (n, 2, n_pix)

    with h5py.File(REST_H5, "w") as f:
        f.create_dataset("TARGETID", data=tgids, dtype="i8")
        f.create_dataset("Z", data=zreds, dtype="f8")
        f.create_dataset("WAVE_REST", data=wave_rest, dtype="f4")
        f.create_dataset("FLUX", data=results[:, 0], dtype="f4")
        f.create_dataset("FLUX_IVAR", data=results[:, 1], dtype="f4")
        f.create_dataset("RFIB_SYNTH", data=rfib, dtype="f4")
        f.create_dataset("IS_POOL", data=is_pool, dtype="i1")
    print(f"Wrote {REST_H5}")


# ----------------------------------------------------------------------------
# NNMF training / fitting (mirrors nnmf_analysis.py)
# ----------------------------------------------------------------------------
def train_templates(flux_cp, ivar_cp, n_templates, rng):
    import cupy as cp
    from nearly_nmf import nmf

    n_pix, n_train = flux_cp.shape
    H = cp.array(rng.uniform(0, 1, (n_templates, n_train)))
    W = cp.array(np.ones((n_pix, n_templates)))

    for i in range(n_templates):
        print(f"  warm-start template {i + 1}/{n_templates}")
        H_itr, W_itr = nmf.nearly_NMF(flux_cp, ivar_cp,
                                      H[:(i + 1), :], W[:, :(i + 1)],
                                      n_iter=config.NNMF_N_ITER_WARM)
        H[:(i + 1), :] = H_itr
        W[:, :(i + 1)] = W_itr

    H, W, _ = nmf.nearly_NMF(flux_cp, ivar_cp, H, W,
                             n_iter=config.NNMF_N_ITER_JOINT, return_chi_2=True)
    return cp.asnumpy(W)


_FIT = None


def _init_fit(args):
    global _FIT
    _FIT = args


def _fit_one(i):
    from scipy.optimize import nnls
    W, flux, ivar = _FIT
    sqrt_ivar = np.sqrt(ivar[:, i])
    coeffs, rnorm = nnls(sqrt_ivar[:, None] * W, sqrt_ivar * flux[:, i])
    return coeffs, rnorm


def fit_templates(W, flux, ivar):
    """NNLS-fit templates to all spectra (columns). Returns (coeffs, rnorms)."""
    n_obj = flux.shape[1]
    with Pool(processes=N_PROC, initializer=_init_fit,
              initargs=((W, flux, ivar),)) as pool:
        out = pool.map(_fit_one, range(n_obj), chunksize=128)
    coeffs = np.array([o[0] for o in out])
    rnorms = np.array([o[1] for o in out])
    return coeffs, rnorms


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import cupy as cp

    specz_utils.check_dirs()
    rng = np.random.default_rng(config.NNMF_SEED)

    build_restframe_file()

    with h5py.File(REST_H5, "r") as f:
        tgids = f["TARGETID"][:]
        zreds = f["Z"][:]
        wave_rest = f["WAVE_REST"][:]
        flux = f["FLUX"][:].T            # (n_pix, n_obj)
        ivar = f["FLUX_IVAR"][:].T
        rfib = f["RFIB_SYNTH"][:]
        is_pool = f["IS_POOL"][:].astype(bool)

    n_obj = flux.shape[1]
    print(f"{n_obj} spectra on {flux.shape[0]} rest-frame pixels")

    # ------------------------------------------------------------------
    # Normalization: single-template amplitude fit (as in nnmf_analysis.py)
    # ------------------------------------------------------------------
    print("Normalization template ...")
    n_sub = min(75000, n_obj)
    idx = rng.choice(n_obj, size=n_sub, replace=False)
    W_norm = train_templates(cp.array(flux[:, idx]), cp.array(ivar[:, idx]),
                             1, rng)

    norm_coeffs, _ = fit_templates(W_norm, flux, ivar)
    norm_coeffs = norm_coeffs[:, 0]
    good = norm_coeffs > 0
    print(f"dropping {int((~good).sum())} spectra with zero normalization")

    flux, ivar = flux[:, good], ivar[:, good]
    tgids, zreds, rfib, is_pool = tgids[good], zreds[good], rfib[good], is_pool[good]
    scaling = 1.0 / norm_coeffs[good]
    flux = flux * scaling
    ivar = ivar / scaling ** 2

    # ------------------------------------------------------------------
    # Train both bases on a random training half
    # ------------------------------------------------------------------
    n_obj = flux.shape[1]
    perm = rng.permutation(n_obj)
    train_inds = perm[: n_obj // 2]

    flux_train_cp = cp.array(flux[:, train_inds])
    ivar_train_cp = cp.array(ivar[:, train_inds])

    print(f"Training W_gen ({config.N_TEMPLATES_GEN} templates) ...")
    W_gen = train_templates(flux_train_cp, ivar_train_cp,
                            config.N_TEMPLATES_GEN, rng)
    np.save(config.GEN_BASIS_NPY, W_gen)

    print(f"Training W_feat ({config.N_TEMPLATES_FEAT} templates) ...")
    W_feat = train_templates(flux_train_cp, ivar_train_cp,
                             config.N_TEMPLATES_FEAT, rng)
    np.save(config.FEAT_BASIS_NPY, W_feat)

    del flux_train_cp, ivar_train_cp

    # ------------------------------------------------------------------
    # Bootstrap pool: W_gen coefficients of the bright subset
    # ------------------------------------------------------------------
    print(f"Fitting W_gen to the bright pool ({int(is_pool.sum())} galaxies) ...")
    pool_flux = np.ascontiguousarray(flux[:, is_pool])
    pool_ivar = np.ascontiguousarray(ivar[:, is_pool])
    pool_coeffs, pool_rnorms = fit_templates(W_gen, pool_flux, pool_ivar)

    # Emission score on the RECONSTRUCTION (what generation will produce)
    recon = W_gen @ pool_coeffs.T          # (n_pix, n_pool)
    scores = specz_utils.emission_line_score(wave_rest, recon)
    strata, _ = specz_utils.assign_strata(scores)

    with h5py.File(config.POOL_H5, "w") as f:
        f.create_dataset("TARGETID", data=tgids[is_pool], dtype="i8")
        f.create_dataset("Z", data=zreds[is_pool], dtype="f8")
        f.create_dataset("RFIB_SYNTH", data=rfib[is_pool], dtype="f4")
        f.create_dataset("COEFFS", data=pool_coeffs, dtype="f4")
        f.create_dataset("RNORM", data=pool_rnorms, dtype="f4")
        f.create_dataset("NORM_FACTOR", data=scaling[is_pool], dtype="f4")
        f.create_dataset("EMISSION_SCORE", data=scores, dtype="f4")
        f.create_dataset("STRATUM", data=strata, dtype="i4")
        f.attrs["note"] = ("COEFFS are W_gen coefficients of NORMALIZED spectra "
                           "(flux * NORM_FACTOR). Generation re-sets amplitude via "
                           "the target fiber mag, so the convention drops out, but "
                           "it is recorded here to prevent double-scaling.")
    print(f"Wrote {config.POOL_H5}")

    # ------------------------------------------------------------------
    # Diagnostics: templates + REQUIRED pool-coverage figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(config.N_TEMPLATES_GEN, 1,
                             figsize=(14, 2 * config.N_TEMPLATES_GEN),
                             layout="constrained")
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(wave_rest, W_gen[:, i], lw=0.8, color="firebrick")
        ax.set_ylabel(f"T{i}")
    axes[-1].set_xlabel("rest wavelength [A]")
    plt.savefig(f"{config.BASES_DIR}/W_gen_templates.pdf", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(config.N_TEMPLATES_FEAT, 1,
                             figsize=(14, 2 * config.N_TEMPLATES_FEAT),
                             layout="constrained")
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(wave_rest, W_feat[:, i], lw=0.8, color="navy")
        ax.set_ylabel(f"T{i}")
    axes[-1].set_xlabel("rest wavelength [A]")
    plt.savefig(f"{config.BASES_DIR}/W_feat_templates.pdf", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].hist(scores, bins=100)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("emission-line score (summed pseudo-EW) [A]")
    axes[0].set_ylabel("pool galaxies")
    counts = np.bincount(strata, minlength=config.N_STRATA)
    axes[1].bar(np.arange(len(counts)), counts)
    axes[1].set_xlabel("stratum (0 = passive)")
    axes[1].set_ylabel("distinct pool vectors")
    axes[1].set_title(f"passive stratum: {counts[0]} distinct vectors")
    plt.savefig(f"{config.BASES_DIR}/pool_coverage_diagnostic.pdf",
                bbox_inches="tight")
    plt.close()

    print("Pool stratum occupancy (0 = passive):", counts)
    if counts[0] < 500:
        print("WARNING: fewer than 500 distinct passive pool vectors -- "
              "bootstrap diversity in the quenched corner is limited. Consider "
              "relaxing POOL_RFIB_MAX or REF_DELTACHI2_MIN before generating.")
    print("Done.")
