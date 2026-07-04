"""
stack_mstar_elg_vs_noelg.py
===========================

ELG vs non-ELG bootstrap-stacked spectra in 1D bins of log M_star.

Bootstrap/stacking logic is kept IDENTICAL to the finalized
``code/nebular_stuff/stack_mstar_haew_5pct.py`` product (the Scholte recipe):

  - Each spectrum is normalized by its catalog HALPHA_FLUX, then N_BOOTSTRAP
    realizations are formed by drawing the full valid sample WITH replacement
    (``bootstrap_stack`` draws size=n_valid; there is no fixed n_draw cap).
  - The mean stack (row 0) carries ``central_ivar`` = the mean over realizations
    of the per-realization PROPAGATED MEASUREMENT ivar (Scholte step v). Each
    bootstrap row k carries its own ``real_flux[k]`` and ``real_ivar[k]``. The
    per-pixel bootstrap standard deviation is a DIAGNOSTIC only and is never
    written as an ivar.
  - All N_BOOTSTRAP realizations are saved, in order, as rows 1..N_BOOT_SAVE of
    the FITS file for stackfit.
  - Each (sample, mass) cell uses its own stable RNG seed derived from the bin
    definition, so realizations are independent across cells and reproducible.

Per cell: stack only when the CATALOG count >= STACK_NLIM (gate on the catalog
count only; once it passes, however many spectra matched are stacked, down to 1,
via ``min_n_valid=1``). Sample selection is ``select_sample``, expressed via the
IS_* target-membership flags (the SAMPLE column has been dropped from the catalog):
  - ELG    : IS_ELG & DWARF_PRIMARY & MAG_TYPE=="TRACTOR_OG" & 0.09 < z < 0.13
             (== the notebook's main_elg_f); in --norm continuum mode the ELG
             sample additionally requires MSTAR_MASKBIT==0 (reliable stellar
             masses), so mass-binned trends are not driven by bad masses
  - NO_ELG : (IS_BGS_BRIGHT | IS_BGS_FAINT | IS_LOWZ) & DWARF_PRIMARY, full z
Both samples share a continuum-S/N gate (MAG_R_FIBER_NOEMI_ERR < 0.1); the Halpha-
detection gate in the default load_catalog is bypassed here via
``load_catalog(apply_halpha_cut=False)`` (a continuum stack does not need Halpha).
The ELG z-slice yields a clean aperture-matched population; non-ELG keeps full z
(the slice would gut BGS), so the comparison is asymmetric in z by design.

The de-redshifted spectra are read from the SAME flux-conserving (``_noinvvar``)
file as haew_5pct, so both products stack identically-rebinned spectra.

Filenames intentionally match what the notebook / fastspec job read downstream:
  - stacks_spec_elg_mstar_{mlo}_{mhi}.pkl      (and ..._noelg_...)
  - stack_mstar_elg_{mlo}_{mhi}.fits           (and ..._noelg_...)

Outputs (written to STACK_PATH; stale files removed at the start of each run):
  - stacks_spec_{elg,noelg}_mstar_{mlo}_{mhi}.pkl
  - stack_mstar_{elg,noelg}_{mlo}_{mhi}.fits   (1 mean + N_BOOT_SAVE rows)
  - plots/overlay_mstar_{mlo}_{mhi}.png        (ELG vs non-ELG per mass bin)
  - plots/grid_all_stacks.png                  (rows=mass bins, cols=ELG/non-ELG)
  - plots/ivar_vs_bootstd_{label}.png          (validation, representative bins)

Normalization mode (``--norm``):
  - ``halpha`` (default): divide each spectrum by its catalog HALPHA_FLUX
    (SF-weighted stack); writes to ``stack_files/mstar/``.
  - ``continuum``: divide by a continuum r-band flux scalar derived from the
    rest-frame, emission-removed model magnitude MAG_R_SDSS_Z0_MODEL_NOEMI,
    ``F_r = 10^(-0.4*(mag - 22.5))`` (luminosity-weighted stack); writes to
    ``stack_files/mstar_contnorm/``. The per-galaxy scalar cancels in all
    downstream line ratios in either mode -- it only reweights galaxies in the
    mean stack. Both modes use the identical ``catalog`` path in bootstrap_stack.

Usage:
    python stack_mstar_elg_vs_noelg.py                 # Halpha-normalized
    python stack_mstar_elg_vs_noelg.py --norm continuum  # continuum-normalized
"""

import argparse
import glob
import os
import sys

# Defensive: also make code/ importable here so this runner works whether
# invoked from code/, code/stacking_analysis/, or anywhere else. stack_explore
# does the same insertion when imported, but doing it locally too means this
# script keeps working if the helper layout changes.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pickle

import numpy as np

from stack_explore import (
    load_catalog,
    load_spectra,
    select_sample,
    get_sample_spectra_with_linenorm,
    bootstrap_stack,
    write_stacked_spectra,
)


# =============================================================================
# CONFIG
# =============================================================================

# (catalog_sample_name, filename_key). The catalog_sample_name is what
# select_sample matches against (it understands "ELG" and "NO_ELG"
# specially); the filename_key is the lowercase token used in the
# pickle / FITS filenames.
SAMPLE_SPECS = [
    ("ELG",    "elg"),
    ("NO_ELG", "noelg"),
]

# Stellar-mass bin edges (log Msun). 0.25 dex bins throughout.
MSTAR_BINS = np.arange(6.0, 9.25 + 1e-6, 0.25)

# Redshift windows. The ELG sample is restricted to a narrow slice centered on
# z=0.11 (z = 0.11 +/- 0.02), matching the notebook's main_elg_f -- a clean,
# aperture-matched ELG population for the continuum-normalized stack. The non-ELG
# (BGS+LOWZ) sample keeps the full range (BGS dwarfs sit at z << 0.11, so the
# slice would gut them); the comparison is asymmetric in redshift by design.
ELG_Z_MIN, ELG_Z_MAX = 0.07, 0.16
NOELG_Z_MIN, NOELG_Z_MAX = 0.0, 0.5

# Retained as the saved-metadata default; the actual per-sample window is resolved
# at selection time via sample_z_window().
Z_MIN_GLOBAL = 0.0
Z_MAX_GLOBAL = 0.5

# Continuum-S/N gate applied to BOTH samples (this is a continuum-normalized
# product, so the Halpha-detection gate in the default load_catalog is bypassed
# via load_catalog(apply_halpha_cut=False)): keep only galaxies whose emission-
# removed r-band fiber magnitude error is below this (a well-measured continuum).
RBAND_NOEMI_ERR_MAX = 0.1

# Minimum number of CATALOG galaxies in a bin to attempt a stack. Gate is on
# the catalog count only; once it passes, however many spectra matched are
# stacked (min_n_valid=1 inside bootstrap_stack). Matches haew_5pct's EW_STACK_NLIM.
STACK_NLIM = 50

# Bootstrap settings (Scholte: 200 realizations, full-sample draws with replacement).
N_BOOTSTRAP = 50       # number of bootstrap realizations (reduced from 200 for speed)
N_BOOT_SAVE = 200      # how many realizations to save as FITS rows 1.. (capped at N_BOOTSTRAP)
RANDOM_SEED = 42       # base seed; each (sample, mass) cell adds a stable offset

# Worker processes for the per-realization coadds inside bootstrap_stack. The
# non-ELG bins are large (a few GB working set per worker), so 16 keeps the peak
# comfortably within a full Perlmutter CPU node (512 GB, run with --mem=0). Raise
# if bins are small / memory is ample; lower if you hit OOM. Run with
# OMP_NUM_THREADS=1 (the orchestrator sets this) so workers don't oversubscribe.
BOOT_NJOBS = 16

# Cap on the number of spectra fed into any single stack. The non-ELG high-mass
# bins reach 50k-100k galaxies; the full-sample stack is S/N-saturated, so a
# random subsample to this size gives an indistinguishable mean stack at a small
# fraction of the per-realization memory/time (it was 16 workers x a ~6.5 GB
# resample copy that OOM-killed the 100k bin). Bins below the cap are untouched,
# so the low-mass dwarf bins keep full fidelity; the bootstrap error on capped
# (high-mass) bins reflects this cap, i.e. mildly conservative. None disables it.
# Tunable: raise to ~20000 for tighter errors on the biggest bins -- still
# memory-safe on a full node.
MAX_STACK_SPECTRA = 20000

# Output location. Selected by --norm in main(): the Halpha-normalized product
# writes to mstar/, the continuum (r-band) normalized product to mstar_contnorm/,
# so the two coexist with identical per-bin filenames in separate directories.
STACK_PATH_HALPHA = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar/"
STACK_PATH_CONTNORM = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_contnorm/"
STACK_PATH = STACK_PATH_HALPHA   # overridden for continuum mode in main()

# De-redshifted spectra file -- the SAME flux-conserving (_noinvvar) rebin used
# by stack_mstar_haew_5pct.py, so both products stack identically-rebinned spectra.
SPECTRA_FILE = (
    "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/"
    "desi_y1_dwarf_combine_deredshift_hires_noinvvar.h5"
)

# Re-do stacks even if a cached pickle already exists on disk?
OVERWRITE_STACKS = True

# Rest-frame wavelength upper bound (consistent with haew_5pct).
WAVE_MAX = 6800

# Spectrum-normalization method passed into bootstrap_stack. Both --norm modes
# use the "catalog" path (divide each spectrum by a per-galaxy scalar); only the
# scalar differs, so the stacking core is untouched.
#   "catalog"     - normalize each spectrum by a per-galaxy catalog flux scalar
#   "boxcar_line" - normalize by a self-measured boxcar Halpha flux
#   "flux_window" - normalize by integrated flux in a continuum window (unused)
NORM_METHOD = "catalog"

# Spectrum-normalization MODE, set by --norm in main(); default "halpha" so the
# existing run is unchanged. The per-galaxy scalar cancels in all downstream line
# ratios in either mode; it only reweights galaxies in the mean stack (Halpha ->
# SF-weighted; continuum -> luminosity-weighted).
#   "halpha"    - divide each spectrum by its catalog HALPHA_FLUX
#   "continuum" - divide by a continuum r-band flux scalar (see CONT_* below)
NORM_MODE = "halpha"

# Per-galaxy normalization column (the scalar each spectrum is divided by).
# Set in main() to CONT_FLUX_COL when --norm continuum. Default HALPHA_FLUX:
# Gaussian FLUX is the more reliable line-flux measurement, and since every
# load_catalog galaxy already passes HALPHA_FLUX > 1, normalizing by it drops no
# galaxies. Matches haew_5pct.
NORM_COL = "HALPHA_FLUX"

# Continuum-mode normalization (--norm continuum). The scalar is built from the
# rest-frame (z=0), emission-removed FastSpec model r-band magnitude:
#   F_r = 10^(-0.4*(MAG_R_SDSS_Z0_MODEL_NOEMI - 22.5))   [nanomaggies]
# The nanomaggie zeropoint keeps normalized continua O(1) (the scalar cancels
# downstream, so the ZP is purely numerical hygiene). Galaxies with a non-finite
# model mag (unmatched in the model-photometry catalog) get F_r=NaN and are
# dropped before stacking, so NOBJ reflects the honestly-stacked count.
CONT_MAG_COL = "MAG_R_SDSS_Z0_MODEL_NOEMI"
CONT_MAG_ZP = 22.5
CONT_FLUX_COL = "CONT_FLUX_R"

# Human-readable normalization label for plot axes/titles (set in main()).
NORM_LABEL = "Halpha-normalized"

# Plotting.
SAMPLE_COLORS = {"elg": "#d62728", "noelg": "#1f77b4"}
SAMPLE_PLOT_LABELS = {"elg": "ELG", "noelg": "non-ELG"}
LINE_GUIDES = {
    "[OII]": 3727.0,
    r"H$\beta$": 4862.7,
    "[OIII]": 5008.2,
    r"H$\alpha$": 6564.6,
}


# =============================================================================
# HELPERS
# =============================================================================

def clean_stack_outputs(stack_path):
    """Remove previous stack, pickle, and FastSpec output files before a fresh run."""
    patterns = (
        "stacks_spec_elg_mstar_*.pkl",
        "stacks_spec_noelg_mstar_*.pkl",
        "stack_mstar_elg_*.fits",
        "stack_mstar_noelg_*.fits",
        "fastspec_stack_mstar_elg_*.fits",
        "fastspec_stack_mstar_noelg_*.fits",
    )
    n_removed = 0
    for pattern in patterns:
        for fpath in glob.glob(os.path.join(stack_path, pattern)):
            os.remove(fpath)
            n_removed += 1
    print(f"    Removed {n_removed} previous stack/.pkl/fastspec files from {stack_path}")


def bin_seed_index(sample_idx, i_mstar, n_mstar):
    """Stable per-bin RNG seed offset from bin definition (not loop order).

    Each (sample, mass-bin) cell gets a distinct offset so realizations are
    independent across cells and reproducible regardless of iteration order.
    """
    return sample_idx * n_mstar + i_mstar


def sample_z_window(sample_name):
    """(z_min, z_max) for a sample: narrow slice for ELG, full range for non-ELG."""
    if sample_name == "ELG":
        return ELG_Z_MIN, ELG_Z_MAX
    return NOELG_Z_MIN, NOELG_Z_MAX


def stack_one_bin(sub_cat, spectra_data, wave, sample_name, file_key,
                  mstar_min, mstar_max, bin_index,
                  z_min=Z_MIN_GLOBAL, z_max=Z_MAX_GLOBAL):
    """Bootstrap-stack one (sample, mass) cell; return saved dict or None."""
    n_sub = len(sub_cat)
    if n_sub < STACK_NLIM:
        print(f"      Skipping (catalog N={n_sub} < {STACK_NLIM})")
        return None

    pkl_path = os.path.join(
        STACK_PATH,
        f"stacks_spec_{file_key}_mstar_{mstar_min:.2f}_{mstar_max:.2f}.pkl",
    )

    if os.path.exists(pkl_path) and not OVERWRITE_STACKS:
        print(f"      Loading cached: {os.path.basename(pkl_path)}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # Match catalog rows to spectra by TARGETID and pull the per-spectrum
    # normalization scalar (HALPHA_FLUX or the continuum r-band flux).
    out = get_sample_spectra_with_linenorm(
        sub_cat, spectra_data, line_norm="HALPHA", norm_col=NORM_COL,
    )
    fluxes, ivars, norm_scalars, tgids_matched = out

    if fluxes is None or len(fluxes) == 0:
        print("      No matched spectra; skipping.")
        return None

    # Drop spectra with a non-finite / non-positive normalization scalar before
    # stacking. This is a no-op for Halpha mode (load_catalog already requires
    # HALPHA_FLUX > 1), but in continuum mode the model r-mag is NaN for galaxies
    # unmatched in the model-photometry catalog; dropping them up front keeps
    # n_matched / NOBJ honest (bootstrap_stack would otherwise silently mask them).
    norm_scalars = np.asarray(norm_scalars, dtype=float)
    tgids_matched = np.asarray(tgids_matched)
    good = np.isfinite(norm_scalars) & (norm_scalars > 0)
    n_drop = int((~good).sum())
    if n_drop:
        print(f"      Dropping {n_drop}/{len(good)} spectra with non-finite/<=0 "
              f"{NORM_COL} (norm scalar)")
        fluxes = fluxes[good]
        ivars = ivars[good]
        norm_scalars = norm_scalars[good]
        tgids_matched = tgids_matched[good]
    if len(fluxes) == 0:
        print("      No spectra with a valid normalization scalar; skipping.")
        return None

    n_matched = len(fluxes)
    seed = RANDOM_SEED + bin_index
    print(f"      Bootstrap-stacking: N={n_sub}, matched={n_matched}, "
          f"n_bootstrap={N_BOOTSTRAP}, seed={seed}")

    # Bootstrap-stack (Scholte recipe):
    #   - normalize each spectrum (catalog Halpha)
    #   - draw n_valid indices WITH replacement, N_BOOTSTRAP times
    #   - central_flux = mean of realizations; central_ivar = mean of the
    #     per-realization propagated measurement ivar; boot_std = diagnostic std
    central_flux, boot_std, real_flux, real_ivar, central_ivar = bootstrap_stack(
        fluxes=fluxes,
        ivars=ivars,
        wave=wave,
        n_bootstrap=N_BOOTSTRAP,
        random_seed=seed,
        norm_method=NORM_METHOD,
        catalog_line_fluxes=norm_scalars,
        min_n_valid=1,
        max_spectra=MAX_STACK_SPECTRA,
        n_jobs=BOOT_NJOBS,
    )

    if real_flux is None:
        print("      bootstrap_stack returned None; skipping.")
        return None

    saved = {
        "stack_spec":   central_flux,
        "stack_err":    boot_std,       # diagnostic only (never written as ivar)
        "central_ivar": central_ivar,
        "real_flux":    real_flux,
        "real_ivar":    real_ivar,
        "all_stacks":   real_flux,
        "sample":       sample_name,
        "file_key":     file_key,
        "mstar_min":    float(mstar_min),
        "mstar_max":    float(mstar_max),
        "z_min":        z_min,
        "z_max":        z_max,
        "n_galaxies":   int(n_sub),
        "n_matched":    int(n_matched),
        "tgids":        np.asarray(tgids_matched),
        "bin_index":    int(bin_index),
        "random_seed":  int(seed),
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(saved, f)
    print(f"      Saved {os.path.basename(pkl_path)}")
    return saved


def write_multi_row_fits(saved, wave_for_fits, file_key):
    """Write 1 central + N_BOOT_SAVE bootstrap rows for FastSpecFit stackfit.

    Row 0 = mean stack with central_ivar (mean propagated measurement ivar over
    realizations). Rows 1..N_BOOT_SAVE = the bootstrap realizations IN ORDER,
    each with its own propagated measurement ivar. The bootstrap std is NOT
    written -- the spread ACROSS rows is the actual error budget for stackfit.
    """
    central_flux = saved["stack_spec"]
    central_ivar = saved["central_ivar"]
    real_flux = saved["real_flux"]
    real_ivar = saved["real_ivar"]
    # NOBJ is the number of spectra actually stacked (matched to a spectrum),
    # which can be < the catalog cell count when spectra are missing. The
    # N>=STACK_NLIM gate is on the catalog count (n_cat), kept here as NCAT for
    # provenance so the true stacked N is never misrepresented.
    n_matched = saved["n_matched"]
    n_cat = saved["n_galaxies"]
    mstar_min = saved["mstar_min"]
    mstar_max = saved["mstar_max"]

    n_boot_keep = min(N_BOOT_SAVE, len(real_flux))
    n_rows = 1 + n_boot_keep
    n_wave = len(wave_for_fits)

    all_flux = np.zeros((n_rows, n_wave), dtype=np.float32)
    all_ivar = np.zeros((n_rows, n_wave), dtype=np.float32)

    all_flux[0] = np.asarray(central_flux, dtype=np.float32)
    all_ivar[0] = np.asarray(central_ivar, dtype=np.float32)
    for k in range(n_boot_keep):
        all_flux[k + 1] = np.asarray(real_flux[k], dtype=np.float32)
        all_ivar[k + 1] = np.asarray(real_ivar[k], dtype=np.float32)

    out_fits = os.path.join(
        STACK_PATH,
        f"stack_mstar_{file_key}_{mstar_min:.2f}_{mstar_max:.2f}.fits",
    )

    write_stacked_spectra(
        outfile=out_fits,
        wave=wave_for_fits,
        flux=all_flux,
        ivar=all_ivar,
        stackids=np.arange(n_rows, dtype=np.int64),
        stack_redshift=np.zeros(n_rows),
        table_column_dict={
            "IS_MEAN":   np.array([1] + [0] * n_boot_keep, dtype=np.int64),
            "NOBJ":      np.full(n_rows, n_matched, dtype=np.int64),
            "NCAT":      np.full(n_rows, n_cat, dtype=np.int64),
            "MSTAR_MIN": np.full(n_rows, mstar_min, dtype=np.float32),
            "MSTAR_MAX": np.full(n_rows, mstar_max, dtype=np.float32),
        },
        table_format_dict={
            "IS_MEAN":   "K",
            "NOBJ":      "K",
            "NCAT":      "K",
            "MSTAR_MIN": "E",
            "MSTAR_MAX": "E",
        },
    )
    print(f"    {saved['sample']} | log M*=[{mstar_min:.2f},{mstar_max:.2f}]: "
          f"N_stacked={n_matched} (catalog N={n_cat}), "
          f"1 mean + {n_boot_keep} bootstraps -> {os.path.basename(out_fits)}")


# =============================================================================
# PLOTTING
# =============================================================================

def _add_line_guides(ax):
    for lam in LINE_GUIDES.values():
        ax.axvline(lam, color="grey", ls=":", lw=0.8, alpha=0.6, zorder=0)


def make_overlay_plots(results, wave, mstar_bins, plot_dir):
    """One panel per stellar-mass bin overlaying the ELG and non-ELG stacks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1

    for i in range(n_mstar):
        m_lo, m_hi = mstar_bins[i], mstar_bins[i + 1]
        present = [
            (file_key, results[file_key].get(i))
            for _, file_key in SAMPLE_SPECS
        ]
        present = [(fk, s) for fk, s in present if s is not None]
        if not present:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        for file_key, saved in present:
            flux = saved["stack_spec"]
            err = saved["stack_err"]
            n_gal = saved["n_galaxies"]
            color = SAMPLE_COLORS.get(file_key, "k")
            label = SAMPLE_PLOT_LABELS.get(file_key, file_key)
            ax.plot(wave, flux, color=color, lw=1.0, label=f"{label}  (N={n_gal})")
            ax.fill_between(wave, flux - err, flux + err, color=color, alpha=0.20, lw=0)

        _add_line_guides(ax)
        ax.set_xlim(wave.min(), wave.max())
        ax.set_xlabel(r"Rest wavelength [$\AA$]")
        ax.set_ylabel(f"{NORM_LABEL} stacked flux")
        ax.set_title(f"log M* in [{m_lo:.2f}, {m_hi:.2f}]")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        fig.tight_layout()

        out_png = os.path.join(plot_dir, f"overlay_mstar_{m_lo:.2f}_{m_hi:.2f}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"    Saved {os.path.basename(out_png)}")


def make_grid_plot(results, wave, mstar_bins, plot_dir):
    """Grid: rows = mass bins, columns = sample (ELG, non-ELG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1
    n_samp = len(SAMPLE_SPECS)
    any_stack = any(
        results[fk].get(i) is not None
        for _, fk in SAMPLE_SPECS for i in range(n_mstar)
    )
    if not any_stack:
        print("    (no stacks; skipping grid_all_stacks)")
        return

    fig, axes = plt.subplots(
        n_mstar, n_samp,
        figsize=(3.5 * n_samp, 2.4 * n_mstar),
        sharex=True, squeeze=False,
    )

    for i in range(n_mstar):
        m_lo, m_hi = mstar_bins[i], mstar_bins[i + 1]
        for j, (_, file_key) in enumerate(SAMPLE_SPECS):
            ax = axes[i][j]
            saved = results[file_key].get(i)
            if saved is not None:
                flux = saved["stack_spec"]
                err = saved["stack_err"]
                color = SAMPLE_COLORS.get(file_key, "k")
                ax.plot(wave, flux, color=color, lw=0.8)
                ax.fill_between(wave, flux - err, flux + err, color=color, alpha=0.20, lw=0)
                ax.text(
                    0.97, 0.92, f"N={saved['n_galaxies']}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=7,
                )
                _add_line_guides(ax)
                ax.set_xlim(wave.min(), wave.max())
            else:
                ax.axis("off")
            if i == 0:
                ax.set_title(SAMPLE_PLOT_LABELS.get(file_key, file_key), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"[{m_lo:.1f},{m_hi:.1f}]", fontsize=9)
            if i == n_mstar - 1:
                ax.set_xlabel(r"Rest $\lambda$ [$\AA$]", fontsize=8)

    fig.suptitle(f"{NORM_LABEL} stacks (ELG vs non-ELG per mass bin)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_png = os.path.join(plot_dir, "grid_all_stacks.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


def make_ivar_diagnostic_plots(results, wave, plot_dir, max_bins=2):
    """Plot propagated measurement error vs bootstrap std (validation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_done = 0
    for _, file_key in SAMPLE_SPECS:
        for i, saved in sorted(results[file_key].items()):
            if saved is None or n_done >= max_bins:
                continue
            if "central_ivar" not in saved:
                continue

            central_ivar = np.asarray(saved["central_ivar"], dtype=float)
            boot_std = np.asarray(saved["stack_err"], dtype=float)
            with np.errstate(invalid="ignore"):
                meas_err = np.where(central_ivar > 0, 1.0 / np.sqrt(central_ivar), np.nan)

            label = f"{file_key}_mstar_{saved['mstar_min']:.2f}_{saved['mstar_max']:.2f}"
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(wave, meas_err, color="C0", lw=1.0,
                    label=r"$1/\sqrt{\mathrm{ivar}}$ (measurement)")
            ax.plot(wave, boot_std, color="C1", lw=1.0, alpha=0.8,
                    label="bootstrap std (diagnostic)")
            _add_line_guides(ax)
            ax.set_xlim(wave.min(), wave.max())
            ax.set_xlabel(r"Rest wavelength [$\AA$]")
            ax.set_ylabel("Per-pixel uncertainty")
            ax.set_title(f"ivar sanity check: {label}")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            fig.tight_layout()
            out_png = os.path.join(plot_dir, f"ivar_vs_bootstd_{label}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"    Saved {os.path.basename(out_png)}")
            n_done += 1


# =============================================================================
# MAIN
# =============================================================================

def main(resume=False, norm_mode="halpha"):
    global OVERWRITE_STACKS, STACK_PATH, NORM_COL, NORM_MODE, NORM_LABEL

    # Resolve the normalization mode -> output dir, scalar column, plot label.
    NORM_MODE = norm_mode
    if norm_mode == "halpha":
        STACK_PATH = STACK_PATH_HALPHA
        NORM_COL = "HALPHA_FLUX"
        NORM_LABEL = "Halpha-normalized"
    elif norm_mode == "continuum":
        STACK_PATH = STACK_PATH_CONTNORM
        NORM_COL = CONT_FLUX_COL
        NORM_LABEL = "continuum (r-band) normalized"
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode!r} (expected 'halpha' or 'continuum')")
    print(f"[0] Normalization mode: {norm_mode}  (scalar column: {NORM_COL})")
    print(f"    Output dir: {STACK_PATH}")

    os.makedirs(STACK_PATH, exist_ok=True)
    plot_dir = os.path.join(STACK_PATH, "plots")

    if resume:
        # Keep what's on disk and reuse cached per-bin pickles; stack_one_bin
        # then loads any existing .pkl instead of recomputing, so only the bins
        # still missing are (re)computed.
        OVERWRITE_STACKS = False
        print("[0] --resume: keeping existing outputs; cached per-bin pickles "
              "are reused, only missing bins are computed.")
        print("    NOTE: bins already on disk were stacked with whatever cap was "
              "in effect when they were made; mixing them with freshly capped "
              "bins is inconsistent. Prefer a clean (capped) rerun -- with the "
              "cap the whole job is fast.")
    else:
        print("[0] Cleaning previous stack outputs ...")
        clean_stack_outputs(STACK_PATH)

    # -------------------------------------------------------------------------
    # 1. Load catalog + spectra
    # -------------------------------------------------------------------------
    print("[1] Loading catalog ...")
    # Relaxed base cut (DWARF_MASKBIT only): this continuum-normalized product
    # gates on continuum S/N (MAG_R_FIBER_NOEMI_ERR) in select_sample, not on
    # Halpha. The default load_catalog Halpha gate is left intact for the
    # Halpha-normalized products that depend on it.
    tot_cat = load_catalog(apply_halpha_cut=False)
    print(f"    Total galaxies after base cut: {len(tot_cat)}")

    # In continuum mode, derive the per-galaxy continuum r-band flux scalar from
    # the rest-frame, emission-removed model r-mag: F_r = 10^(-0.4*(mag - ZP)).
    # Non-finite mags (galaxies unmatched in the model-photometry catalog) -> NaN
    # flux, which stack_one_bin then drops before stacking.
    if norm_mode == "continuum":
        mag = np.asarray(tot_cat[CONT_MAG_COL], dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            cont_flux = np.where(
                np.isfinite(mag), 10.0 ** (-0.4 * (mag - CONT_MAG_ZP)), np.nan,
            )
        tot_cat[CONT_FLUX_COL] = cont_flux
        n_finite = int(np.isfinite(cont_flux).sum())
        print(f"    Continuum scalar from {CONT_MAG_COL} (ZP={CONT_MAG_ZP}): "
              f"{n_finite}/{len(tot_cat)} galaxies have a finite model r-mag "
              f"(the rest are dropped per-bin at stacking).")

    print("\n[2] Loading de-redshifted spectra (flux-conserving _noinvvar) ...")
    spectra_data = load_spectra(SPECTRA_FILE)
    print(f"    Total spectra loaded: {len(spectra_data['targetid'])}")

    # Trim to lambda < WAVE_MAX (consistent with haew_5pct).
    wave_mask = spectra_data["wave_rest"] < WAVE_MAX
    spectra_data["wave_rest"] = spectra_data["wave_rest"][wave_mask]
    spectra_data["flux"]      = spectra_data["flux"][:, wave_mask]
    spectra_data["flux_ivar"] = spectra_data["flux_ivar"][:, wave_mask]
    wave = spectra_data["wave_rest"]
    print(f"    After trim: lambda in [{wave.min():.1f}, {wave.max():.1f}] A"
          f"  ({len(wave)} pixels)")

    # -------------------------------------------------------------------------
    # 2. Loop over mass bins x (ELG, NO_ELG)
    # -------------------------------------------------------------------------
    n_mstar = len(MSTAR_BINS) - 1
    print(f"\n[3] Stacking grid: {len(SAMPLE_SPECS)} samples"
          f" x {n_mstar} mstar bins"
          f" = {len(SAMPLE_SPECS) * n_mstar} candidate stacks "
          f"(stack when catalog N >= {STACK_NLIM})")
    print(f"    M* edges  : {MSTAR_BINS}")
    print(f"    Output dir: {STACK_PATH}")

    # results[file_key][i_mstar] -> dict (or None if skipped)
    results = {file_key: {} for _, file_key in SAMPLE_SPECS}

    for i in range(n_mstar):
        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]
        print(f"\n=== log M* in [{mstar_min:.2f}, {mstar_max:.2f}] ===")

        for sample_idx, (sample_name, file_key) in enumerate(SAMPLE_SPECS):
            print(f"\n  --- {sample_name} ---")

            z_lo, z_hi = sample_z_window(sample_name)
            # Continuum-normalized product only: require MSTAR_MASKBIT==0 for the
            # ELG sample (the flag is a no-op for NO_ELG inside select_sample), so
            # the ELG mass bins are not contaminated by unreliable stellar masses.
            # The Halpha-normalized product is left unchanged.
            sub_cat = select_sample(
                tot_cat, sample_name,
                z_min=z_lo, z_max=z_hi,
                logmstar_min=mstar_min, logmstar_max=mstar_max,
                rband_noemi_err_max=RBAND_NOEMI_ERR_MAX,
                require_mstar_maskbit=(NORM_MODE == "continuum"),
            )
            print(f"      N galaxies in bin: {len(sub_cat)} "
                  f"(z in [{z_lo:.2f}, {z_hi:.2f}])")

            bin_index = bin_seed_index(sample_idx, i, n_mstar)
            results[file_key][i] = stack_one_bin(
                sub_cat, spectra_data, wave, sample_name, file_key,
                mstar_min, mstar_max, bin_index,
                z_min=z_lo, z_max=z_hi,
            )

    # -------------------------------------------------------------------------
    # 3. Write FastSpecFit-compatible FITS for each non-empty bin
    # -------------------------------------------------------------------------
    print("\n[4] Writing FastSpecFit (stackfit) input FITS files ...")

    n_written = 0
    for _, file_key in SAMPLE_SPECS:
        for i, saved in sorted(results[file_key].items()):
            if saved is None:
                continue
            write_multi_row_fits(saved, wave, file_key)
            n_written += 1

    print(f"\n[5] Wrote {n_written} FITS files to {STACK_PATH}")

    # -------------------------------------------------------------------------
    # 4. Comparison + validation plots
    # -------------------------------------------------------------------------
    print("\n[6] Making ELG vs non-ELG comparison plots ...")
    make_overlay_plots(results, wave, MSTAR_BINS, plot_dir)
    make_grid_plot(results, wave, MSTAR_BINS, plot_dir)

    print("\n[7] ivar vs bootstrap-std diagnostic plots ...")
    make_ivar_diagnostic_plots(results, wave, plot_dir, max_bins=2)

    print(f"\n[8] Done. Plots in {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ELG vs non-ELG bootstrap-stacked spectra in log M* bins.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Keep existing outputs (skip the initial clean) and reuse cached "
             "per-bin pickles, computing only the bins still missing on disk.",
    )
    parser.add_argument(
        "--norm", choices=["halpha", "continuum"], default="halpha",
        help="Per-galaxy normalization: 'halpha' divides each spectrum by its "
             "catalog HALPHA_FLUX (SF-weighted stack, default, writes to mstar/); "
             "'continuum' divides by a continuum r-band flux scalar from "
             "MAG_R_SDSS_Z0_MODEL_NOEMI (luminosity-weighted, writes to "
             "mstar_contnorm/).",
    )
    args = parser.parse_args()
    main(resume=args.resume, norm_mode=args.norm)
