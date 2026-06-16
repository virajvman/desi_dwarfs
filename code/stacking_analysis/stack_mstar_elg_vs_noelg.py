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
via ``min_n_valid=1``). Sample selection is ``select_sample`` (load_catalog cuts
+ ELG/NO_ELG membership + z + mass) -- the stricter H-alpha EW/boxflux detection
cuts used by the haew_5pct EW product are deliberately NOT applied here, so the
non-ELG population is not gutted.

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

Usage:
    python stack_mstar_elg_vs_noelg.py
"""

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
MSTAR_BINS = np.arange(6.0, 9.5 + 1e-6, 0.25)

# Redshift range. Effectively no cut (we bin only in M*).
Z_MIN_GLOBAL = 0.0
Z_MAX_GLOBAL = 0.5

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

# Output location.
STACK_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar/"

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

# Spectrum-normalization method passed into bootstrap_stack.
#   "catalog"     - normalize each spectrum by its catalog HALPHA_FLUX (recommended)
#   "boxcar_line" - normalize by a self-measured boxcar Halpha flux
#   "flux_window" - normalize by integrated flux in a continuum window
NORM_METHOD = "catalog"
# Per-galaxy normalization line flux. Gaussian HALPHA_FLUX (not boxcar): FLUX is
# the more reliable line-flux measurement, and since every load_catalog galaxy
# already passes HALPHA_FLUX > 1, normalizing by it drops no galaxies. The
# normalization constant cancels in all downstream line ratios. Matches haew_5pct.
NORM_COL = "HALPHA_FLUX"

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


def stack_one_bin(sub_cat, spectra_data, wave, sample_name, file_key,
                  mstar_min, mstar_max, bin_index):
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
    # Halpha flux for normalization.
    out = get_sample_spectra_with_linenorm(
        sub_cat, spectra_data, line_norm="HALPHA", norm_col=NORM_COL,
    )
    fluxes, ivars, halpha_fluxes, tgids_matched = out

    if fluxes is None or len(fluxes) == 0:
        print("      No matched spectra; skipping.")
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
        catalog_line_fluxes=halpha_fluxes,
        min_n_valid=1,
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
        "z_min":        Z_MIN_GLOBAL,
        "z_max":        Z_MAX_GLOBAL,
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
        ax.set_ylabel("Halpha-normalized stacked flux")
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

    fig.suptitle("Halpha-normalized stacks (ELG vs non-ELG per mass bin)", fontsize=12)
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

def main():
    os.makedirs(STACK_PATH, exist_ok=True)
    plot_dir = os.path.join(STACK_PATH, "plots")

    print("[0] Cleaning previous stack outputs ...")
    clean_stack_outputs(STACK_PATH)

    # -------------------------------------------------------------------------
    # 1. Load catalog + spectra
    # -------------------------------------------------------------------------
    print("[1] Loading catalog ...")
    tot_cat = load_catalog()
    print(f"    Total galaxies after quality cuts: {len(tot_cat)}")

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

            sub_cat = select_sample(
                tot_cat, sample_name,
                z_min=Z_MIN_GLOBAL, z_max=Z_MAX_GLOBAL,
                logmstar_min=mstar_min, logmstar_max=mstar_max,
            )
            print(f"      N galaxies in bin: {len(sub_cat)}")

            bin_index = bin_seed_index(sample_idx, i, n_mstar)
            results[file_key][i] = stack_one_bin(
                sub_cat, spectra_data, wave, sample_name, file_key,
                mstar_min, mstar_max, bin_index,
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
    main()
