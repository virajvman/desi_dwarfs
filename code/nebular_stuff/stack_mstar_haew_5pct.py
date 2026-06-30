"""
stack_mstar_haew_5pct.py
========================

Two stacking products in one run (BGS_BRIGHT | BGS_FAINT | LOWZ combined):

**Product A — M* x H-alpha EW (bootstrap):**
Bootstrap-stacked spectra in log M_star bins (6 -> 9.25) crossed with three
fixed H-alpha EW bins:

Mass edges: 6, 6.5, 7, 7.5, 8 (0.5 dex); 8.25, 8.5, 8.75, 9, 9.25 (0.25 dex).

    1. EW <= 30 A
    2. 30 < EW <= 100 A
    3. EW > 100 A

Detection cuts (applied globally before EW binning):
  - HALPHA_EW * sqrt(HALPHA_EW_IVAR) >= 3
  - HALPHA_EW > 1 A
  - HALPHA_BOXFLUX * sqrt(HALPHA_BOXFLUX_IVAR) >= 3

Per (mass, EW) cell: stack only when N >= 50; otherwise skip (no pooled fallback).

Each output stack FITS has 1 central row (IS_MEAN=1) plus 200 bootstrap
realizations (IS_MEAN=0), following the Scholte et al. recipe (200 samples).

**Product B — M* only (mean, no bootstrap):**
Same mass bins, no EW sub-binning. Uses load_catalog cuts only (DWARF_MASKBIT,
HALPHA_FLUX SNR > 3, HALPHA_FLUX > 1) — not the stricter H-alpha detection
cuts above. No minimum N per bin. One mean coadd per mass bin.

**Product C — M* viz stacks (mean, FITS only, visualization):**
Integer-centered 0.5-dex bins (+/-0.25 dex around log M* = 7, 8, 9), same broad
load_catalog sample and mean coadd as Product B, but kept on the FULL red
wavelength grid (lambda < WAVE_MAX_VIZ = 9800 A; Products A/B trim to 6800).
Written to its own folder; NOT run through FastSpecFit or direct metallicity.

Outputs (written to STACK_PATH; stale files removed at the start of each run):

  EW-binned (STACK_PATH/):
  - stacks_spec_ALL_mstar_{mlo}_{mhi}_{ewtoken}.pkl
  - stack_ALL_mstar_{mlo}_{mhi}_{ewtoken}.fits   (1 + 200 rows)
  - plots/overlay_mstar_{mlo}_{mhi}.png
  - plots/grid_all_stacks.png
  - plots/ivar_vs_bootstd_{label}.png  (validation, representative bins)

  Mass-only (STACK_PATH/mstar_only/):
  - stacks_spec_ALL_mstar_{mlo}_{mhi}.pkl
  - stack_ALL_mstar_{mlo}_{mhi}.fits   (1 row)
  - plots/overlay_all_mass_bins.png

  Mass viz (STACK_PATH/mstar_viz/):
  - stack_ALL_mstar_{mlo}_{mhi}.fits   (1 row, full lambda grid; FITS only)

Custom M* x EW viz stacks (separate script, no bootstrap / no FastSpecFit):
  see stack_mstar_haew_viz.py -> stack_files/mstar_viz_haew/

Usage:
    python stack_mstar_haew_5pct.py
"""

import glob
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
_STACK_DIR = os.path.join(_CODE_DIR, "stacking_analysis")
for _p in (_CODE_DIR, _STACK_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pickle

import numpy as np

from stack_explore import (
    load_catalog,
    load_spectra,
    get_sample_spectra_with_linenorm,
    bootstrap_stack,
    mean_stack,
    write_stacked_spectra,
)


# =============================================================================
# CONFIG
# =============================================================================

SAMPLES = ["BGS_BRIGHT", "BGS_FAINT", "LOWZ"]
COMBINED_TAG = "ALL"

MSTAR_BINS = np.array([6.0, 6.5, 7.0, 7.5, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25])

# Fixed H-alpha EW bins (linear, Angstroms). Half-open (lo, hi].
EW_EDGES = [0.0, 30.0, 100.0, np.inf]
EW_TOKENS = ["ew_lt30", "ew_30_100", "ew_gt100"]
EW_LABELS = [
    r"EW $\leq$ 30 $\AA$",
    r"30 $<$ EW $\leq$ 100 $\AA$",
    r"EW $>$ 100 $\AA$",
]

EW_SNR_MIN = 3.0
EW_MIN = 1.0
BOXFLUX_SNR_MIN = 3.0
EW_STACK_NLIM = 50
Z_MIN_GLOBAL = 0.0
Z_MAX_GLOBAL = 0.5

N_BOOTSTRAP = 200
N_BOOT_SAVE = 200
RANDOM_SEED = 42

# Worker processes for the per-realization coadds inside bootstrap_stack
# (result-preserving: parallel results are bit-identical to serial). haew cells
# are EW-cut and usually small, so many fall below BOOT_PARALLEL_MIN_NVALID and
# run serially regardless; this only speeds up the larger cells. Safe within a
# full CPU node (--mem=0); run with OMP_NUM_THREADS=1 (the orchestrator sets it).
BOOT_NJOBS = 16

STACK_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_haew_5pct/"
MSTAR_ONLY_SUBDIR = "mstar_only"
MSTAR_ONLY_PATH = os.path.join(STACK_PATH, MSTAR_ONLY_SUBDIR)

# Visualization-only product: integer-centered 0.5-dex mass bins (+/-0.25 dex
# around log M* = 7, 8, 9). Same broad load_catalog sample + HALPHA_FLUX mean
# coadd as mstar_only, written to its OWN folder so the FastSpecFit (stage 2)
# and direct-metallicity (stage 3) steps never pick them up. FITS only (no
# pickle, no plot); these are for the user's own visualization.
MSTAR_VIZ_SUBDIR = "mstar_viz"
MSTAR_VIZ_PATH = os.path.join(STACK_PATH, MSTAR_VIZ_SUBDIR)
MSTAR_VIZ_CENTERS = [7.0, 8.0, 9.0]
MSTAR_VIZ_HALFWIDTH = 0.25
# Viz stacks keep the full red wavelength range (the science products trim to
# WAVE_MAX=6800). The de-redshift grid runs 3600->9800 A, so this keeps it all.
WAVE_MAX_VIZ = 9800

OVERWRITE_STACKS = True
WAVE_MAX = 6800
NORM_METHOD = "catalog"
# Per-galaxy normalization line flux. Gaussian HALPHA_FLUX (not boxcar
# HALPHA_BOXFLUX): FLUX is the more reliable line-flux measurement, and since
# every load_catalog galaxy already passes HALPHA_FLUX > 1, normalizing by it
# drops no galaxies (removes the old hidden HALPHA_BOXFLUX > 0 cut). The
# normalization constant cancels in all downstream line ratios; only the
# absolute stack flux scale changes relative to the old boxcar normalization.
NORM_COL = "HALPHA_FLUX"

LINE_GUIDES = {
    "[OII]": 3727.0,
    r"H$\beta$": 4862.7,
    "[OIII]": 5008.2,
    r"H$\alpha$": 6564.6,
}

EW_COLORS = {
    "ew_lt30": "#1f77b4",
    "ew_30_100": "#ff7f0e",
    "ew_gt100": "#d62728",
}


# =============================================================================
# HELPERS
# =============================================================================

def apply_halpha_detection_cuts(
    catalog,
    ew_snr_min=EW_SNR_MIN,
    ew_min=EW_MIN,
    boxflux_snr_min=BOXFLUX_SNR_MIN,
):
    """Keep galaxies passing H-alpha EW and boxflux detection cuts."""
    ha_ew = np.asarray(catalog["HALPHA_EW"])
    ha_ew_ivar = np.asarray(catalog["HALPHA_EW_IVAR"])
    ha_box = np.asarray(catalog["HALPHA_BOXFLUX"])
    ha_box_ivar = np.asarray(catalog["HALPHA_BOXFLUX_IVAR"])
    with np.errstate(invalid="ignore"):
        ew_snr = ha_ew * np.sqrt(ha_ew_ivar)
        box_snr = ha_box * np.sqrt(ha_box_ivar)
    mask = (
        np.isfinite(ha_ew) & (ha_ew > ew_min)
        & np.isfinite(ha_ew_ivar) & (ha_ew_ivar > 0)
        & np.isfinite(ew_snr) & (ew_snr >= ew_snr_min)
        & np.isfinite(ha_box) & (ha_box > 0)
        & np.isfinite(ha_box_ivar) & (ha_box_ivar > 0)
        & np.isfinite(box_snr) & (box_snr >= boxflux_snr_min)
    )
    return catalog[mask]


def _sample_mask(catalog, sample_names, z_min, z_max, logmstar_min, logmstar_max):
    sample_col = catalog["SAMPLE"]
    samp_mask = np.zeros(len(catalog), dtype=bool)
    for name in sample_names:
        samp_mask |= (sample_col == name)
    return (
        samp_mask
        & (catalog["Z"] > z_min) & (catalog["Z"] < z_max)
        & (catalog["LOG_MSTAR_M24"] > logmstar_min)
        & (catalog["LOG_MSTAR_M24"] <= logmstar_max)
    )


def select_sample_mstar_bin(
    catalog, sample_names,
    z_min, z_max,
    logmstar_min, logmstar_max,
):
    """Select galaxies in a (samples, z, log M*) cell (no EW cut)."""
    return catalog[_sample_mask(
        catalog, sample_names, z_min, z_max, logmstar_min, logmstar_max,
    )]


def select_sample_ew_bin(
    catalog, sample_names,
    z_min, z_max,
    logmstar_min, logmstar_max,
    ew_min, ew_max,
):
    """Select galaxies in a (samples, z, log M*, linear Halpha-EW) cell."""
    halpha_ew = np.asarray(catalog["HALPHA_EW"])
    mask = (
        _sample_mask(catalog, sample_names, z_min, z_max, logmstar_min, logmstar_max)
        & (halpha_ew > ew_min)
        & (halpha_ew <= ew_max)
    )
    return catalog[mask]


def bin_label(mstar_min, mstar_max, ew_token):
    """Filename-safe label for one (mass, EW) bin."""
    return f"mstar_{mstar_min:.2f}_{mstar_max:.2f}_{ew_token}"


def plot_label_for_token(token):
    """Human-readable legend label for a stack token."""
    if token in EW_TOKENS:
        return EW_LABELS[EW_TOKENS.index(token)]
    return token


def ew_bin_center(token):
    """Reference EW center (Angstrom) for plot ordering."""
    j = EW_TOKENS.index(token)
    lo, hi = EW_EDGES[j], EW_EDGES[j + 1]
    if np.isfinite(hi):
        return 0.5 * (lo + hi)
    return lo + 0.5 * (EW_EDGES[j] - EW_EDGES[j - 1])


def bin_seed_index(mstar_min, mstar_max, ew_token):
    """Stable per-bin RNG seed offset from bin definition (not loop order)."""
    i_m = int(np.searchsorted(MSTAR_BINS, mstar_min, side="left"))
    j_ew = EW_TOKENS.index(ew_token)
    return i_m * len(EW_TOKENS) + j_ew


def bin_label_mstar_only(mstar_min, mstar_max):
    """Filename-safe label for one mass-only bin (no EW token)."""
    return f"mstar_{mstar_min:.2f}_{mstar_max:.2f}"


def clean_stack_outputs(stack_path, extra_paths=()):
    """Remove previous stack, pickle, and FastSpec output files before a fresh run.

    Cleans ``stack_path`` plus any directories in ``extra_paths`` (e.g. the
    mstar_only/ and mstar_viz/ subfolders). Missing directories are skipped
    (glob on a nonexistent dir yields nothing).
    """
    patterns = (
        "stack_ALL_mstar_*.fits",
        "stacks_spec_ALL_mstar_*.pkl",
        "fastspec_stack_ALL_mstar_*.fits",
    )
    for path in (stack_path, *extra_paths):
        n_removed = 0
        for pattern in patterns:
            for fpath in glob.glob(os.path.join(path, pattern)):
                os.remove(fpath)
                n_removed += 1
        print(f"    Removed {n_removed} previous stack/.pkl/fastspec files from {path}")


def stack_one_bin(sub_cat, spectra_data, wave, token, ew_min, ew_max_fits,
                  mstar_min, mstar_max, label, bin_index):
    """Bootstrap-stack one bin; return saved dict or None."""
    n_sub = len(sub_cat)
    if n_sub == 0:
        return None

    pkl_path = os.path.join(
        STACK_PATH,
        f"stacks_spec_{COMBINED_TAG}_{label}.pkl",
    )

    if os.path.exists(pkl_path) and not OVERWRITE_STACKS:
        print(f"      Loading cached: {os.path.basename(pkl_path)}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

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
        "stack_spec":    central_flux,
        "stack_err":     boot_std,
        "central_ivar":  central_ivar,
        "real_flux":     real_flux,
        "real_ivar":     real_ivar,
        "all_stacks":    real_flux,
        "samples":       list(SAMPLES),
        "mstar_min":     float(mstar_min),
        "mstar_max":     float(mstar_max),
        "ew_min":        float(ew_min),
        "ew_max":        float(ew_max_fits),
        "ew_token":      token,
        "z_min":         Z_MIN_GLOBAL,
        "z_max":         Z_MAX_GLOBAL,
        "n_galaxies":    int(n_sub),
        "n_matched":     int(n_matched),
        "tgids":         np.asarray(tgids_matched),
        "bin_index":     int(bin_index),
        "random_seed":   int(seed),
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(saved, f)
    print(f"      Saved {os.path.basename(pkl_path)}")
    return saved


def write_multi_row_fits(saved, wave_for_fits, label):
    """Write 1 central + N_BOOT_SAVE bootstrap rows for FastSpecFit stackfit."""
    central_flux = saved["stack_spec"]
    central_ivar = saved["central_ivar"]
    real_flux = saved["real_flux"]
    real_ivar = saved["real_ivar"]
    # NOBJ is the number of spectra actually stacked (matched to a spectrum),
    # which can be < the catalog cell count when spectra are missing. The
    # N>=50 stack gate is on the catalog count (n_cat), kept here as NCAT for
    # provenance so the true stacked N is never misrepresented.
    n_matched = saved["n_matched"]
    n_cat = saved["n_galaxies"]
    mstar_min = saved["mstar_min"]
    mstar_max = saved["mstar_max"]
    ew_min = saved["ew_min"]
    ew_max_fits = saved["ew_max"]

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

    out_fits = os.path.join(STACK_PATH, f"stack_{COMBINED_TAG}_{label}.fits")

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
            "EW_MIN":    np.full(n_rows, ew_min, dtype=np.float32),
            "EW_MAX":    np.full(n_rows, ew_max_fits, dtype=np.float32),
        },
        table_format_dict={
            "IS_MEAN":   "K",
            "NOBJ":      "K",
            "NCAT":      "K",
            "MSTAR_MIN": "E",
            "MSTAR_MAX": "E",
            "EW_MIN":    "E",
            "EW_MAX":    "E",
        },
    )
    print(f"    {COMBINED_TAG} | {label}: "
          f"N_stacked={n_matched} (catalog N={n_cat}), "
          f"1 mean + {n_boot_keep} bootstraps "
          f"-> {os.path.basename(out_fits)}")


def stack_one_mass_bin(sub_cat, spectra_data, wave, mstar_min, mstar_max, label,
                       out_dir=MSTAR_ONLY_PATH, write_pkl=True):
    """Mean-stack one mass bin (no bootstrap); return saved dict or None.

    ``out_dir`` is where the pickle is written; ``write_pkl=False`` skips the
    pickle entirely (used by the FITS-only mstar_viz product).
    """
    n_sub = len(sub_cat)
    if n_sub == 0:
        return None

    pkl_path = os.path.join(
        out_dir,
        f"stacks_spec_{COMBINED_TAG}_{label}.pkl",
    )

    if write_pkl and os.path.exists(pkl_path) and not OVERWRITE_STACKS:
        print(f"      Loading cached: {os.path.basename(pkl_path)}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    out = get_sample_spectra_with_linenorm(
        sub_cat, spectra_data, line_norm="HALPHA", norm_col=NORM_COL,
    )
    fluxes, ivars, halpha_fluxes, tgids_matched = out

    if fluxes is None or len(fluxes) == 0:
        print("      No matched spectra; skipping.")
        return None

    n_matched = len(fluxes)
    print(f"      Mean-stacking: N={n_sub}, matched={n_matched}")
    stack_flux, stack_ivar, n_valid = mean_stack(
        fluxes=fluxes,
        ivars=ivars,
        wave=wave,
        norm_method=NORM_METHOD,
        catalog_line_fluxes=halpha_fluxes,
        min_n_valid=1,
    )

    if stack_flux is None:
        print("      mean_stack returned None; skipping.")
        return None

    saved = {
        "stack_spec":   stack_flux,
        "stack_ivar":   stack_ivar,
        "samples":      list(SAMPLES),
        "mstar_min":    float(mstar_min),
        "mstar_max":    float(mstar_max),
        "z_min":        Z_MIN_GLOBAL,
        "z_max":        Z_MAX_GLOBAL,
        "n_galaxies":   int(n_sub),
        "n_matched":    int(n_matched),
        "n_stacked":    int(n_valid),
        "tgids":        np.asarray(tgids_matched),
    }

    if write_pkl:
        with open(pkl_path, "wb") as f:
            pickle.dump(saved, f)
        print(f"      Saved {os.path.basename(pkl_path)}")
    return saved


def write_single_row_fits(saved, wave_for_fits, label, out_dir=MSTAR_ONLY_PATH):
    """Write one mean stack row to a FastSpecFit-compatible FITS in ``out_dir``."""
    stack_flux = saved["stack_spec"]
    stack_ivar = saved["stack_ivar"]
    n_stacked = saved.get("n_stacked", saved["n_matched"])
    n_cat = saved["n_galaxies"]
    mstar_min = saved["mstar_min"]
    mstar_max = saved["mstar_max"]

    all_flux = np.asarray(stack_flux, dtype=np.float32)[None, :]
    all_ivar = np.asarray(stack_ivar, dtype=np.float32)[None, :]

    out_fits = os.path.join(out_dir, f"stack_{COMBINED_TAG}_{label}.fits")

    write_stacked_spectra(
        outfile=out_fits,
        wave=wave_for_fits,
        flux=all_flux,
        ivar=all_ivar,
        stackids=np.array([0], dtype=np.int64),
        stack_redshift=np.zeros(1),
        table_column_dict={
            "IS_MEAN":   np.array([1], dtype=np.int64),
            "NOBJ":      np.array([n_stacked], dtype=np.int64),
            "NCAT":      np.array([n_cat], dtype=np.int64),
            "MSTAR_MIN": np.array([mstar_min], dtype=np.float32),
            "MSTAR_MAX": np.array([mstar_max], dtype=np.float32),
        },
        table_format_dict={
            "IS_MEAN":   "K",
            "NOBJ":      "K",
            "NCAT":      "K",
            "MSTAR_MIN": "E",
            "MSTAR_MAX": "E",
        },
    )
    print(f"    {COMBINED_TAG} | {label}: "
          f"N_stacked={n_stacked} (catalog N={n_cat}), "
          f"1 mean row -> {os.path.basename(out_fits)}")



def make_ivar_diagnostic_plots(results, wave, plot_dir, max_bins=2):
    """Plot propagated measurement error vs bootstrap std (diagnostic)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_done = 0
    for saved in results.values():
        if saved is None or n_done >= max_bins:
            continue
        if "central_ivar" not in saved:
            continue

        central_ivar = np.asarray(saved["central_ivar"], dtype=float)
        boot_std = np.asarray(saved["stack_err"], dtype=float)
        with np.errstate(invalid="ignore"):
            meas_err = np.where(central_ivar > 0, 1.0 / np.sqrt(central_ivar), np.nan)

        label = bin_label(saved["mstar_min"], saved["mstar_max"], saved["ew_token"])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wave, meas_err, color="C0", lw=1.0, label=r"$1/\sqrt{\mathrm{ivar}}$ (measurement)")
        ax.plot(wave, boot_std, color="C1", lw=1.0, alpha=0.8, label="bootstrap std (diagnostic)")
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
# PLOTTING
# =============================================================================

def _add_line_guides(ax):
    for lam in LINE_GUIDES.values():
        ax.axvline(lam, color="grey", ls=":", lw=0.8, alpha=0.6, zorder=0)


def _stacks_for_mass_bin(results, i_mstar):
    """Return (token, saved) pairs in one mass bin, sorted by EW bin center."""
    out = []
    for (i, token), saved in results.items():
        if i == i_mstar and saved is not None:
            out.append((token, saved))
    return sorted(out, key=lambda x: ew_bin_center(x[0]))


def _stack_by_token_for_mass_bin(results, i_mstar):
    """Map EW token -> saved dict for non-None stacks in one mass bin."""
    return {
        token: saved
        for (i, token), saved in results.items()
        if i == i_mstar and saved is not None
    }


def make_overlay_plots(results, wave, mstar_bins, plot_dir):
    """One panel per stellar-mass bin overlaying whatever stacks exist."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1

    for i in range(n_mstar):
        m_lo, m_hi = mstar_bins[i], mstar_bins[i + 1]
        stacks = _stacks_for_mass_bin(results, i)
        if not stacks:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        for token, saved in stacks:
            flux = saved["stack_spec"]
            err = saved["stack_err"]
            n_gal = saved["n_galaxies"]
            color = EW_COLORS.get(token, "k")
            label = plot_label_for_token(token)
            ax.plot(wave, flux, color=color, lw=1.0,
                    label=f"{label}  (N={n_gal})")
            ax.fill_between(
                wave, flux - err, flux + err,
                color=color, alpha=0.20, lw=0,
            )

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
    """Grid: rows = mass bins, columns = EW bins (fixed, increasing EW center)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1
    n_ew = len(EW_TOKENS)
    if not any(_stacks_for_mass_bin(results, i) for i in range(n_mstar)):
        print("    (no stacks; skipping grid_all_stacks)")
        return

    fig, axes = plt.subplots(
        n_mstar, n_ew,
        figsize=(3.5 * n_ew, 2.4 * n_mstar),
        sharex=True, squeeze=False,
    )

    for i in range(n_mstar):
        m_lo, m_hi = mstar_bins[i], mstar_bins[i + 1]
        by_token = _stack_by_token_for_mass_bin(results, i)
        for j, token in enumerate(EW_TOKENS):
            ax = axes[i][j]
            saved = by_token.get(token)
            if saved is not None:
                flux = saved["stack_spec"]
                err = saved["stack_err"]
                color = EW_COLORS.get(token, "k")
                ax.plot(wave, flux, color=color, lw=0.8)
                ax.fill_between(
                    wave, flux - err, flux + err,
                    color=color, alpha=0.20, lw=0,
                )
                ax.text(
                    0.97, 0.92, f"N={saved['n_galaxies']}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=7,
                )
                _add_line_guides(ax)
                ax.set_xlim(wave.min(), wave.max())
            else:
                ax.axis("off")
            if i == 0:
                ax.set_title(plot_label_for_token(token), fontsize=7)
            if j == 0:
                ax.set_ylabel(f"[{m_lo:.1f},{m_hi:.1f}]", fontsize=9)
            if i == n_mstar - 1:
                ax.set_xlabel(r"Rest $\lambda$ [$\AA$]", fontsize=8)

    fig.suptitle("Halpha-normalized stacks (fixed EW bins per mass bin)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_png = os.path.join(plot_dir, "grid_all_stacks.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


def make_mass_only_overlay_plot(results, wave, mstar_bins, plot_dir):
    """Overlay all mass-only stacks in one panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    stacks = [(i, saved) for i, saved in results.items() if saved is not None]
    if not stacks:
        print("    (no mass-only stacks; skipping overlay_all_mass_bins)")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(stacks)))

    for k, (i, saved) in enumerate(stacks):
        m_lo = saved["mstar_min"]
        m_hi = saved["mstar_max"]
        flux = saved["stack_spec"]
        ivar = saved["stack_ivar"]
        n_gal = saved["n_galaxies"]
        with np.errstate(invalid="ignore"):
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
        ax.plot(wave, flux, color=cmap[k], lw=1.0,
                label=f"[{m_lo:.2f}, {m_hi:.2f}]  (N={n_gal})")
        ax.fill_between(wave, flux - err, flux + err, color=cmap[k], alpha=0.15, lw=0)

    _add_line_guides(ax)
    ax.set_xlim(wave.min(), wave.max())
    ax.set_xlabel(r"Rest wavelength [$\AA$]")
    ax.set_ylabel("Halpha-normalized stacked flux")
    ax.set_title("Mass-only stacks (all bins, no EW split)")
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()

    out_png = os.path.join(plot_dir, "overlay_all_mass_bins.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


def run_mass_viz_stacks(tot_cat_full, spectra_data, wave_viz):
    """FITS-only mean stacks in integer-centered 0.5-dex mass bins (viz only).

    Bins: (center +/- MSTAR_VIZ_HALFWIDTH] for each center in MSTAR_VIZ_CENTERS
    (half-open, low-exclusive, like the rest of the pipeline). Same broad
    load_catalog sample and HALPHA_FLUX mean coadd as the mstar_only product,
    but written to MSTAR_VIZ_PATH on the FULL (untrimmed) wavelength grid. No
    pickle, no plot, and never globbed by the FastSpecFit / direct-metallicity
    stages (separate folder).
    """
    print(f"\n[2c] Mass viz stacks: {len(MSTAR_VIZ_CENTERS)} integer-centered "
          f"+/-{MSTAR_VIZ_HALFWIDTH:.2f} dex bins, mean coadd, FITS only")
    print(f"     lambda in [{wave_viz.min():.1f}, {wave_viz.max():.1f}] A "
          f"({len(wave_viz)} pixels); output dir: {MSTAR_VIZ_PATH}")

    n_written = 0
    for center in MSTAR_VIZ_CENTERS:
        mstar_min = center - MSTAR_VIZ_HALFWIDTH
        mstar_max = center + MSTAR_VIZ_HALFWIDTH
        sub_cat = select_sample_mstar_bin(
            tot_cat_full, SAMPLES,
            Z_MIN_GLOBAL, Z_MAX_GLOBAL,
            mstar_min, mstar_max,
        )
        n_sub = len(sub_cat)
        print(f"\n  --- center log M*={center:.2f} -> "
              f"({mstar_min:.2f}, {mstar_max:.2f}] --- N={n_sub}")
        if n_sub == 0:
            continue

        label = bin_label_mstar_only(mstar_min, mstar_max)
        saved = stack_one_mass_bin(
            sub_cat, spectra_data, wave_viz, mstar_min, mstar_max, label,
            out_dir=MSTAR_VIZ_PATH, write_pkl=False,
        )
        if saved is None:
            continue
        write_single_row_fits(saved, wave_viz, label, out_dir=MSTAR_VIZ_PATH)
        n_written += 1

    print(f"\n[2c] Wrote {n_written} mass-viz FITS files to {MSTAR_VIZ_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(STACK_PATH, exist_ok=True)
    os.makedirs(MSTAR_ONLY_PATH, exist_ok=True)
    os.makedirs(MSTAR_VIZ_PATH, exist_ok=True)
    plot_dir = os.path.join(STACK_PATH, "plots")
    mass_only_plot_dir = os.path.join(MSTAR_ONLY_PATH, "plots")

    print("[0] Cleaning previous stack outputs ...")
    clean_stack_outputs(STACK_PATH, extra_paths=[MSTAR_ONLY_PATH, MSTAR_VIZ_PATH])

    print("[1] Loading catalog ...")
    tot_cat_full = load_catalog()
    print(f"    Total galaxies after load_catalog cuts: {len(tot_cat_full)}")

    tot_cat_ew = apply_halpha_detection_cuts(tot_cat_full)
    print(f"    After H-alpha detection cuts (EW stacks only) "
          f"(EW SNR>={EW_SNR_MIN:.0f}, EW>{EW_MIN:.0f} A, "
          f"boxflux SNR>={BOXFLUX_SNR_MIN:.0f}): {len(tot_cat_ew)}")

    print(f"\n    Fixed EW bins (half-open, Angstroms):")
    for token, label in zip(EW_TOKENS, EW_LABELS):
        print(f"      {token}: {label}")
    print(f"    Stack minimum per EW cell: N >= {EW_STACK_NLIM}")

    print("\n[2] Loading de-redshifted spectra ...")
    spectra_data = load_spectra(
        "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/"
        "desi_y1_dwarf_combine_deredshift_hires_noinvvar.h5"
    )
    print(f"    Total spectra loaded: {len(spectra_data['targetid'])}")

    # Mass viz stacks need the full red range (lambda < WAVE_MAX_VIZ), so build
    # them BEFORE the in-place WAVE_MAX trim below. The de-redshift grid already
    # ends below WAVE_MAX_VIZ, so the full loaded arrays are used directly (no
    # copy); only slice if a future grid extends past WAVE_MAX_VIZ.
    viz_mask = spectra_data["wave_rest"] < WAVE_MAX_VIZ
    if viz_mask.all():
        run_mass_viz_stacks(tot_cat_full, spectra_data, spectra_data["wave_rest"])
    else:
        viz_spec = {
            "targetid":  spectra_data["targetid"],
            "wave_rest": spectra_data["wave_rest"][viz_mask],
            "flux":      spectra_data["flux"][:, viz_mask],
            "flux_ivar": spectra_data["flux_ivar"][:, viz_mask],
        }
        run_mass_viz_stacks(tot_cat_full, viz_spec, viz_spec["wave_rest"])
        del viz_spec

    wave_mask = spectra_data["wave_rest"] < WAVE_MAX
    spectra_data["wave_rest"] = spectra_data["wave_rest"][wave_mask]
    spectra_data["flux"]      = spectra_data["flux"][:, wave_mask]
    spectra_data["flux_ivar"] = spectra_data["flux_ivar"][:, wave_mask]
    wave = spectra_data["wave_rest"]
    print(f"    After trim: lambda in [{wave.min():.1f}, {wave.max():.1f}] A"
          f"  ({len(wave)} pixels)")

    n_mstar = len(MSTAR_BINS) - 1
    n_ew = len(EW_TOKENS)
    print(f"\n[3] Stacking: {n_mstar} mstar bins x {n_ew} fixed EW bins "
          f"(stack when N>={EW_STACK_NLIM})")
    print(f"    M* edges  : {MSTAR_BINS}")
    print(f"    EW edges  : {EW_EDGES}")
    print(f"    Output dir: {STACK_PATH}")

    results = {}

    print(f"\n========== Combined sample: {'|'.join(SAMPLES)} ==========")

    for i in range(n_mstar):
        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]

        sub_all = select_sample_mstar_bin(
            tot_cat_ew, SAMPLES,
            Z_MIN_GLOBAL, Z_MAX_GLOBAL,
            mstar_min, mstar_max,
        )
        n_total = len(sub_all)

        ew_counts = []
        for j in range(n_ew):
            sub = select_sample_ew_bin(
                tot_cat_ew, SAMPLES,
                Z_MIN_GLOBAL, Z_MAX_GLOBAL,
                mstar_min, mstar_max,
                EW_EDGES[j], EW_EDGES[j + 1],
            )
            ew_counts.append(len(sub))

        counts_str = ", ".join(
            f"{EW_TOKENS[j]}={ew_counts[j]}" for j in range(n_ew)
        )
        print(f"\n  --- log M*=[{mstar_min:.2f},{mstar_max:.2f}] ---")
        print(f"      N_total={n_total}")
        print(f"      EW counts: {counts_str}  (sum={sum(ew_counts)})")
        if sum(ew_counts) != n_total:
            print(f"      WARNING: EW bin sum ({sum(ew_counts)}) != N_total ({n_total})")

        for j, token in enumerate(EW_TOKENS):
            if ew_counts[j] < EW_STACK_NLIM:
                print(f"      {token}: N={ew_counts[j]} < {EW_STACK_NLIM}; skipping")
                continue

            ew_min = EW_EDGES[j]
            ew_max = EW_EDGES[j + 1]
            ew_max_fits = float(ew_max) if np.isfinite(ew_max) else -1.0
            sub_cat = select_sample_ew_bin(
                tot_cat_ew, SAMPLES,
                Z_MIN_GLOBAL, Z_MAX_GLOBAL,
                mstar_min, mstar_max,
                ew_min, ew_max,
            )
            label = bin_label(mstar_min, mstar_max, token)
            ew_hi_str = f"{ew_max_fits:.1f}" if ew_max_fits >= 0 else "inf"
            print(f"\n    >> {token} | EW in ({ew_min:.1f}, {ew_hi_str}] | "
                  f"N={len(sub_cat)}")

            bin_index = bin_seed_index(mstar_min, mstar_max, token)
            saved = stack_one_bin(
                sub_cat, spectra_data, wave, token, float(ew_min), ew_max_fits,
                mstar_min, mstar_max, label, bin_index,
            )
            results[(i, token)] = saved

    print("\n[4] Writing FastSpecFit (stackfit) input FITS files ...")

    n_written = 0
    for (i, token), saved in results.items():
        if saved is None:
            continue
        label = bin_label(saved["mstar_min"], saved["mstar_max"], token)
        write_multi_row_fits(saved, wave, label)
        n_written += 1

    print(f"\n[5] Wrote {n_written} EW-binned FITS files to {STACK_PATH}")

    print(f"\n[3b] Mass-only stacking: {n_mstar} bins, no EW split, no bootstrap")
    print(f"     Catalog: load_catalog cuts only (N={len(tot_cat_full)})")
    print(f"     Output dir: {MSTAR_ONLY_PATH}")

    mass_only_results = {}

    for i in range(n_mstar):
        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]
        sub_cat = select_sample_mstar_bin(
            tot_cat_full, SAMPLES,
            Z_MIN_GLOBAL, Z_MAX_GLOBAL,
            mstar_min, mstar_max,
        )
        n_sub = len(sub_cat)
        print(f"\n  --- log M*=[{mstar_min:.2f},{mstar_max:.2f}] --- N={n_sub}")
        if n_sub == 0:
            continue

        label = bin_label_mstar_only(mstar_min, mstar_max)
        saved = stack_one_mass_bin(
            sub_cat, spectra_data, wave, mstar_min, mstar_max, label,
        )
        mass_only_results[i] = saved

    print("\n[4b] Writing mass-only FastSpecFit (stackfit) input FITS files ...")

    n_mo_written = 0
    for i, saved in mass_only_results.items():
        if saved is None:
            continue
        label = bin_label_mstar_only(saved["mstar_min"], saved["mstar_max"])
        write_single_row_fits(saved, wave, label)
        n_mo_written += 1

    print(f"\n[5b] Wrote {n_mo_written} mass-only FITS files to {MSTAR_ONLY_PATH}")

    print("\n[6] Making EW-binned comparison plots ...")
    make_overlay_plots(results, wave, MSTAR_BINS, plot_dir)
    make_grid_plot(results, wave, MSTAR_BINS, plot_dir)

    print("\n[7] ivar vs bootstrap-std diagnostic plots (EW stacks) ...")
    make_ivar_diagnostic_plots(results, wave, plot_dir, max_bins=2)

    print("\n[8] Making mass-only overlay plot ...")
    make_mass_only_overlay_plot(mass_only_results, wave, MSTAR_BINS, mass_only_plot_dir)

    print(f"\n[9] Done. EW plots in {plot_dir}; mass-only plots in {mass_only_plot_dir}")


if __name__ == "__main__":
    main()
