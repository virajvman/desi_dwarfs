"""
stack_mstar_haew.py
===================

Bootstrap-stacked spectra in 2D bins of (log M_star, log10 H-alpha EW)
combining BGS_BRIGHT, BGS_FAINT, and LOWZ into a single stacked sample.

This is a direct extension of the M*-only stacking pipeline in
`elg_property_explore.ipynb` / `elg_explore.py`:

  - select_sample is generalized to also cut on log10(HALPHA_EW), and
    accepts a list of SAMPLE values so that galaxies from BGS_BRIGHT,
    BGS_FAINT, and LOWZ are pooled together in each (M*, EW) cell.
  - bootstrap_stack is reused unchanged (200 realizations, draws WITH
    replacement; per-realization mean is one row of `all_stacks`)
  - write_stacked_spectra is reused to dump 1 mean + N_BOOT_SAVE
    bootstrap realizations into a FastSpecFit-compatible FITS file

The bootstrap with replacement *is* the error model: when stackfit
runs on the FITS file you get one line measurement for the mean row
and N_BOOT_SAVE measurements for the bootstrap rows; the 16/84
percentile across the bootstrap rows gives you the error on every
derived line ratio / EW. No extra error machinery is needed.

Outputs (one per mstar-bin x ew-bin, combined across all samples):
  - stacks_spec_ALL_mstar_{mlo}_{mhi}_logew_{elo}_{ehi}.pkl
    (stack_spec, stack_err, all_stacks, bin metadata)
  - stack_ALL_mstar_{mlo}_{mhi}_logew_{elo}_{ehi}.fits
    (FastSpecFit stackfit input: row 0 = mean, rows 1.. = bootstraps)

Usage:
    python stack_mstar_haew.py
"""

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
    get_sample_spectra_with_linenorm,
    bootstrap_stack,
    write_stacked_spectra,
)


# =============================================================================
# CONFIG
# =============================================================================

# Samples to pool into the combined stack. Galaxies whose SAMPLE matches
# any entry are pooled together (SAMPLE == BGS_BRIGHT | BGS_FAINT | LOWZ).
SAMPLES = ["BGS_BRIGHT", "BGS_FAINT", "LOWZ"]

# Tag used in output filenames for the combined-sample stack.
COMBINED_TAG = "ALL"

# Stellar-mass bin edges (log Msun). 0.5 dex bins.
# Coarser than the 0.25 dex used in the M*-only analysis because going 2D
# multiplies the number of bins.
MSTAR_BINS = np.arange(6.0, 9.5 + 1e-6, 0.5)
#   -> [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]   (7 bins)

# H-alpha EW binning. LOG bins because EW ranges over ~3 dex and is
# strongly skewed; linear bins would dump ~80% of the sample into the
# first bin. Edges are in log10(EW [A]).
LOG_EW_BINS = np.arange(0.0, 3.0 + 1e-6, 0.5)
#   -> [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]    (6 bins)
#   -> EW [A]: [1, 3.16, 10, 31.6, 100, 316, 1000]
#
# If you'd rather use linear bins (e.g. 4 bins), do something like:
#     USE_LOG_EW = False
#     EW_BINS = np.array([0, 10, 30, 100, 1000.0])
# and adapt the cut in `select_sample_2d` below. The log scheme is the
# default because it produces more balanced bins.

# Redshift range. Same as the M*-only pipeline -- effectively no cut.
Z_MIN_GLOBAL = 0.0
Z_MAX_GLOBAL = 0.5

# Minimum number of galaxies in a bin to attempt a stack.
STACK_NLIM = 50

# Bootstrap settings (same defaults as the M*-only pipeline).
N_BOOTSTRAP = 200      # number of bootstrap realizations
N_DRAW      = 5000     # spectra per realization (capped at n_valid in-bin)
RANDOM_SEED = 42

# How many of the N_BOOTSTRAP realizations to save into the FITS file
# (these become rows 1.. and provide the error budget on stackfit-derived
# quantities).
N_BOOT_SAVE = 50

# Output location.
STACK_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_ew/"

# Re-do stacks even if a cached pickle already exists on disk?
OVERWRITE_STACKS = True

# Rest-frame wavelength upper bound (same as the M*-only pipeline).
WAVE_MAX = 6800

# Spectrum-normalization method used inside bootstrap_stack.
#
# "catalog"     - normalize each spectrum by its catalog HALPHA_FLUX. Same
#                 as the existing M*-only pipeline. Recommended default
#                 even when binning in EW: the EW bin only controls the
#                 line/continuum ratio, not the absolute flux level, soS
#                 bright/close galaxies will still dominate if you don't
#                 normalize.
# "boxcar_line" - normalize by a self-measured boxcar H-alpha flux.
#                 More self-consistent with on-the-stack measurements
#                 but slower.
# "flux_window" - normalize by integrated flux in a continuum window.
NORM_METHOD = "catalog"


# =============================================================================
# HELPERS
# =============================================================================

def select_sample_2d(
    catalog, sample_names,
    z_min, z_max,
    logmstar_min, logmstar_max,
    log_ew_min, log_ew_max,
):
    """Select galaxies in (samples, z, log M*, log10 Halpha-EW) cell.

    `sample_names` is an iterable of SAMPLE values that are pooled
    together (logical OR), e.g. ["BGS_BRIGHT", "BGS_FAINT", "LOWZ"].

    The mass cut is right-open ( > min,  <= max ) so that adjacent bins
    don't double-count their shared edge; same for the EW cut. The lowest
    bin's `min` is `>` not `>=`, which excludes galaxies sitting exactly
    on the bottom edge; this is fine for our purposes (no galaxy has
    HALPHA_EW exactly == 1.0 A in practice).
    """
    sample_col = np.asarray(catalog["SAMPLE"])
    samp_mask = np.zeros(len(catalog), dtype=bool)
    for name in sample_names:
        samp_mask |= (sample_col == name)

    halpha_ew = np.asarray(catalog["HALPHA_EW"])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ew = np.log10(halpha_ew)   # EW <= 0 -> -inf / nan, will fail bin cut

    mask = (
        samp_mask
        & (catalog["Z"] > z_min) & (catalog["Z"] < z_max)
        & (catalog["LOG_MSTAR_M24"] > logmstar_min)
        & (catalog["LOG_MSTAR_M24"] <= logmstar_max)
        & (log_ew > log_ew_min)
        & (log_ew <= log_ew_max)
    )
    return catalog[mask]


def bin_label(mstar_min, mstar_max, lew_min, lew_max):
    """Filename-safe label for one (mass, EW) bin."""
    return (
        f"mstar_{mstar_min:.2f}_{mstar_max:.2f}"
        f"_logew_{lew_min:.2f}_{lew_max:.2f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(STACK_PATH, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load catalog + spectra
    # -------------------------------------------------------------------------
    print("[1] Loading catalog ...")
    tot_cat = load_catalog()
    print(f"    Total galaxies after quality cuts: {len(tot_cat)}")

    print("\n[2] Loading de-redshifted spectra ...")
    spectra_data = load_spectra()
    print(f"    Total spectra loaded: {len(spectra_data['targetid'])}")

    # Trim to lambda < WAVE_MAX (consistent with the M*-only pipeline).
    wave_mask = spectra_data["wave_rest"] < WAVE_MAX
    spectra_data["wave_rest"] = spectra_data["wave_rest"][wave_mask]
    spectra_data["flux"]      = spectra_data["flux"][:, wave_mask]
    spectra_data["flux_ivar"] = spectra_data["flux_ivar"][:, wave_mask]
    wave = spectra_data["wave_rest"]
    print(f"    After trim: lambda in [{wave.min():.1f}, {wave.max():.1f}] A"
          f"  ({len(wave)} pixels)")

    # -------------------------------------------------------------------------
    # 2. Loop over mass-bins x EW-bins (samples are pooled together)
    # -------------------------------------------------------------------------
    n_mstar = len(MSTAR_BINS) - 1
    n_ew    = len(LOG_EW_BINS) - 1
    print(f"\n[3] Stacking grid: {n_mstar} mstar bins x {n_ew} EW bins"
          f" = {n_mstar * n_ew} candidate stacks"
          f"  (combining samples: {'|'.join(SAMPLES)})")
    print(f"    M* edges     : {MSTAR_BINS}")
    print(f"    log10(EW) eds: {LOG_EW_BINS}")
    print(f"    Output dir   : {STACK_PATH}")

    # results[(i_mstar, j_ew)] -> dict (or None if skipped)
    results = {}

    print(f"\n========== Combined sample: {'|'.join(SAMPLES)} ==========")

    for i in range(n_mstar):
        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]

        for j in range(n_ew):
            lew_min, lew_max = LOG_EW_BINS[j], LOG_EW_BINS[j + 1]
            label = bin_label(mstar_min, mstar_max, lew_min, lew_max)

            print(f"\n  --- {COMBINED_TAG} | log M*=[{mstar_min:.2f},{mstar_max:.2f}]"
                  f" | log10(EW)=[{lew_min:.2f},{lew_max:.2f}] ---")

            sub_cat = select_sample_2d(
                tot_cat, SAMPLES,
                Z_MIN_GLOBAL, Z_MAX_GLOBAL,
                mstar_min, mstar_max,
                lew_min, lew_max,
            )
            n_sub = len(sub_cat)
            print(f"      N galaxies in bin: {n_sub}")

            if n_sub < STACK_NLIM:
                print(f"      Skipping (< {STACK_NLIM} galaxies)")
                results[(i, j)] = None
                continue

            pkl_path = os.path.join(
                STACK_PATH,
                f"stacks_spec_{COMBINED_TAG}_{label}.pkl",
            )

            # ---- Either reuse cached pickle, or compute the stack ----
            if os.path.exists(pkl_path) and not OVERWRITE_STACKS:
                print(f"      Loading cached: {os.path.basename(pkl_path)}")
                with open(pkl_path, "rb") as f:
                    saved = pickle.load(f)
            else:
                # Match catalog rows to spectra by TARGETID and
                # pull the per-spectrum Halpha flux for normalization.
                out = get_sample_spectra_with_linenorm(
                    sub_cat, spectra_data, line_norm="HALPHA",
                )
                fluxes, ivars, halpha_fluxes, tgids_matched = out

                if fluxes is None or len(fluxes) < STACK_NLIM:
                    n_matched = 0 if fluxes is None else len(fluxes)
                    print(f"      Too few matched spectra "
                          f"(matched={n_matched} < {STACK_NLIM}); skipping.")
                    results[(i, j)] = None
                    continue

                # Bootstrap-stack:
                #   - normalize each spectrum (catalog Halpha by default)
                #   - draw N_DRAW indices WITH replacement, N_BOOTSTRAP times
                #   - stack_spec = mean of realizations, stack_err = std
                #   - all_stacks = (N_BOOTSTRAP, n_wave) realizations
                print(f"      Bootstrap-stacking: N={n_sub}, "
                      f"n_bootstrap={N_BOOTSTRAP}, n_draw={min(N_DRAW, n_sub)}")
                stack_spec, stack_err, all_stacks = bootstrap_stack(
                    fluxes=fluxes,
                    ivars=ivars,
                    wave=wave,
                    n_bootstrap=N_BOOTSTRAP,
                    n_draw=N_DRAW,
                    random_seed=RANDOM_SEED,
                    norm_method=NORM_METHOD,
                    catalog_line_fluxes=halpha_fluxes,
                )

                if all_stacks is None:
                    print(f"      bootstrap_stack returned None; skipping.")
                    results[(i, j)] = None
                    continue

                saved = {
                    "stack_spec":  stack_spec,
                    "stack_err":   stack_err,
                    "all_stacks":  all_stacks,
                    "samples":     list(SAMPLES),
                    "mstar_min":   float(mstar_min),
                    "mstar_max":   float(mstar_max),
                    "log_ew_min":  float(lew_min),
                    "log_ew_max":  float(lew_max),
                    "z_min":       Z_MIN_GLOBAL,
                    "z_max":       Z_MAX_GLOBAL,
                    "n_galaxies":  int(n_sub),
                    "n_matched":   int(len(fluxes)),
                    "tgids":       np.asarray(tgids_matched),
                }

                with open(pkl_path, "wb") as f:
                    pickle.dump(saved, f)
                print(f"      Saved {os.path.basename(pkl_path)}")

            results[(i, j)] = saved

    # -------------------------------------------------------------------------
    # 3. Write FastSpecFit-compatible FITS for each non-empty bin
    # -------------------------------------------------------------------------
    print("\n[4] Writing FastSpecFit (stackfit) input FITS files ...")

    wave_for_fits = wave  # already trimmed to < WAVE_MAX

    n_written = 0
    for (i, j), saved in results.items():
        if saved is None:
            continue

        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]
        lew_min,   lew_max   = LOG_EW_BINS[j], LOG_EW_BINS[j + 1]
        label = bin_label(mstar_min, mstar_max, lew_min, lew_max)

        flux_mean  = saved["stack_spec"]
        err_mean   = saved["stack_err"]
        all_stacks = saved["all_stacks"]
        n_galaxies = saved["n_galaxies"]

        # Variance -> ivar (zero where err is non-positive / non-finite)
        ivar_mean = np.where(
            np.isfinite(err_mean) & (err_mean > 0),
            1.0 / err_mean ** 2,
            0.0,
        )

        # Sub-sample of bootstrap realizations to dump (rows 1..N_BOOT_KEEP)
        rng = np.random.default_rng(RANDOM_SEED)
        n_boot_avail = len(all_stacks)
        n_boot_keep  = min(N_BOOT_SAVE, n_boot_avail)
        boot_idx     = rng.choice(n_boot_avail, size=n_boot_keep, replace=False)
        boot_stacks  = all_stacks[boot_idx]

        # Row 0 = mean; rows 1..n_boot_keep = bootstrap realizations
        n_rows   = 1 + n_boot_keep
        all_flux = np.zeros((n_rows, len(wave_for_fits)), dtype=np.float32)
        all_ivar = np.zeros((n_rows, len(wave_for_fits)), dtype=np.float32)

        all_flux[0] = np.where(np.isfinite(flux_mean), flux_mean, 0.0)
        all_ivar[0] = np.where(
            np.isfinite(ivar_mean) & (all_flux[0] != 0),
            ivar_mean, 0.0,
        )
        for k, bk in enumerate(boot_stacks, start=1):
            all_flux[k] = np.where(np.isfinite(bk), bk, 0.0)
            # Use the same ivar for every bootstrap row -- the spread
            # across rows is the actual error budget; per-row ivar is
            # only used by stackfit for chi^2 weighting on each row.
            all_ivar[k] = all_ivar[0]

        out_fits = os.path.join(
            STACK_PATH,
            f"stack_{COMBINED_TAG}_{label}.fits",
        )

        write_stacked_spectra(
            outfile=out_fits,
            wave=wave_for_fits,
            flux=all_flux,
            ivar=all_ivar,
            stackids=np.arange(n_rows),
            stack_redshift=np.zeros(n_rows),
            table_column_dict={
                "IS_MEAN":    np.array([1] + [0] * n_boot_keep, dtype=np.int64),
                "NOBJ":       np.full(n_rows, n_galaxies, dtype=np.int64),
                "MSTAR_MIN":  np.full(n_rows, mstar_min, dtype=np.float32),
                "MSTAR_MAX":  np.full(n_rows, mstar_max, dtype=np.float32),
                "LOG_EW_MIN": np.full(n_rows, lew_min,  dtype=np.float32),
                "LOG_EW_MAX": np.full(n_rows, lew_max,  dtype=np.float32),
            },
            table_format_dict={
                "IS_MEAN":    "K",
                "NOBJ":       "K",
                "MSTAR_MIN":  "E",
                "MSTAR_MAX":  "E",
                "LOG_EW_MIN": "E",
                "LOG_EW_MAX": "E",
            },
        )
        print(f"    {COMBINED_TAG} | {label}: "
              f"N_gal={n_galaxies}, 1 mean + {n_boot_keep} bootstraps "
              f"-> {os.path.basename(out_fits)}")
        n_written += 1

    print(f"\n[5] Done. Wrote {n_written} FITS files to {STACK_PATH}")


if __name__ == "__main__":
    main()