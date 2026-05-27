"""
stack_mstar_elg_vs_noelg.py
===========================

ELG vs non-ELG bootstrap-stacked spectra in 1D bins of log M_star.

This is the notebook's M*-only stacking pipeline lifted into a standalone
script, so the notebook can be reserved for plotting. The bootstrap
machinery (200 realizations of N_DRAW spectra sampled WITH replacement,
mean of realizations = stack, std of realizations = error spectrum,
realizations saved as rows 1..N_BOOT_SAVE of the FITS file for stackfit)
is identical to what the notebook was doing.

Filenames intentionally match what the notebook reads downstream:
  - stacks_spec_elg_mstar_{mlo}_{mhi}.pkl     (and ..._noelg_...)
  - stack_mstar_elg_{mlo}_{mhi}.fits          (and ..._noelg_...)
so the FastSpecFit-reading cells in the notebook keep working without
edits.

Usage:
    python stack_mstar_elg_vs_noelg.py
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

# Stellar-mass bin edges (log Msun). 0.25 dex bins -- same as the notebook.
MSTAR_BINS = np.arange(6.0, 9.5 + 1e-6, 0.25)

# Redshift range. Effectively no cut (we bin only in M*, as per the
# notebook's analysis plan).
Z_MIN_GLOBAL = 0.0
Z_MAX_GLOBAL = 0.5

# Minimum number of galaxies in a bin to attempt a stack.
STACK_NLIM = 50

# Bootstrap settings (same defaults as bootstrap_stack itself).
N_BOOTSTRAP = 200      # number of bootstrap realizations
N_DRAW      = 5000     # spectra per realization (capped at n_valid in-bin)
RANDOM_SEED = 42

# How many of the N_BOOTSTRAP realizations to save into the FITS file
# (these become rows 1.. and provide the error budget on stackfit-derived
# quantities).
N_BOOT_SAVE = 50

# Output location -- same as the notebook's mass-bin pipeline.
STACK_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar/"

# Re-do stacks even if a cached pickle already exists on disk?
OVERWRITE_STACKS = True

# Rest-frame wavelength upper bound (consistent with the notebook).
WAVE_MAX = 6800

# Spectrum-normalization method passed into bootstrap_stack.
#   "catalog"     - normalize each spectrum by its catalog HALPHA_FLUX
#                   (notebook default; recommended)
#   "boxcar_line" - normalize by a self-measured boxcar Halpha flux
#   "flux_window" - normalize by integrated flux in a continuum window
NORM_METHOD = "catalog"


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

    # Trim to lambda < WAVE_MAX (consistent with the notebook).
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
          f" = {len(SAMPLE_SPECS) * n_mstar} candidate stacks")
    print(f"    M* edges  : {MSTAR_BINS}")
    print(f"    Output dir: {STACK_PATH}")

    # results[file_key][i_mstar] -> dict (or None if skipped)
    results = {file_key: {} for _, file_key in SAMPLE_SPECS}

    for i in range(n_mstar):
        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]
        print(f"\n=== log M* in [{mstar_min:.2f}, {mstar_max:.2f}] ===")

        for sample_name, file_key in SAMPLE_SPECS:
            print(f"\n  --- {sample_name} ---")

            sub_cat = select_sample(
                tot_cat, sample_name,
                z_min=Z_MIN_GLOBAL, z_max=Z_MAX_GLOBAL,
                logmstar_min=mstar_min, logmstar_max=mstar_max,
            )
            n_sub = len(sub_cat)
            print(f"      N galaxies in bin: {n_sub}")

            if n_sub < STACK_NLIM:
                print(f"      Skipping (< {STACK_NLIM} galaxies)")
                results[file_key][i] = None
                continue

            pkl_path = os.path.join(
                STACK_PATH,
                f"stacks_spec_{file_key}_mstar_{mstar_min:.2f}_{mstar_max:.2f}.pkl",
            )

            # ---- Either reuse cached pickle, or compute the stack ----
            if os.path.exists(pkl_path) and not OVERWRITE_STACKS:
                print(f"      Loading cached: {os.path.basename(pkl_path)}")
                with open(pkl_path, "rb") as f:
                    saved = pickle.load(f)
            else:
                # Match catalog rows to spectra by TARGETID and pull the
                # per-spectrum Halpha flux for normalization.
                out = get_sample_spectra_with_linenorm(
                    sub_cat, spectra_data, line_norm="HALPHA",
                )
                fluxes, ivars, halpha_fluxes, tgids_matched = out

                if fluxes is None or len(fluxes) < STACK_NLIM:
                    n_matched = 0 if fluxes is None else len(fluxes)
                    print(f"      Too few matched spectra "
                          f"(matched={n_matched} < {STACK_NLIM}); skipping.")
                    results[file_key][i] = None
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
                    results[file_key][i] = None
                    continue

                saved = {
                    "stack_spec":  stack_spec,
                    "stack_err":   stack_err,
                    "all_stacks":  all_stacks,
                    "sample":      sample_name,
                    "mstar_min":   float(mstar_min),
                    "mstar_max":   float(mstar_max),
                    "z_min":       Z_MIN_GLOBAL,
                    "z_max":       Z_MAX_GLOBAL,
                    "n_galaxies":  int(n_sub),
                    "n_matched":   int(len(fluxes)),
                    "tgids":       np.asarray(tgids_matched),
                }

                with open(pkl_path, "wb") as f:
                    pickle.dump(saved, f)
                print(f"      Saved {os.path.basename(pkl_path)}")

            results[file_key][i] = saved

    # -------------------------------------------------------------------------
    # 3. Write FastSpecFit-compatible FITS for each non-empty bin
    # -------------------------------------------------------------------------
    print("\n[4] Writing FastSpecFit (stackfit) input FITS files ...")

    wave_for_fits = wave  # already trimmed to < WAVE_MAX

    n_written = 0
    for sample_name, file_key in SAMPLE_SPECS:
        for i, saved in results[file_key].items():
            if saved is None:
                continue

            mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]

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

            # Sub-sample of bootstrap realizations to dump (rows 1..n_boot_keep)
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
                f"stack_mstar_{file_key}_{mstar_min:.2f}_{mstar_max:.2f}.fits",
            )

            write_stacked_spectra(
                outfile=out_fits,
                wave=wave_for_fits,
                flux=all_flux,
                ivar=all_ivar,
                stackids=np.arange(n_rows),
                stack_redshift=np.zeros(n_rows),
                table_column_dict={
                    "IS_MEAN":   np.array([1] + [0] * n_boot_keep, dtype=np.int64),
                    "NOBJ":      np.full(n_rows, n_galaxies, dtype=np.int64),
                    "MSTAR_MIN": np.full(n_rows, mstar_min, dtype=np.float32),
                    "MSTAR_MAX": np.full(n_rows, mstar_max, dtype=np.float32),
                },
                table_format_dict={
                    "IS_MEAN":   "K",
                    "NOBJ":      "K",
                    "MSTAR_MIN": "E",
                    "MSTAR_MAX": "E",
                },
            )
            print(f"    {sample_name} | log M*=[{mstar_min:.2f},{mstar_max:.2f}]: "
                  f"N_gal={n_galaxies}, 1 mean + {n_boot_keep} bootstraps "
                  f"-> {os.path.basename(out_fits)}")
            n_written += 1

    print(f"\n[5] Done. Wrote {n_written} FITS files to {STACK_PATH}")


if __name__ == "__main__":
    main()