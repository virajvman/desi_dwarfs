"""
stack_mstar_haew_3bin.py
========================

Bootstrap-stacked spectra in 0.5 dex bins of log M_star (6 -> 9) crossed
with THREE fixed H-alpha equivalent-width bins:

    1. EW <= 30 A          (weak / quiescent line emitters)
    2. 30 A < EW <= 300 A  (intermediate)
    3. EW > 300 A          (extreme emission-line galaxies)

BGS_BRIGHT, BGS_FAINT, and LOWZ are pooled together in each (M*, EW) cell.

This is a focused variant of
`code/stacking_analysis/stack_mstar_haew.py`: instead of continuous
log10(EW) bins it uses the three science bins above, and it only keeps
objects whose H-alpha EW is *detected* at SNR >= 3
(HALPHA_EW * sqrt(HALPHA_EW_IVAR) >= 3) on top of the flux-based cuts
already applied inside `stack_explore.load_catalog`.

All of the heavy lifting (catalog/spectra loading, TARGETID matching,
Halpha-flux normalization, bootstrap stacking, and the FastSpecFit FITS
writer) is reused from `code/stacking_analysis/stack_explore.py`.

Normalization: each spectrum is divided by its catalog HALPHA_FLUX before
stacking (norm_method="catalog", line_norm="HALPHA"), so the stack is a
Halpha-normalized mean and bright/close galaxies do not dominate.

Outputs (one per mstar-bin x EW-bin, combined across samples), written to
STACK_PATH:
  - stacks_spec_ALL_mstar_{mlo}_{mhi}_{ewtoken}.pkl
        (stack_spec, stack_err, all_stacks, bin metadata)
  - stack_ALL_mstar_{mlo}_{mhi}_{ewtoken}.fits
        (FastSpecFit stackfit input: row 0 = mean, rows 1.. = bootstraps)
  - plots/overlay_mstar_{mlo}_{mhi}.png   (3 EW stacks overlaid per mass bin)
  - plots/grid_all_stacks.png             (rows = mass bins, cols = EW bins)

Usage:
    python stack_mstar_haew_3bin.py
"""

import os
import sys

# This script lives in code/nebular_stuff/, but its helpers live in
# code/stacking_analysis/ (stack_explore) and code/ (transitive imports).
# Make both importable regardless of the caller's cwd / PYTHONPATH.
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
    write_stacked_spectra,
)


# =============================================================================
# CONFIG
# =============================================================================

# Samples pooled into the combined stack (logical OR on SAMPLE).
SAMPLES = ["BGS_BRIGHT", "BGS_FAINT", "LOWZ"]

# Tag used in output filenames for the combined-sample stack.
COMBINED_TAG = "ALL"

# Stellar-mass bin edges (log Msun), 0.5 dex between 6 and 9.
#   -> [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]   (6 bins)
MSTAR_BINS = np.arange(6.0, 9.0 + 1e-6, 0.5)

# Three fixed H-alpha EW bins (linear, in Angstroms). Half-open (lo, hi].
#   bin 0: EW <= 30
#   bin 1: 30 < EW <= 300
#   bin 2: EW > 300
EW_EDGES = [0.0, 30.0, 300.0, np.inf]

# Filename-/label-safe tokens for the three EW bins (one per gap in EW_EDGES).
EW_TOKENS = ["ew_lt30", "ew_30_300", "ew_gt300"]
EW_LABELS = [r"EW $\leq$ 30", r"30 < EW $\leq$ 300", "EW > 300"]

# Minimum H-alpha EW detection SNR (HALPHA_EW * sqrt(HALPHA_EW_IVAR)).
EW_SNR_MIN = 3.0

# Redshift range. Effectively no cut (same as the M*-only pipeline).
Z_MIN_GLOBAL = 0.0
Z_MAX_GLOBAL = 0.5

# Minimum number of galaxies in a bin to attempt a stack.
STACK_NLIM = 50

# Bootstrap settings (same defaults as the existing pipeline).
N_BOOTSTRAP = 200      # number of bootstrap realizations
N_DRAW      = 5000     # spectra per realization (capped at n_valid in-bin)
RANDOM_SEED = 42

# How many of the N_BOOTSTRAP realizations to save into the FITS file
# (these become rows 1.. and provide the error budget on stackfit-derived
# quantities).
N_BOOT_SAVE = 50

# Output location.
STACK_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_haew_3bin/"

# Re-do stacks even if a cached pickle already exists on disk?
OVERWRITE_STACKS = True

# Rest-frame wavelength upper bound (same as the M*-only pipeline). Still
# covers [OII] 3727, Hbeta, [OIII] 5007, Halpha.
WAVE_MAX = 6800

# Spectrum-normalization method used inside bootstrap_stack. "catalog"
# normalizes each spectrum by its catalog HALPHA_FLUX before stacking.
NORM_METHOD = "catalog"

# Reference emission lines (rest-frame vacuum, A) used only as visual guides
# in the comparison plots.
LINE_GUIDES = {
    "[OII]": 3727.0,
    r"H$\beta$": 4862.7,
    "[OIII]": 5008.2,
    r"H$\alpha$": 6564.6,
}


# =============================================================================
# HELPERS
# =============================================================================

def apply_ew_snr_cut(catalog, snr_min=EW_SNR_MIN):
    """Keep only galaxies with H-alpha EW detected at SNR >= snr_min.

    EW SNR = HALPHA_EW * sqrt(HALPHA_EW_IVAR). Also requires EW > 0 so the
    later linear-EW binning is well defined. This is applied on top of the
    flux-based cuts already performed in stack_explore.load_catalog.
    """
    ha_ew = np.asarray(catalog["HALPHA_EW"])
    ha_ew_ivar = np.asarray(catalog["HALPHA_EW_IVAR"])
    with np.errstate(invalid="ignore"):
        ew_snr = ha_ew * np.sqrt(ha_ew_ivar)
    mask = np.isfinite(ew_snr) & (ew_snr >= snr_min) & (ha_ew > 0)
    return catalog[mask]


def select_sample_ew_bin(
    catalog, sample_names,
    z_min, z_max,
    logmstar_min, logmstar_max,
    ew_min, ew_max,
):
    """Select galaxies in a (samples, z, log M*, linear Halpha-EW) cell.

    `sample_names` is an iterable of SAMPLE values pooled together
    (logical OR). The mass and EW cuts are half-open (lo-exclusive,
    hi-inclusive) so adjacent bins don't double-count shared edges; the
    top EW bin uses ew_max = +inf so it captures everything above 300 A.
    """
    sample_col = catalog["SAMPLE"]
    samp_mask = np.zeros(len(catalog), dtype=bool)
    for name in sample_names:
        samp_mask |= (sample_col == name)

    halpha_ew = np.asarray(catalog["HALPHA_EW"])

    mask = (
        samp_mask
        & (catalog["Z"] > z_min) & (catalog["Z"] < z_max)
        & (catalog["LOG_MSTAR_M24"] > logmstar_min)
        & (catalog["LOG_MSTAR_M24"] <= logmstar_max)
        & (halpha_ew > ew_min)
        & (halpha_ew <= ew_max)
    )
    return catalog[mask]


def bin_label(mstar_min, mstar_max, ew_token):
    """Filename-safe label for one (mass, EW) bin."""
    return f"mstar_{mstar_min:.2f}_{mstar_max:.2f}_{ew_token}"


# =============================================================================
# PLOTTING
# =============================================================================

def _add_line_guides(ax, ymin=None):
    """Draw faint vertical guides + labels at the reference emission lines."""
    for name, lam in LINE_GUIDES.items():
        ax.axvline(lam, color="grey", ls=":", lw=0.8, alpha=0.6, zorder=0)


def make_overlay_plots(results, wave, mstar_bins, plot_dir):
    """One panel per stellar-mass bin overlaying the 3 EW stacks.

    `results` is keyed by (i_mstar, j_ew) -> saved-dict or None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]  # one per EW bin

    for i in range(n_mstar):
        m_lo, m_hi = mstar_bins[i], mstar_bins[i + 1]

        # Skip mass bins with no successful stacks at all.
        if all(results.get((i, j)) is None for j in range(len(EW_TOKENS))):
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for j, (token, label) in enumerate(zip(EW_TOKENS, EW_LABELS)):
            saved = results.get((i, j))
            if saved is None:
                continue
            flux = saved["stack_spec"]
            err = saved["stack_err"]
            n_gal = saved["n_galaxies"]
            ax.plot(wave, flux, color=colors[j], lw=1.0,
                    label=f"{label}  (N={n_gal})")
            ax.fill_between(
                wave, flux - err, flux + err,
                color=colors[j], alpha=0.20, lw=0,
            )

        _add_line_guides(ax)
        ax.set_xlim(wave.min(), wave.max())
        ax.set_xlabel(r"Rest wavelength [$\AA$]")
        ax.set_ylabel("Halpha-normalized stacked flux")
        ax.set_title(
            f"log M* in [{m_lo:.2f}, {m_hi:.2f}]  -- H-alpha EW bins"
        )
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        fig.tight_layout()

        out_png = os.path.join(
            plot_dir, f"overlay_mstar_{m_lo:.2f}_{m_hi:.2f}.png"
        )
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"    Saved {os.path.basename(out_png)}")


def make_grid_plot(results, wave, mstar_bins, plot_dir):
    """Single grid figure: rows = mass bins, columns = EW bins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1
    n_ew = len(EW_TOKENS)
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(
        n_mstar, n_ew,
        figsize=(4.0 * n_ew, 2.4 * n_mstar),
        sharex=True, squeeze=False,
    )

    for i in range(n_mstar):
        m_lo, m_hi = mstar_bins[i], mstar_bins[i + 1]
        for j in range(n_ew):
            ax = axes[i][j]
            saved = results.get((i, j))
            if saved is not None:
                flux = saved["stack_spec"]
                err = saved["stack_err"]
                ax.plot(wave, flux, color=colors[j], lw=0.8)
                ax.fill_between(
                    wave, flux - err, flux + err,
                    color=colors[j], alpha=0.20, lw=0,
                )
                ax.text(
                    0.97, 0.92, f"N={saved['n_galaxies']}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=8,
                )
            else:
                ax.text(
                    0.5, 0.5, "no stack", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="grey",
                )
            _add_line_guides(ax)
            ax.set_xlim(wave.min(), wave.max())
            if i == 0:
                ax.set_title(EW_LABELS[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"[{m_lo:.1f},{m_hi:.1f}]", fontsize=9)
            if i == n_mstar - 1:
                ax.set_xlabel(r"Rest $\lambda$ [$\AA$]", fontsize=9)

    fig.suptitle("Halpha-normalized stacks: M* (rows) x H-alpha EW (cols)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_png = os.path.join(plot_dir, "grid_all_stacks.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(STACK_PATH, exist_ok=True)
    plot_dir = os.path.join(STACK_PATH, "plots")

    # -------------------------------------------------------------------------
    # 1. Load catalog + spectra
    # -------------------------------------------------------------------------
    print("[1] Loading catalog ...")
    tot_cat = load_catalog()
    print(f"    Total galaxies after quality cuts: {len(tot_cat)}")

    tot_cat = apply_ew_snr_cut(tot_cat, snr_min=EW_SNR_MIN)
    print(f"    After H-alpha EW SNR >= {EW_SNR_MIN:.0f} cut: {len(tot_cat)}")

    print("\n[2] Loading de-redshifted spectra ...")
    # spectra_data = load_spectra()
    spectra_data = load_spectra("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_combine_deredshift_hires_noinvvar.h5")
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
    n_ew    = len(EW_TOKENS)
    print(f"\n[3] Stacking grid: {n_mstar} mstar bins x {n_ew} EW bins"
          f" = {n_mstar * n_ew} candidate stacks"
          f"  (combining samples: {'|'.join(SAMPLES)})")
    print(f"    M* edges  : {MSTAR_BINS}")
    print(f"    EW edges  : {EW_EDGES}")
    print(f"    Output dir: {STACK_PATH}")

    # results[(i_mstar, j_ew)] -> dict (or None if skipped)
    results = {}

    print(f"\n========== Combined sample: {'|'.join(SAMPLES)} ==========")

    for i in range(n_mstar):
        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]

        for j in range(n_ew):
            ew_min, ew_max = EW_EDGES[j], EW_EDGES[j + 1]
            token = EW_TOKENS[j]
            label = bin_label(mstar_min, mstar_max, token)

            print(f"\n  --- {COMBINED_TAG} | log M*=[{mstar_min:.2f},{mstar_max:.2f}]"
                  f" | EW in ({ew_min:.0f},{ew_max:.0f}] ({token}) ---")

            sub_cat = select_sample_ew_bin(
                tot_cat, SAMPLES,
                Z_MIN_GLOBAL, Z_MAX_GLOBAL,
                mstar_min, mstar_max,
                ew_min, ew_max,
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
                    results[(i, j)] = None
                    continue

                # Bootstrap-stack:
                #   - normalize each spectrum by catalog Halpha flux
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
                    "ew_min":      float(ew_min),
                    "ew_max":      float(ew_max),
                    "ew_token":    token,
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
        ew_min, ew_max = EW_EDGES[j], EW_EDGES[j + 1]
        token = EW_TOKENS[j]
        label = bin_label(mstar_min, mstar_max, token)

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

        # +inf top edge is not FITS-friendly; store a sentinel for EW_MAX.
        ew_max_fits = ew_max if np.isfinite(ew_max) else -1.0

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
                "IS_MEAN":   np.array([1] + [0] * n_boot_keep, dtype=np.int64),
                "NOBJ":      np.full(n_rows, n_galaxies, dtype=np.int64),
                "MSTAR_MIN": np.full(n_rows, mstar_min, dtype=np.float32),
                "MSTAR_MAX": np.full(n_rows, mstar_max, dtype=np.float32),
                "EW_MIN":    np.full(n_rows, ew_min,      dtype=np.float32),
                "EW_MAX":    np.full(n_rows, ew_max_fits, dtype=np.float32),
            },
            table_format_dict={
                "IS_MEAN":   "K",
                "NOBJ":      "K",
                "MSTAR_MIN": "E",
                "MSTAR_MAX": "E",
                "EW_MIN":    "E",
                "EW_MAX":    "E",
            },
        )
        print(f"    {COMBINED_TAG} | {label}: "
              f"N_gal={n_galaxies}, 1 mean + {n_boot_keep} bootstraps "
              f"-> {os.path.basename(out_fits)}")
        n_written += 1

    print(f"\n[5] Wrote {n_written} FITS files to {STACK_PATH}")

    # -------------------------------------------------------------------------
    # 4. Comparison plots
    # -------------------------------------------------------------------------
    print("\n[6] Making comparison plots ...")
    make_overlay_plots(results, wave, MSTAR_BINS, plot_dir)
    make_grid_plot(results, wave, MSTAR_BINS, plot_dir)

    print(f"\n[7] Done. Plots in {plot_dir}")


if __name__ == "__main__":
    main()
