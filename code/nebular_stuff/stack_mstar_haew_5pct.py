"""
stack_mstar_haew_5pct.py
========================

Bootstrap-stacked spectra in 0.5 dex bins of log M_star (6 -> 9) crossed
with FIVE H-alpha equivalent-width bins defined by percentiles of HALPHA_EW
in the pooled BGS_BRIGHT | BGS_FAINT | LOWZ catalog (after EW SNR >= 3):

    0-20%, 20-40%, 40-60%, 60-80%, 80-100% of HALPHA_EW.

BGS_BRIGHT, BGS_FAINT, and LOWZ are pooled together in each (M*, EW) cell.
Every non-empty bin is stacked (no minimum count); N=1 uses the single
spectrum as-is.

All of the heavy lifting (catalog/spectra loading, TARGETID matching,
Halpha-flux normalization, bootstrap stacking, and the FastSpecFit FITS
writer) is reused from `code/stacking_analysis/stack_explore.py`.

Normalization: each spectrum is divided by its catalog HALPHA_BOXFLUX
(boxcar Halpha flux) before stacking (norm_method="catalog",
norm_col="HALPHA_BOXFLUX"), so the stack is a Halpha-normalized mean and
bright/close galaxies do not dominate.

Outputs (one per mstar-bin x EW-bin, combined across samples), written to
STACK_PATH:
  - stacks_spec_ALL_mstar_{mlo}_{mhi}_{ewtoken}.pkl
  - stack_ALL_mstar_{mlo}_{mhi}_{ewtoken}.fits
  - plots/overlay_mstar_{mlo}_{mhi}.png   (5 EW stacks overlaid per mass bin)
  - plots/grid_all_stacks.png             (rows = mass bins, cols = EW bins)

Usage:
    python stack_mstar_haew_5pct.py
"""

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
    write_stacked_spectra,
)


# =============================================================================
# CONFIG
# =============================================================================

SAMPLES = ["BGS_BRIGHT", "BGS_FAINT", "LOWZ"]
COMBINED_TAG = "ALL"

MSTAR_BINS = np.arange(6.0, 9.0 + 1e-6, 0.5)

# Five percentile bins; edges computed at runtime from the catalog.
EW_PERCENTILES = [20, 40, 60, 80]
EW_TOKENS = ["ew_p00_20", "ew_p20_40", "ew_p40_60", "ew_p60_80", "ew_p80_100"]
EW_PCT_NAMES = ["p00-20", "p20-40", "p40-60", "p60-80", "p80-100"]

EW_SNR_MIN = 3.0
Z_MIN_GLOBAL = 0.0
Z_MAX_GLOBAL = 0.5

N_BOOTSTRAP = 200
N_DRAW      = 5000
RANDOM_SEED = 42
N_BOOT_SAVE = 50

STACK_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_haew_5pct/"

OVERWRITE_STACKS = True
WAVE_MAX = 6800
NORM_METHOD = "catalog"
NORM_COL = "HALPHA_BOXFLUX"

LINE_GUIDES = {
    "[OII]": 3727.0,
    r"H$\beta$": 4862.7,
    "[OIII]": 5008.2,
    r"H$\alpha$": 6564.6,
}

EW_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# =============================================================================
# HELPERS
# =============================================================================

def apply_ew_snr_cut(catalog, snr_min=EW_SNR_MIN):
    """Keep only galaxies with H-alpha EW detected at SNR >= snr_min."""
    ha_ew = np.asarray(catalog["HALPHA_EW"])
    ha_ew_ivar = np.asarray(catalog["HALPHA_EW_IVAR"])
    with np.errstate(invalid="ignore"):
        ew_snr = ha_ew * np.sqrt(ha_ew_ivar)
    mask = np.isfinite(ew_snr) & (ew_snr >= snr_min) & (ha_ew > 0)
    return catalog[mask]


def select_samples(catalog, sample_names):
    """Mask catalog to the pooled SAMPLE values."""
    sample_col = catalog["SAMPLE"]
    samp_mask = np.zeros(len(catalog), dtype=bool)
    for name in sample_names:
        samp_mask |= (sample_col == name)
    return catalog[samp_mask]


def compute_ew_percentile_bins(catalog, sample_names, percentiles=EW_PERCENTILES):
    """Return (EW_EDGES, EW_LABELS) from HALPHA_EW percentiles on pooled samples."""
    pct_cat = select_samples(catalog, sample_names)
    ha_ew = np.asarray(pct_cat["HALPHA_EW"])
    ha_ew = ha_ew[np.isfinite(ha_ew) & (ha_ew > 0)]
    if len(ha_ew) == 0:
        raise ValueError("No finite HALPHA_EW values for percentile binning.")

    pct_vals = np.percentile(ha_ew, percentiles)
    ew_edges = [0.0, *pct_vals.tolist(), np.inf]
    ew_labels = []
    for j in range(len(EW_PCT_NAMES)):
        lo, hi = ew_edges[j], ew_edges[j + 1]
        if np.isfinite(hi):
            ew_labels.append(
                f"{EW_PCT_NAMES[j]}: ({lo:.1f}, {hi:.1f}] $\\AA$"
            )
        else:
            ew_labels.append(
                f"{EW_PCT_NAMES[j]}: ({lo:.1f}, $\\infty$] $\\AA$"
            )
    return ew_edges, ew_labels, len(pct_cat)


def print_ew_percentile_table(ew_edges, ew_labels, n_pct):
    """Print HALPHA_EW percentile ranges for reference."""
    print(f"\nH-alpha EW percentile bins (BGS+LOWZ, EW SNR>={EW_SNR_MIN:.0f}, N={n_pct}):")
    for name, label in zip(EW_PCT_NAMES, ew_labels):
        print(f"  {name}: {label}")


def select_sample_ew_bin(
    catalog, sample_names,
    z_min, z_max,
    logmstar_min, logmstar_max,
    ew_min, ew_max,
):
    """Select galaxies in a (samples, z, log M*, linear Halpha-EW) cell."""
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

def _add_line_guides(ax):
    """Draw faint vertical guides at reference emission lines."""
    for lam in LINE_GUIDES.values():
        ax.axvline(lam, color="grey", ls=":", lw=0.8, alpha=0.6, zorder=0)


def make_overlay_plots(results, wave, mstar_bins, ew_labels, plot_dir):
    """One panel per stellar-mass bin overlaying the 5 EW stacks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1
    n_ew = len(EW_TOKENS)

    for i in range(n_mstar):
        m_lo, m_hi = mstar_bins[i], mstar_bins[i + 1]

        if all(results.get((i, j)) is None for j in range(n_ew)):
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        for j, label in enumerate(ew_labels):
            saved = results.get((i, j))
            if saved is None:
                continue
            flux = saved["stack_spec"]
            err = saved["stack_err"]
            n_gal = saved["n_galaxies"]
            ax.plot(wave, flux, color=EW_COLORS[j], lw=1.0,
                    label=f"{label}  (N={n_gal})")
            ax.fill_between(
                wave, flux - err, flux + err,
                color=EW_COLORS[j], alpha=0.20, lw=0,
            )

        _add_line_guides(ax)
        ax.set_xlim(wave.min(), wave.max())
        ax.set_xlabel(r"Rest wavelength [$\AA$]")
        ax.set_ylabel("Halpha-normalized stacked flux")
        ax.set_title(
            f"log M* in [{m_lo:.2f}, {m_hi:.2f}]  -- H-alpha EW percentile bins"
        )
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        fig.tight_layout()

        out_png = os.path.join(
            plot_dir, f"overlay_mstar_{m_lo:.2f}_{m_hi:.2f}.png"
        )
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"    Saved {os.path.basename(out_png)}")


def make_grid_plot(results, wave, mstar_bins, ew_labels, plot_dir):
    """Single grid figure: rows = mass bins, columns = EW bins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1
    n_ew = len(EW_TOKENS)

    fig, axes = plt.subplots(
        n_mstar, n_ew,
        figsize=(3.5 * n_ew, 2.4 * n_mstar),
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
                ax.plot(wave, flux, color=EW_COLORS[j], lw=0.8)
                ax.fill_between(
                    wave, flux - err, flux + err,
                    color=EW_COLORS[j], alpha=0.20, lw=0,
                )
                ax.text(
                    0.97, 0.92, f"N={saved['n_galaxies']}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=7,
                )
            else:
                ax.text(
                    0.5, 0.5, "no stack", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="grey",
                )
            _add_line_guides(ax)
            ax.set_xlim(wave.min(), wave.max())
            if i == 0:
                ax.set_title(ew_labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"[{m_lo:.1f},{m_hi:.1f}]", fontsize=9)
            if i == n_mstar - 1:
                ax.set_xlabel(r"Rest $\lambda$ [$\AA$]", fontsize=8)

    fig.suptitle("Halpha-normalized stacks: M* (rows) x H-alpha EW pct (cols)",
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

    print("[1] Loading catalog ...")
    tot_cat = load_catalog()
    print(f"    Total galaxies after quality cuts: {len(tot_cat)}")

    tot_cat = apply_ew_snr_cut(tot_cat, snr_min=EW_SNR_MIN)
    print(f"    After H-alpha EW SNR >= {EW_SNR_MIN:.0f} cut: {len(tot_cat)}")

    EW_EDGES, EW_LABELS, n_pct = compute_ew_percentile_bins(tot_cat, SAMPLES)
    print_ew_percentile_table(EW_EDGES, EW_LABELS, n_pct)

    print("\n[2] Loading de-redshifted spectra ...")
    spectra_data = load_spectra(
        "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/"
        "desi_y1_dwarf_combine_deredshift_hires_noinvvar.h5"
    )
    print(f"    Total spectra loaded: {len(spectra_data['targetid'])}")

    wave_mask = spectra_data["wave_rest"] < WAVE_MAX
    spectra_data["wave_rest"] = spectra_data["wave_rest"][wave_mask]
    spectra_data["flux"]      = spectra_data["flux"][:, wave_mask]
    spectra_data["flux_ivar"] = spectra_data["flux_ivar"][:, wave_mask]
    wave = spectra_data["wave_rest"]
    print(f"    After trim: lambda in [{wave.min():.1f}, {wave.max():.1f}] A"
          f"  ({len(wave)} pixels)")

    n_mstar = len(MSTAR_BINS) - 1
    n_ew    = len(EW_TOKENS)
    print(f"\n[3] Stacking grid: {n_mstar} mstar bins x {n_ew} EW bins"
          f" = {n_mstar * n_ew} candidate stacks"
          f"  (combining samples: {'|'.join(SAMPLES)})")
    print(f"    M* edges  : {MSTAR_BINS}")
    print(f"    EW edges  : {EW_EDGES}")
    print(f"    Output dir: {STACK_PATH}")

    results = {}

    print(f"\n========== Combined sample: {'|'.join(SAMPLES)} ==========")

    for i in range(n_mstar):
        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]

        for j in range(n_ew):
            ew_min, ew_max = EW_EDGES[j], EW_EDGES[j + 1]
            token = EW_TOKENS[j]
            label = bin_label(mstar_min, mstar_max, token)

            ew_hi_str = f"{ew_max:.1f}" if np.isfinite(ew_max) else "inf"
            print(f"\n  --- {COMBINED_TAG} | log M*=[{mstar_min:.2f},{mstar_max:.2f}]"
                  f" | EW in ({ew_min:.1f},{ew_hi_str}] ({token}) ---")

            sub_cat = select_sample_ew_bin(
                tot_cat, SAMPLES,
                Z_MIN_GLOBAL, Z_MAX_GLOBAL,
                mstar_min, mstar_max,
                ew_min, ew_max,
            )
            n_sub = len(sub_cat)
            print(f"      N galaxies in bin: {n_sub}")

            if n_sub == 0:
                print("      Skipping (empty bin)")
                results[(i, j)] = None
                continue

            pkl_path = os.path.join(
                STACK_PATH,
                f"stacks_spec_{COMBINED_TAG}_{label}.pkl",
            )

            if os.path.exists(pkl_path) and not OVERWRITE_STACKS:
                print(f"      Loading cached: {os.path.basename(pkl_path)}")
                with open(pkl_path, "rb") as f:
                    saved = pickle.load(f)
            else:
                out = get_sample_spectra_with_linenorm(
                    sub_cat, spectra_data, line_norm="HALPHA", norm_col=NORM_COL,
                )
                fluxes, ivars, halpha_fluxes, tgids_matched = out

                if fluxes is None or len(fluxes) == 0:
                    print("      No matched spectra; skipping.")
                    results[(i, j)] = None
                    continue

                n_matched = len(fluxes)
                print(f"      Bootstrap-stacking: N={n_sub}, matched={n_matched}, "
                      f"n_bootstrap={N_BOOTSTRAP}, "
                      f"n_draw={min(N_DRAW, n_matched)}")
                stack_spec, stack_err, all_stacks = bootstrap_stack(
                    fluxes=fluxes,
                    ivars=ivars,
                    wave=wave,
                    n_bootstrap=N_BOOTSTRAP,
                    n_draw=N_DRAW,
                    random_seed=RANDOM_SEED,
                    norm_method=NORM_METHOD,
                    catalog_line_fluxes=halpha_fluxes,
                    min_n_valid=1,
                )

                if all_stacks is None:
                    print("      bootstrap_stack returned None; skipping.")
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
                    "ew_max":      float(ew_max) if np.isfinite(ew_max) else -1.0,
                    "ew_token":    token,
                    "z_min":       Z_MIN_GLOBAL,
                    "z_max":       Z_MAX_GLOBAL,
                    "n_galaxies":  int(n_sub),
                    "n_matched":   int(n_matched),
                    "tgids":       np.asarray(tgids_matched),
                }

                with open(pkl_path, "wb") as f:
                    pickle.dump(saved, f)
                print(f"      Saved {os.path.basename(pkl_path)}")

            results[(i, j)] = saved

    print("\n[4] Writing FastSpecFit (stackfit) input FITS files ...")

    wave_for_fits = wave
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

        ivar_mean = np.where(
            np.isfinite(err_mean) & (err_mean > 0),
            1.0 / err_mean ** 2,
            0.0,
        )

        rng = np.random.default_rng(RANDOM_SEED)
        n_boot_avail = len(all_stacks)
        n_boot_keep  = min(N_BOOT_SAVE, n_boot_avail)
        boot_idx     = rng.choice(n_boot_avail, size=n_boot_keep, replace=False)
        boot_stacks  = all_stacks[boot_idx]

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
            all_ivar[k] = all_ivar[0]

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

    print("\n[6] Making comparison plots ...")
    make_overlay_plots(results, wave, MSTAR_BINS, EW_LABELS, plot_dir)
    make_grid_plot(results, wave, MSTAR_BINS, EW_LABELS, plot_dir)

    print(f"\n[7] Done. Plots in {plot_dir}")


if __name__ == "__main__":
    main()
