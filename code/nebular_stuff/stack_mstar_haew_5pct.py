"""
stack_mstar_haew_5pct.py
========================

Bootstrap-stacked spectra in 0.5 dex bins of log M_star (6 -> 9) with
conditional H-alpha EW percentile splitting (BGS_BRIGHT | BGS_FAINT | LOWZ,
EW SNR >= 3):

  - EW percentile edges (0-20, 20-40, ..., 80-100%) are computed globally.
  - Per mass bin: stack EW quintiles only when N >= EW_STACK_NLIM (50).
  - If all five EW cells have N < 50, one pooled mass-only stack (ew_all).
  - Per mass bin: additionally stack EW > 300 A (ew_gt300) when N >= 50;
    overlaps ew_p80_100 but is written as a separate test bin.

Each output stack is a single FITS spectrum (mean over internal bootstrap
coadds); bootstrap std sets the pixel ivar. Internal bootstrap is not written
as extra FITS rows.

Outputs (written to STACK_PATH; stale stack_ALL_*.fits / stacks_spec_*.pkl
removed at the start of each run):
  - stacks_spec_ALL_mstar_{mlo}_{mhi}_{ewtoken}.pkl
  - stack_ALL_mstar_{mlo}_{mhi}_{ewtoken}.fits   (1 row)
  - plots/overlay_mstar_{mlo}_{mhi}.png
  - plots/grid_all_stacks.png

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
    write_stacked_spectra,
)


# =============================================================================
# CONFIG
# =============================================================================

SAMPLES = ["BGS_BRIGHT", "BGS_FAINT", "LOWZ"]
COMBINED_TAG = "ALL"

MSTAR_BINS = np.arange(6.0, 9.0 + 1e-6, 0.5)

EW_PERCENTILES = [20, 40, 60, 80]
EW_TOKENS = ["ew_p00_20", "ew_p20_40", "ew_p40_60", "ew_p60_80", "ew_p80_100"]
EW_PCT_NAMES = ["p00-20", "p20-40", "p40-60", "p60-80", "p80-100"]
POOLED_EW_TOKEN = "ew_all"
EW_GT300_MIN = 300.0
EW_GT300_TOKEN = "ew_gt300"

EW_SNR_MIN = 3.0
EW_STACK_NLIM = 50
Z_MIN_GLOBAL = 0.0
Z_MAX_GLOBAL = 0.5

N_BOOTSTRAP = 200
N_DRAW      = 5000
RANDOM_SEED = 42

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

EW_COLORS = {
    "ew_p00_20": "#1f77b4",
    "ew_p20_40": "#ff7f0e",
    "ew_p40_60": "#2ca02c",
    "ew_p60_80": "#d62728",
    "ew_p80_100": "#9467bd",
    EW_GT300_TOKEN: "#e377c2",
    POOLED_EW_TOKEN: "#7f7f7f",
}


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
    print(f"  EW split minimum per cell: N >= {EW_STACK_NLIM}")


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


def plan_stacks_for_mass_bin(tot_cat, mstar_min, mstar_max, ew_edges):
    """Return (plan, ew_counts, mode) for one stellar-mass bin.

    plan: list of (ew_token, sub_cat, ew_min, ew_max_fits) to stack.
    ew_counts: galaxy counts in each of the 5 EW percentile cells.
    mode: 'full', 'partial', 'pooled', or 'empty'.
    """
    ew_counts = []
    for j in range(len(EW_TOKENS)):
        sub = select_sample_ew_bin(
            tot_cat, SAMPLES,
            Z_MIN_GLOBAL, Z_MAX_GLOBAL,
            mstar_min, mstar_max,
            ew_edges[j], ew_edges[j + 1],
        )
        ew_counts.append(len(sub))

    if all(n < EW_STACK_NLIM for n in ew_counts):
        sub_all = select_sample_mstar_bin(
            tot_cat, SAMPLES,
            Z_MIN_GLOBAL, Z_MAX_GLOBAL,
            mstar_min, mstar_max,
        )
        if len(sub_all) == 0:
            return [], ew_counts, "empty"
        return [(POOLED_EW_TOKEN, sub_all, 0.0, -1.0)], ew_counts, "pooled"

    plan = []
    for j, token in enumerate(EW_TOKENS):
        if ew_counts[j] >= EW_STACK_NLIM:
            ew_max = ew_edges[j + 1]
            ew_max_fits = float(ew_max) if np.isfinite(ew_max) else -1.0
            sub = select_sample_ew_bin(
                tot_cat, SAMPLES,
                Z_MIN_GLOBAL, Z_MAX_GLOBAL,
                mstar_min, mstar_max,
                ew_edges[j], ew_edges[j + 1],
            )
            plan.append((token, sub, float(ew_edges[j]), ew_max_fits))

    if len(plan) == 0:
        return [], ew_counts, "empty"
    mode = "full" if len(plan) == len(EW_TOKENS) else "partial"
    return plan, ew_counts, mode


def bin_label(mstar_min, mstar_max, ew_token):
    """Filename-safe label for one (mass, EW) bin."""
    return f"mstar_{mstar_min:.2f}_{mstar_max:.2f}_{ew_token}"


def plot_label_for_token(token, ew_edges, ew_labels):
    """Human-readable legend label for a stack token."""
    if token == POOLED_EW_TOKEN:
        return "EW pooled (all)"
    if token == EW_GT300_TOKEN:
        return r"EW $>$ 300 $\AA$"
    if token in EW_TOKENS:
        return ew_labels[EW_TOKENS.index(token)]
    return token


def clean_stack_outputs(stack_path):
    """Remove previous stack FITS and pickle files before a fresh run."""
    n_removed = 0
    for pattern in ("stack_ALL_mstar_*.fits", "stacks_spec_ALL_mstar_*.pkl"):
        for fpath in glob.glob(os.path.join(stack_path, pattern)):
            os.remove(fpath)
            n_removed += 1
    print(f"    Removed {n_removed} previous stack .fits/.pkl files from {stack_path}")


def stack_one_bin(sub_cat, spectra_data, wave, token, ew_min, ew_max_fits,
                  mstar_min, mstar_max, label):
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
    print(f"      Bootstrap-stacking: N={n_sub}, matched={n_matched}, "
          f"n_bootstrap={N_BOOTSTRAP}, n_draw={min(N_DRAW, n_matched)}")
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
        return None

    saved = {
        "stack_spec":  stack_spec,
        "stack_err":   stack_err,
        "all_stacks":  all_stacks,
        "samples":     list(SAMPLES),
        "mstar_min":   float(mstar_min),
        "mstar_max":   float(mstar_max),
        "ew_min":      float(ew_min),
        "ew_max":      float(ew_max_fits),
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
    return saved


def write_single_row_fits(saved, wave_for_fits, label):
    """Write one-row FastSpecFit input FITS for a stacked bin."""
    flux_mean = saved["stack_spec"]
    err_mean = saved["stack_err"]
    n_galaxies = saved["n_galaxies"]
    mstar_min = saved["mstar_min"]
    mstar_max = saved["mstar_max"]
    ew_min = saved["ew_min"]
    ew_max_fits = saved["ew_max"]

    ivar_mean = np.where(
        np.isfinite(err_mean) & (err_mean > 0),
        1.0 / err_mean ** 2,
        0.0,
    )

    flux_row = np.where(np.isfinite(flux_mean), flux_mean, 0.0).astype(np.float32)
    ivar_row = np.where(
        np.isfinite(ivar_mean) & (flux_row != 0),
        ivar_mean, 0.0,
    ).astype(np.float32)

    out_fits = os.path.join(STACK_PATH, f"stack_{COMBINED_TAG}_{label}.fits")

    write_stacked_spectra(
        outfile=out_fits,
        wave=wave_for_fits,
        flux=flux_row[np.newaxis, :],
        ivar=ivar_row[np.newaxis, :],
        stackids=np.array([0], dtype=np.int64),
        stack_redshift=np.array([0.0]),
        table_column_dict={
            "IS_MEAN":   np.array([1], dtype=np.int64),
            "NOBJ":      np.array([n_galaxies], dtype=np.int64),
            "MSTAR_MIN": np.array([mstar_min], dtype=np.float32),
            "MSTAR_MAX": np.array([mstar_max], dtype=np.float32),
            "EW_MIN":    np.array([ew_min], dtype=np.float32),
            "EW_MAX":    np.array([ew_max_fits], dtype=np.float32),
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
          f"N_gal={n_galaxies}, 1 spectrum -> {os.path.basename(out_fits)}")


# =============================================================================
# PLOTTING
# =============================================================================

def _add_line_guides(ax):
    for lam in LINE_GUIDES.values():
        ax.axvline(lam, color="grey", ls=":", lw=0.8, alpha=0.6, zorder=0)


def _stacks_for_mass_bin(results, i_mstar):
    """Return list of (token, saved) for non-None stacks in one mass bin."""
    out = []
    for (i, token), saved in sorted(results.items()):
        if i == i_mstar and saved is not None:
            out.append((token, saved))
    return out


def make_overlay_plots(results, wave, mstar_bins, ew_labels, plot_dir):
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
            label = plot_label_for_token(token, None, ew_labels)
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


def make_grid_plot(results, wave, mstar_bins, ew_labels, plot_dir):
    """Grid: rows = mass bins, columns = stacks that exist (variable width)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    n_mstar = len(mstar_bins) - 1
    max_cols = max(len(_stacks_for_mass_bin(results, i)) for i in range(n_mstar))
    if max_cols == 0:
        print("    (no stacks; skipping grid_all_stacks)")
        return

    fig, axes = plt.subplots(
        n_mstar, max_cols,
        figsize=(3.5 * max_cols, 2.4 * n_mstar),
        sharex=True, squeeze=False,
    )

    for i in range(n_mstar):
        m_lo, m_hi = mstar_bins[i], mstar_bins[i + 1]
        stacks = _stacks_for_mass_bin(results, i)
        for j in range(max_cols):
            ax = axes[i][j]
            if j < len(stacks):
                token, saved = stacks[j]
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
                if i == 0:
                    ax.set_title(
                        plot_label_for_token(token, None, ew_labels), fontsize=7,
                    )
            else:
                ax.axis("off")
            if j == 0 and len(stacks) > 0:
                ax.set_ylabel(f"[{m_lo:.1f},{m_hi:.1f}]", fontsize=9)
            if i == n_mstar - 1 and j < len(stacks):
                ax.set_xlabel(r"Rest $\lambda$ [$\AA$]", fontsize=8)
            if j < len(stacks):
                _add_line_guides(ax)
                ax.set_xlim(wave.min(), wave.max())

    fig.suptitle("Halpha-normalized stacks (variable EW split per mass bin)",
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

    print("[0] Cleaning previous stack outputs ...")
    clean_stack_outputs(STACK_PATH)

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
    print(f"\n[3] Stacking: {n_mstar} mstar bins, conditional EW split "
          f"(N>={EW_STACK_NLIM} per EW cell else pooled ew_all)")
    print(f"    M* edges  : {MSTAR_BINS}")
    print(f"    EW edges  : {EW_EDGES}")
    print(f"    Output dir: {STACK_PATH}")

    results = {}

    print(f"\n========== Combined sample: {'|'.join(SAMPLES)} ==========")

    for i in range(n_mstar):
        mstar_min, mstar_max = MSTAR_BINS[i], MSTAR_BINS[i + 1]
        plan, ew_counts, mode = plan_stacks_for_mass_bin(
            tot_cat, mstar_min, mstar_max, EW_EDGES,
        )

        sub_gt300 = select_sample_ew_bin(
            tot_cat, SAMPLES,
            Z_MIN_GLOBAL, Z_MAX_GLOBAL,
            mstar_min, mstar_max,
            EW_GT300_MIN, np.inf,
        )
        n_gt300 = len(sub_gt300)
        if n_gt300 >= EW_STACK_NLIM:
            plan.append((EW_GT300_TOKEN, sub_gt300, EW_GT300_MIN, -1.0))

        counts_str = ", ".join(
            f"{EW_PCT_NAMES[j]}={ew_counts[j]}" for j in range(len(EW_TOKENS))
        )
        print(f"\n  --- log M*=[{mstar_min:.2f},{mstar_max:.2f}] ---")
        print(f"      EW counts: {counts_str}, ew_gt300={n_gt300}")
        print(f"      Mode: {mode} ({len(plan)} stack(s) planned)")

        if len(plan) == 0:
            continue

        for token, sub_cat, ew_min, ew_max_fits in plan:
            label = bin_label(mstar_min, mstar_max, token)
            ew_hi_str = (
                f"{ew_max_fits:.1f}" if ew_max_fits >= 0 else "inf/pooled"
            )
            print(f"\n    >> {token} | EW in ({ew_min:.1f}, {ew_hi_str}] | "
                  f"N={len(sub_cat)}")

            saved = stack_one_bin(
                sub_cat, spectra_data, wave, token, ew_min, ew_max_fits,
                mstar_min, mstar_max, label,
            )
            results[(i, token)] = saved

    print("\n[4] Writing FastSpecFit (stackfit) input FITS files ...")

    n_written = 0
    for (i, token), saved in results.items():
        if saved is None:
            continue
        label = bin_label(saved["mstar_min"], saved["mstar_max"], token)
        write_single_row_fits(saved, wave, label)
        n_written += 1

    print(f"\n[5] Wrote {n_written} FITS files to {STACK_PATH}")

    print("\n[6] Making comparison plots ...")
    make_overlay_plots(results, wave, MSTAR_BINS, EW_LABELS, plot_dir)
    make_grid_plot(results, wave, MSTAR_BINS, EW_LABELS, plot_dir)

    print(f"\n[7] Done. Plots in {plot_dir}")


if __name__ == "__main__":
    main()
