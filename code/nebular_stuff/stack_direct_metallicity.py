"""
stack_direct_metallicity.py
===========================

Direct-method (T_e) nebular abundances for the M* x H-alpha-EW stacked
spectra produced by `stack_mstar_haew_5pct.py` and fit with custom
FastSpecFit (`job_scripts/fastspec/run_stack_fastspec_haew_5pct.sh`).

For each stack bin there is one FastSpecFit stack output
`fastspec_stack_ALL_mstar_{mlo}_{mhi}_{ewtoken}.fits` whose emission-line
table (hdu=3) has 1 central row (IS_MEAN stack) plus 200 bootstrap rows.
EW tokens are fixed bins:
`ew_lt30`, `ew_30_100`, `ew_gt100` (<30, 30-100, >100 Angstrom). Mass bins:
0.5 dex from log M*=6 to 8, then 0.25 dex to 9.25 (9 mass bins). Stacks are
produced only when N >= 50 in the (mass, EW) cell.

This script:
  1. Globs the per-bin FastSpecFit stack outputs.
  2. Keeps only bins where the MEAN stack (row 0 ONLY, the total stack)
     passes the TE line-SNR gate at SNR > 10 with flux > 0.005 (FastSpec
     units) in ALL relevant lines, via `sfr_and_metallicity.line_snr_mask`.
     Base lines: HALPHA, HBETA, HGAMMA, OIII_4363, OIII_5007, OII_3726,
     OII_3729 (7); SII density mode also requires SII_6716, SII_6731 (9).
     With --fix-ne100 the density doublet is not needed, so the gate is the
     7 base lines regardless of --density-diagnostic.
  3. Of the 200 bootstrap rows, keeps those that INDEPENDENTLY pass the
     SAME gate, and runs the UltraNest direct-method fit
     (`pn_functions.compute_direct_metallicities`) on row 0 + those passing
     bootstrap rows only (default density diagnostic: SII). With --fix-ne100
     the electron density is held fixed at 100 cm^-3 instead of being fit
     from the density doublet.
  4. Reports per bin:
       - central value  = posterior median from the mean-stack fit (row 0), and
       - uncertainty    = 16/50/84 percentiles of bootstrap-row posterior medians
         when N_BOOT_FIT >= MIN_N_BOOT_FIT; otherwise posterior interval only
         with BOOT_ERR_RELIABLE=False. Mean-stack posterior interval stored
         separately as {PARAM}_MEANFIT_LO/HI (not combined with bootstrap error).
  5. Writes ONE results row per candidate bin (kept and dropped alike, with
     a DETECTED_7LINE flag for bins that passed the TE line gate) to a
     FITS + ECSV table so the M*/EW trends can be plotted later.

Outputs (written to STACK_PATH):
  - direct_metallicity_results.fits
  - direct_metallicity_results.ecsv
  - plots/oh_av_vs_mstar.png
  - plots/te_ne_hahb_vs_mstar.png
  - plots/obs_hahb_vs_mstar.png
  - plots/doublet_ratios_vs_mstar.png
  - plots/ne_diagnostic_ratio_vs_density.png

Usage:
    python stack_direct_metallicity.py --line-flux-type BOXFLUX
    python stack_direct_metallicity.py --line-flux-type FLUX --density-diagnostic OII
"""

import argparse
import os
import sys
import glob
import re

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _CODE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from astropy.table import Table

from sfr_and_metallicity import line_snr_mask


# =============================================================================
# CONFIG
# =============================================================================

STACK_PATH = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_haew_5pct/"

INPUT_GLOB = "fastspec_stack_ALL_mstar_*.fits"
FASTSPEC_HDU = 3

EW_BIN_TOKENS = ["ew_lt30", "ew_30_100", "ew_gt100"]
EW_TOKENS = EW_BIN_TOKENS
EW_BIN_LABELS = {
    "ew_lt30": r"EW $\leq$ 30 $\AA$",
    "ew_30_100": r"30 $<$ EW $\leq$ 100 $\AA$",
    "ew_gt100": r"EW $>$ 100 $\AA$",
}
EW_COLORS = {
    "ew_lt30": "#1f77b4",
    "ew_30_100": "#ff7f0e",
    "ew_gt100": "#d62728",
}

_TE_LINE_NAMES_BASE = ["HALPHA", "HBETA", "HGAMMA",
                       "OIII_4363", "OIII_5007",
                       "OII_3726", "OII_3729"]
SNR_VAL = 10
MIN_FLUX = 0.005

# Bootstrap reliability gate for direct-method errors (survivor bias on OIII_4363).
MIN_N_BOOT_FIT = 30
N_BOOT_TOTAL = 200

TE_DENSITY_DIAGNOSTIC = "SII"

# Electron density held fixed when --fix-ne100 is passed (cm^-3).
FIXED_NE_VALUE = 100.0

N_JOBS = 128
MIN_NUM_LIVE_POINTS = 400
SAMPLER_KWARGS = {"frac_remain": 0.01, "max_iters": 40000, "max_ncalls": int(1e5)}
USE_INFORMATIVE_PRIORS = False

PARAM_MAP = [
    ("ne_oii",        "NE_OII"),
    ("te_oiii",       "TE_OIII"),
    ("Av",            "AV"),
    ("log_O2_abund",  "LOG_O2_ABUND"),
    ("log_O3_abund",  "LOG_O3_ABUND"),
    ("twelve_log_OH", "TWELVE_LOG_OH"),
]

OUT_BASENAME = "direct_metallicity_results"


# =============================================================================
# HELPERS
# =============================================================================

_FNAME_RE = re.compile(
    r"fastspec_stack_ALL_mstar_"
    r"(?P<mlo>[-\d.]+)_(?P<mhi>[-\d.]+)_"
    r"(?P<ewtoken>ew_\w+)\.fits$"
)


def _input_stack_path(mlo, mhi, token):
    return os.path.join(
        STACK_PATH, f"stack_ALL_mstar_{mlo:.2f}_{mhi:.2f}_{token}.fits"
    )


def read_stackinfo_from_input_stack(mlo, mhi, token):
    """Read NOBJ, EW_MIN, EW_MAX from the input stack FITS STACKINFO table."""
    in_path = _input_stack_path(mlo, mhi, token)
    if not os.path.exists(in_path):
        return None
    try:
        info = Table.read(in_path, hdu="STACKINFO")
    except Exception:
        return None
    if len(info) == 0:
        return None
    out = {}
    if "NOBJ" in info.colnames:
        out["nobj"] = int(info["NOBJ"][0])
    if "EW_MIN" in info.colnames:
        out["ew_min"] = float(info["EW_MIN"][0])
    if "EW_MAX" in info.colnames:
        ew_max = float(info["EW_MAX"][0])
        out["ew_max"] = np.inf if ew_max < 0 else ew_max
    return out


def parse_bin_from_filename(path):
    """Return (mstar_min, mstar_max, ew_token, ew_min, ew_max) or None."""
    m = _FNAME_RE.search(os.path.basename(path))
    if m is None:
        return None
    mlo = float(m.group("mlo"))
    mhi = float(m.group("mhi"))
    token = m.group("ewtoken")
    if token not in EW_TOKENS:
        return None
    info = read_stackinfo_from_input_stack(mlo, mhi, token)
    if info is None or "ew_min" not in info or "ew_max" not in info:
        return None
    return mlo, mhi, token, info["ew_min"], info["ew_max"]


def read_nobj_from_input_stack(mlo, mhi, token):
    """Best-effort NOBJ for a bin from the matching input stack FITS."""
    info = read_stackinfo_from_input_stack(mlo, mhi, token)
    if info is None or "nobj" not in info:
        return -1
    return info["nobj"]


def ew_plot_label(token, ew_min, ew_max):
    """Legend label with fixed-bin name and numeric EW range."""
    base = EW_BIN_LABELS.get(token, token)
    if np.isfinite(ew_max):
        return f"{base} ({ew_min:.1f}–{ew_max:.1f} $\\AA$)"
    return f"{base} ({ew_min:.1f}–$\\infty$ $\\AA$)"


def _line_flux_type_for(line, line_flux_type):
    """Flux column family for a line (O II doublet always uses FLUX)."""
    if line in ("OII_3726", "OII_3729"):
        return "FLUX"
    return line_flux_type


def measure_obs_line_ratio(row, num_line, den_line, line_flux_type):
    """Observed line ratio from FastSpec fluxes on the mean stack (row 0)."""
    num_ftype = _line_flux_type_for(num_line, line_flux_type)
    den_ftype = _line_flux_type_for(den_line, line_flux_type)
    num_col = f"{num_line}_{num_ftype}"
    den_col = f"{den_line}_{den_ftype}"
    num_ivar_col = f"{num_col}_IVAR"
    den_ivar_col = f"{den_col}_IVAR"
    if num_col not in row.colnames or den_col not in row.colnames:
        return np.nan, np.nan
    num = float(row[num_col][0])
    den = float(row[den_col][0])
    num_ivar = (
        float(row[num_ivar_col][0])
        if num_ivar_col in row.colnames else np.nan
    )
    den_ivar = (
        float(row[den_ivar_col][0])
        if den_ivar_col in row.colnames else np.nan
    )
    if not (np.isfinite(num) and np.isfinite(den) and den > 0):
        return np.nan, np.nan
    ratio = num / den
    if (
        np.isfinite(num_ivar) and num_ivar > 0
        and np.isfinite(den_ivar) and den_ivar > 0
    ):
        num_err = 1.0 / np.sqrt(num_ivar)
        den_err = 1.0 / np.sqrt(den_ivar)
        err = np.abs(ratio) * np.sqrt(
            (num_err / num) ** 2 + (den_err / den) ** 2
        )
    else:
        err = np.nan
    return ratio, err


def measure_obs_ha_hb(row):
    """Observed Halpha/Hbeta from boxcar fluxes on the mean stack (row 0)."""
    if "HALPHA_BOXFLUX" not in row.colnames or "HBETA_BOXFLUX" not in row.colnames:
        return np.nan, np.nan, np.nan
    ha = float(row["HALPHA_BOXFLUX"][0])
    hb = float(row["HBETA_BOXFLUX"][0])
    ha_ivar = float(row["HALPHA_BOXFLUX_IVAR"][0]) if "HALPHA_BOXFLUX_IVAR" in row.colnames else np.nan
    hb_ivar = float(row["HBETA_BOXFLUX_IVAR"][0]) if "HBETA_BOXFLUX_IVAR" in row.colnames else np.nan
    if not (np.isfinite(ha) and np.isfinite(hb) and hb > 0):
        return np.nan, np.nan, np.nan
    ratio = ha / hb
    if np.isfinite(ha_ivar) and ha_ivar > 0 and np.isfinite(hb_ivar) and hb_ivar > 0:
        ha_err = 1.0 / np.sqrt(ha_ivar)
        hb_err = 1.0 / np.sqrt(hb_ivar)
        err = np.abs(ratio) * np.sqrt((ha_err / ha) ** 2 + (hb_err / hb) ** 2)
    else:
        err = np.nan
    return ratio, err, err


def intrinsic_ha_hb(ne, te):
    """Case-B Halpha/Hbeta from PyNeb emissivities (no extinction)."""
    from pn_functions import H_alpha, H_beta
    ne = np.asarray(ne, dtype=float)
    te = np.asarray(te, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return H_alpha(te, ne) / H_beta(te, ne)


def pyneb_ne_from_ratio(obs_ratio, te_oiii, density_diagnostic):
    """Independent n_e via PyNeb's getTemDen diagnostic solver.

    Inverts the OBSERVED density-doublet ratio to an electron density at the
    fit's low-ionization temperature tlow = 0.7*T_high + 3000 (Campbell 1986),
    using the SAME atomic data configured in pn_functions. This is a separate
    code path from the nested-sampling fit, so agreement with the fitted
    NE_OII validates that the doublet -> n_e mapping is being used correctly.

    The wave1/wave2 order matches the reported convention:
      SII -> 6716/6731, OII -> 3729/3726 (both decrease with n_e).
    Returns NaN if the ratio is outside the doublet's valid (density-sensitive)
    range or PyNeb fails to converge.
    """
    import pyneb as pn
    import pn_functions  # noqa: F401  (configures pn.atomicData on import)
    from pn_functions import tlow_thi

    if not (np.isfinite(obs_ratio) and np.isfinite(te_oiii)):
        return np.nan
    tlow = float(tlow_thi(te_oiii))
    try:
        if density_diagnostic == "SII":
            atom = pn.Atom("S", 2)
            ne = atom.getTemDen(obs_ratio, tem=tlow, wave1=6716, wave2=6731)
        else:
            atom = pn.Atom("O", 2)
            ne = atom.getTemDen(obs_ratio, tem=tlow, wave1=3729, wave2=3726)
        return float(ne)
    except Exception:
        return np.nan


def _te_line_gating(density_diagnostic, fix_ne=False):
    """Line-SNR mask lines and min_lines for the direct-method TE fit.

    With ``fix_ne`` the electron density is held fixed, so the density
    diagnostic doublet is not needed and the gate reduces to the 7 base
    abundance/Balmer lines regardless of ``density_diagnostic``."""
    names = list(_TE_LINE_NAMES_BASE)
    if not fix_ne and density_diagnostic == "SII":
        names.extend(["SII_6716", "SII_6731"])
    return names, len(names)


def boot_spread(values):
    """16/50/84 spread of finite values -> (err, err_lo, err_hi, n_used)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, 0
    p16, p50, p84 = np.nanpercentile(arr, [16, 50, 84])
    return 0.5 * (p84 - p16), p50 - p16, p84 - p50, int(arr.size)


def _fill_param_errors(rec, lo_name, up, mean_fit, boot_ok):
    """Write central value and asymmetric errors (bootstrap or posterior)."""
    val = float(mean_fit[lo_name])
    lo = float(mean_fit[f"{lo_name}_lo"])
    hi = float(mean_fit[f"{lo_name}_hi"])
    rec[up] = val
    rec[f"{up}_MEANFIT_LO"] = lo
    rec[f"{up}_MEANFIT_HI"] = hi
    if len(boot_ok) > 0:
        err, err_lo, err_hi, _ = boot_spread(boot_ok[lo_name])
    else:
        err_lo = val - lo
        err_hi = hi - val
        err = 0.5 * (err_lo + err_hi)
    rec[f"{up}_ERR"] = err
    rec[f"{up}_ERR_LO"] = err_lo
    rec[f"{up}_ERR_HI"] = err_hi


def _fill_intrinsic_ha_hb_errors(rec, mean_fit, boot_ok):
    """Populate HA_HB_INTRINSIC and its error columns."""
    ne_med = rec["NE_OII"]
    te_med = rec["TE_OIII"]
    if np.isfinite(ne_med) and np.isfinite(te_med):
        rec["HA_HB_INTRINSIC"] = float(intrinsic_ha_hb(ne_med, te_med))

    if len(boot_ok) > 0:
        boot_ratios = intrinsic_ha_hb(
            np.asarray(boot_ok["ne_oii"], dtype=float),
            np.asarray(boot_ok["te_oiii"], dtype=float),
        )
        err, err_lo, err_hi, _ = boot_spread(boot_ratios)
    elif np.isfinite(rec["HA_HB_INTRINSIC"]):
        ne_vals = [
            float(mean_fit["ne_oii_lo"]),
            float(mean_fit["ne_oii"]),
            float(mean_fit["ne_oii_hi"]),
        ]
        te_vals = [
            float(mean_fit["te_oiii_lo"]),
            float(mean_fit["te_oiii"]),
            float(mean_fit["te_oiii_hi"]),
        ]
        ratios = [
            intrinsic_ha_hb(ne, te)
            for ne in ne_vals for te in te_vals
            if np.isfinite(ne) and np.isfinite(te)
        ]
        err, err_lo, err_hi, _ = boot_spread(ratios)
    else:
        return

    rec["HA_HB_INTRINSIC_ERR"] = err
    rec["HA_HB_INTRINSIC_ERR_LO"] = err_lo
    rec["HA_HB_INTRINSIC_ERR_HI"] = err_hi


def _mstar_mid(tab):
    return 0.5 * (np.asarray(tab["MSTAR_MIN"]) + np.asarray(tab["MSTAR_MAX"]))


def print_density_diagnostic_verification(out_tab, density_diagnostic):
    """Print n_e sanity checks for the chosen density diagnostic."""
    ne_prior_lo, ne_prior_hi = 10.0, 5000.0
    print(f"\n[2b] Density-diagnostic verification ({density_diagnostic}):")

    good = out_tab[
        (out_tab["DETECTED_7LINE"]) & (out_tab["MEAN_FIT_SUCCESS"])
    ]
    if len(good) == 0:
        print("    No successful fits; skipping n_e checks.")
        return

    ne = np.asarray(good["NE_OII"], dtype=float)
    finite_ne = ne[np.isfinite(ne)]
    if finite_ne.size == 0:
        print("    No finite NE_OII values in successful fits.")
        return

    in_prior = (finite_ne >= ne_prior_lo) & (finite_ne <= ne_prior_hi)
    print(
        f"    NE_OII (fitted n_e): N={finite_ne.size}, "
        f"median={np.median(finite_ne):.1f} cm^-3, "
        f"range=[{finite_ne.min():.1f}, {finite_ne.max():.1f}]"
    )
    print(
        f"    Within prior [{ne_prior_lo:g}, {ne_prior_hi:g}] cm^-3: "
        f"{int(np.sum(in_prior))}/{finite_ne.size}"
    )

    if density_diagnostic == "SII":
        obs_col = "OBS_SII_DOUBLET"
        obs_label = "[S II] 6716/6731"
    else:
        obs_col = "OBS_OII_DOUBLET"
        obs_label = "[O II] 3729/3726"

    if obs_col not in good.colnames:
        return

    obs = np.asarray(good[obs_col], dtype=float)
    ok = np.isfinite(ne) & np.isfinite(obs)
    if int(np.sum(ok)) < 3:
        print(f"    Too few finite {obs_label} ratios for trend check.")
        return

    corr = np.corrcoef(obs[ok], ne[ok])[0, 1]
    print(
        f"    Pearson r({obs_label}, NE_OII) = {corr:.3f} "
        f"(expect negative: higher doublet ratio -> lower n_e)"
    )

    # Independent PyNeb getTemDen cross-check: invert the observed doublet
    # ratio to n_e at the fitted tlow and compare to the fitted NE_OII. This
    # is a separate solver from the nested-sampling fit, so agreement confirms
    # the doublet -> n_e mapping is being applied correctly.
    te = np.asarray(good["TE_OIII"], dtype=float)
    ne_pyneb = np.array([
        pyneb_ne_from_ratio(o, t, density_diagnostic)
        for o, t in zip(obs, te)
    ])
    both = np.isfinite(ne) & np.isfinite(ne_pyneb)
    print(f"    PyNeb getTemDen cross-check (invert observed {obs_label} at tlow):")
    if np.any(both):
        frac = (ne[both] - ne_pyneb[both]) / ne_pyneb[both]
        print(
            f"      N={int(np.sum(both))}: median fitted NE_OII="
            f"{np.median(ne[both]):.1f}, median getTemDen="
            f"{np.median(ne_pyneb[both]):.1f} cm^-3"
        )
        print(
            f"      median fractional diff (fit-pyneb)/pyneb = "
            f"{np.median(frac):+.3f} "
            f"[range {np.min(frac):+.3f}, {np.max(frac):+.3f}]"
        )
    else:
        print("      No bins with both finite fitted and getTemDen n_e "
              "(ratios likely outside the density-sensitive range).")

    high_mass = good[np.asarray(good["MSTAR_MIN"], dtype=float) >= 9.0]
    if len(high_mass) > 0:
        hm_fit = high_mass[
            (high_mass["DETECTED_7LINE"]) & (high_mass["MEAN_FIT_SUCCESS"])
        ]
        print(
            f"    log M* >= 9.0 bins: {len(high_mass)} total, "
            f"{len(hm_fit)} with successful fits"
        )
        for row in hm_fit:
            mlo, mhi = float(row["MSTAR_MIN"]), float(row["MSTAR_MAX"])
            token = row["EW_TOKEN"]
            ne_val = float(row["NE_OII"])
            obs_val = float(row[obs_col]) if np.isfinite(row[obs_col]) else np.nan
            ne_pn = pyneb_ne_from_ratio(
                obs_val, float(row["TE_OIII"]), density_diagnostic
            )
            print(
                f"      [{mlo:.2f},{mhi:.2f}] {token}: "
                f"NE_OII={ne_val:.1f}, getTemDen={ne_pn:.1f}, "
                f"{obs_col}={obs_val:.3f}, "
                f"12+log(O/H)={float(row['TWELVE_LOG_OH']):.3f}"
            )


def _plot_vs_mstar_by_ew(tab, ycol, yerr_lo, yerr_hi, ylab, ax, title=None):
    """Errorbar plot vs log M* midpoint, one series per EW_TOKEN.

    Solid: bootstrap errors reliable (N_BOOT_FIT >= MIN_N_BOOT_FIT).
    Dotted triangles: detected but bootstrap survivor fraction too low.
    """
    mid = _mstar_mid(tab)
    has_unreliable = False
    for token in EW_TOKENS:
        sel = np.asarray(tab["EW_TOKEN"]) == token
        if not np.any(sel):
            continue
        sub = tab[sel]
        label = ew_plot_label(
            token,
            float(sub["EW_MIN"][0]),
            float(sub["EW_MAX"][0]) if sub["EW_MAX"][0] >= 0 else np.inf,
        )
        mid_sub = mid[sel]
        y = np.asarray(sub[ycol], dtype=float)
        err_lo = np.asarray(sub[yerr_lo], dtype=float)
        err_hi = np.asarray(sub[yerr_hi], dtype=float)
        if "BOOT_ERR_RELIABLE" in sub.colnames:
            reliable = np.asarray(sub["BOOT_ERR_RELIABLE"], dtype=bool)
        else:
            reliable = np.ones(len(sub), dtype=bool)

        has_err = np.isfinite(err_lo) & np.isfinite(err_hi)
        rel = reliable & has_err
        unrel = (~reliable) & has_err

        if np.any(rel):
            ax.errorbar(
                mid_sub[rel], y[rel],
                yerr=[err_lo[rel], err_hi[rel]],
                fmt="o-", color=EW_COLORS[token], label=label,
                capsize=3, lw=1.2,
            )
        elif np.any(reliable):
            ax.plot(mid_sub[reliable], y[reliable], "o-",
                    color=EW_COLORS[token], label=label, lw=1.2)
        else:
            ax.plot([], [], "o-", color=EW_COLORS[token], label=label)

        if np.any(unrel):
            has_unreliable = True
            ax.errorbar(
                mid_sub[unrel], y[unrel],
                yerr=[err_lo[unrel], err_hi[unrel]],
                fmt="^:", color=EW_COLORS[token], capsize=3, lw=1.0, alpha=0.75,
            )

    ax.set_xlabel(r"log $M_\star$ [$M_\odot$]")
    ax.set_ylabel(ylab)
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D(
        [], [], color="k", marker="o", ls="-", lw=1.2,
        label="bootstrap error reliable",
    ))
    labels.append("bootstrap error reliable")
    if has_unreliable:
        handles.append(Line2D(
            [], [], color="k", marker="^", ls=":", lw=1.0, alpha=0.75,
            label="low bootstrap survivor fraction",
        ))
        labels.append("low bootstrap survivor fraction")
    ax.legend(handles, labels, frameon=False, fontsize=7)
    if title:
        ax.set_title(title, fontsize=10)


# =============================================================================
# MAIN
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Direct-method nebular abundances for M* x H-alpha-EW stacked "
            "spectra from FastSpecFit stack outputs."
        ),
    )
    parser.add_argument(
        "--line-flux-type",
        required=True,
        choices=("FLUX", "BOXFLUX"),
        help=(
            "FastSpec line-flux family for line detection and direct-method "
            "fits (FLUX Gaussian or BOXFLUX boxcar). Required."
        ),
    )
    parser.add_argument(
        "--density-diagnostic",
        choices=("OII", "SII"),
        default=None,
        help=(
            "Low-ionization doublet for electron density in the direct-method "
            "fit: OII ([O II] 3726/3729) or SII ([S II] 6716/6731). "
            "Default: SII (TE_DENSITY_DIAGNOSTIC)."
        ),
    )
    parser.add_argument(
        "--fix-ne100",
        action="store_true",
        help=(
            "Hold the electron density fixed at n_e = 100 cm^-3 instead of "
            "fitting it from the SII/OII density doublet. Drops the density "
            "doublet from both the direct-method fit and the detection gate, "
            "so --density-diagnostic becomes irrelevant."
        ),
    )
    args = parser.parse_args(argv)

    density_diagnostic = args.density_diagnostic or TE_DENSITY_DIAGNOSTIC
    fix_ne = args.fix_ne100
    fixed_ne = FIXED_NE_VALUE if fix_ne else None
    te_line_names, te_min_lines = _te_line_gating(density_diagnostic, fix_ne=fix_ne)

    import sfr_and_metallicity as sam
    sam.line_flux_type = args.line_flux_type

    plot_dir = os.path.join(STACK_PATH, "plots")

    from pn_functions import compute_direct_metallicities

    pattern = os.path.join(STACK_PATH, INPUT_GLOB)
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "fastspec_fastspec_" not in os.path.basename(f)]
    print(f"[1] Found {len(files)} FastSpecFit stack outputs in {STACK_PATH}")
    if fix_ne:
        print(f"    FIX n_e: holding n_e = {FIXED_NE_VALUE:g} cm^-3 fixed "
              f"(density doublet dropped from fit AND gate; "
              f"--density-diagnostic ignored)")
    else:
        print(f"    TE density diagnostic: {density_diagnostic} "
              f"({te_min_lines} lines required for detection gate)")
    print(f"    Detection gate: SNR > {SNR_VAL}, flux > {MIN_FLUX:g} in all "
          f"{te_min_lines} lines, on the MEAN stack (row 0) only; bootstrap "
          f"rows passing the same gate are fit for the error.")
    print(
        "    NOTE: propagated measurement ivar on row-0 stacks may increase "
        "DETECTED_7LINE pass rate vs the old 1/boot_std^2 ivar (expected)."
    )
    print(
        f"    UltraNest: n_jobs={N_JOBS} parallelizes per-row fits "
        f"(~{N_BOOT_TOTAL + 1} rows/bin when bootstrap stacks are present)."
    )
    if len(files) == 0:
        print(f"    Nothing matches {pattern}; exiting.")
        return 1

    rows = []

    for fpath in files:
        parsed = parse_bin_from_filename(fpath)
        if parsed is None:
            print(f"  ! Skipping unparsable filename: {os.path.basename(fpath)}")
            continue
        mlo, mhi, token, ew_min, ew_max = parsed
        nobj = read_nobj_from_input_stack(mlo, mhi, token)

        ew_hi_str = f"{ew_max:.1f}" if np.isfinite(ew_max) else "inf"
        print(f"\n--- log M*=[{mlo:.2f},{mhi:.2f}] | {token} "
              f"(EW in ({ew_min:.1f},{ew_hi_str}]) | NOBJ={nobj} ---")

        try:
            t = Table.read(fpath, hdu=FASTSPEC_HDU)
        except Exception as e:
            print(f"    Could not read hdu={FASTSPEC_HDU}: {e}; skipping.")
            continue

        if len(t) == 0:
            print("    Empty fastspec table; skipping.")
            continue

        obs_ha_hb, obs_err, _ = measure_obs_ha_hb(t[[0]])
        # [O II] 3729/3726 (standard convention: decreases with n_e, mirrors
        # [S II] 6716/6731). Numerator/denominator order matches DENSITY_RATIO_SPECS.
        obs_oii, obs_oii_err = measure_obs_line_ratio(
            t[[0]], "OII_3729", "OII_3726", args.line_flux_type,
        )
        obs_sii, obs_sii_err = measure_obs_line_ratio(
            t[[0]], "SII_6716", "SII_6731", args.line_flux_type,
        )

        # Gate every row (mean + bootstrap) with the same SNR/flux cut. Row 0
        # is the total/mean stack; rows 1.. are the bootstrap realizations.
        gate_mask = line_snr_mask(
            t, line_names=te_line_names,
            snr_val=SNR_VAL, min_lines=te_min_lines, min_flux=MIN_FLUX,
            line_flux_type=args.line_flux_type,
        )
        detected = bool(gate_mask[0])

        rec = {
            "MSTAR_MIN": mlo, "MSTAR_MAX": mhi,
            "EW_MIN": ew_min,
            "EW_MAX": ew_max if np.isfinite(ew_max) else -1.0,
            "EW_TOKEN": token,
            "NOBJ": nobj,
            "DENSITY_DIAGNOSTIC": density_diagnostic,
            "FIX_NE100": fix_ne,
            "DETECTED_7LINE": detected,
            "MEAN_FIT_SUCCESS": False,
            "N_BOOT_TOTAL": 0,
            "N_BOOT_PASS_GATE": 0,
            "N_BOOT_FIT": 0,
            "BOOT_ERR_RELIABLE": False,
            "N_RATIOS": 0,
            "LOGZ": np.nan,
            "OBS_HA_HB": obs_ha_hb,
            "OBS_HA_HB_ERR": obs_err,
            "OBS_OII_DOUBLET": obs_oii,
            "OBS_OII_DOUBLET_ERR": obs_oii_err,
            "OBS_SII_DOUBLET": obs_sii,
            "OBS_SII_DOUBLET_ERR": obs_sii_err,
            "HA_HB_INTRINSIC": np.nan,
            "HA_HB_INTRINSIC_ERR": np.nan,
            "HA_HB_INTRINSIC_ERR_LO": np.nan,
            "HA_HB_INTRINSIC_ERR_HI": np.nan,
        }
        for _, up in PARAM_MAP:
            rec[up] = np.nan
            rec[f"{up}_ERR"] = np.nan
            rec[f"{up}_ERR_LO"] = np.nan
            rec[f"{up}_ERR_HI"] = np.nan
            rec[f"{up}_MEANFIT_LO"] = np.nan
            rec[f"{up}_MEANFIT_HI"] = np.nan

        if not detected:
            print(f"    Mean stack does NOT detect all {te_min_lines} lines "
                  f"(SNR>{SNR_VAL}, flux>{MIN_FLUX:g}); recording as undetected.")
            rows.append(rec)
            continue

        # Fit ONLY the mean stack (row 0) plus the bootstrap rows that
        # independently pass the same line gate -- bootstrap realizations that
        # fail the gate are dropped before fitting, so no nested-sampling cycles
        # are spent on rows that wouldn't have been detected.
        n_boot_raw = max(len(t) - 1, 0)
        boot_pass_idx = np.where(gate_mask[1:])[0] + 1   # +1: skip mean row 0
        fit_idx = [0] + boot_pass_idx.tolist()
        fit_tab = t[fit_idx]
        rec["N_BOOT_TOTAL"] = int(n_boot_raw)
        rec["N_BOOT_PASS_GATE"] = int(len(boot_pass_idx))

        print(f"    Detected. Fitting 1 mean + "
              f"{rec['N_BOOT_PASS_GATE']}/{n_boot_raw} gate-passing bootstrap "
              f"row(s) with UltraNest ...")
        res = compute_direct_metallicities(
            fit_tab,
            args.line_flux_type,
            n_jobs=N_JOBS,
            min_num_live_points=MIN_NUM_LIVE_POINTS,
            sampler_kwargs=SAMPLER_KWARGS,
            use_informative_priors=USE_INFORMATIVE_PRIORS,
            density_diagnostic=density_diagnostic,
            fixed_ne=fixed_ne,
            verbose=False,
            verbose_sampler=False,
        )

        mean_fit = res[0]
        rec["MEAN_FIT_SUCCESS"] = bool(mean_fit["fit_success"])
        rec["N_RATIOS"] = int(mean_fit["n_ratios"])
        rec["LOGZ"] = float(mean_fit["logz"])

        if len(res) > 1:
            boot = res[1:]
            ok = np.asarray(boot["fit_success"], dtype=bool) & np.isfinite(
                np.asarray(boot["twelve_log_OH"], dtype=float)
            )
            boot_ok = boot[ok]
        else:
            boot_ok = res[:0]
        rec["N_BOOT_FIT"] = int(len(boot_ok))
        rec["BOOT_ERR_RELIABLE"] = rec["N_BOOT_FIT"] >= MIN_N_BOOT_FIT

        for lo_name, up in PARAM_MAP:
            _fill_param_errors(rec, lo_name, up, mean_fit, boot_ok)

        _fill_intrinsic_ha_hb_errors(rec, mean_fit, boot_ok)

        print(f"    -> 12+log(O/H) = {rec['TWELVE_LOG_OH']:.3f} "
              f"(+{rec['TWELVE_LOG_OH_ERR_HI']:.3f}/-{rec['TWELVE_LOG_OH_ERR_LO']:.3f}), "
              f"A_V = {rec['AV']:.3f} "
              f"(+{rec['AV_ERR_HI']:.3f}/-{rec['AV_ERR_LO']:.3f}); "
              f"N_boot_fit={rec['N_BOOT_FIT']}/{rec['N_BOOT_TOTAL']}, BOOT_ERR_RELIABLE={rec['BOOT_ERR_RELIABLE']}")
        rows.append(rec)

    if len(rows) == 0:
        print("\n[2] No bins processed; nothing to write.")
        return 1

    out_tab = Table(rows)
    col_order = [
        "MSTAR_MIN", "MSTAR_MAX", "EW_MIN", "EW_MAX", "EW_TOKEN", "NOBJ",
        "DENSITY_DIAGNOSTIC", "FIX_NE100", "DETECTED_7LINE", "MEAN_FIT_SUCCESS",
        "N_BOOT_TOTAL", "N_BOOT_PASS_GATE", "N_BOOT_FIT", "BOOT_ERR_RELIABLE",
        "N_RATIOS", "LOGZ",
        "OBS_HA_HB", "OBS_HA_HB_ERR",
        "OBS_OII_DOUBLET", "OBS_OII_DOUBLET_ERR",
        "OBS_SII_DOUBLET", "OBS_SII_DOUBLET_ERR",
        "HA_HB_INTRINSIC", "HA_HB_INTRINSIC_ERR",
        "HA_HB_INTRINSIC_ERR_LO", "HA_HB_INTRINSIC_ERR_HI",
    ]
    for _, up in PARAM_MAP:
        col_order += [up, f"{up}_ERR", f"{up}_ERR_LO", f"{up}_ERR_HI",
                      f"{up}_MEANFIT_LO", f"{up}_MEANFIT_HI"]
    out_tab = out_tab[[c for c in col_order if c in out_tab.colnames]]

    out_fits = os.path.join(STACK_PATH, f"{OUT_BASENAME}.fits")
    out_ecsv = os.path.join(STACK_PATH, f"{OUT_BASENAME}.ecsv")
    out_tab.write(out_fits, overwrite=True)
    out_tab.write(out_ecsv, format="ascii.ecsv", overwrite=True)

    n_det = int(np.sum(out_tab["DETECTED_7LINE"]))
    n_fit = int(np.sum(out_tab["MEAN_FIT_SUCCESS"]))
    print(f"\n[2] Wrote {len(out_tab)} bin rows "
          f"({n_det} detected, {n_fit} with successful mean-stack fit):")
    print(f"    {out_fits}")
    print(f"    {out_ecsv}")

    make_all_plots(out_tab, plot_dir)
    print_density_diagnostic_verification(out_tab, density_diagnostic)

    print("\n[3] Done.")
    return 0


def _plot_obs_vs_mstar_by_ew(tab, ycol, yerr_col, ylab, ax, title=None):
    """Observed quantity vs log M* midpoint, one series per EW_TOKEN.

    Solid markers: bin passed the TE line-SNR gate (DETECTED_7LINE).
    Dashed hollow markers: finite ratio but TE gate failed (not fitted).
    """
    finite = np.isfinite(tab[ycol])
    if not np.any(finite):
        return False

    plot_tab = tab[finite]
    has_undetected = False
    for token in EW_TOKENS:
        sel = np.asarray(plot_tab["EW_TOKEN"]) == token
        if not np.any(sel):
            continue
        sub = plot_tab[sel]
        mid_sub = _mstar_mid(sub)
        label = ew_plot_label(
            token,
            float(sub["EW_MIN"][0]),
            float(sub["EW_MAX"][0]) if sub["EW_MAX"][0] >= 0 else np.inf,
        )
        y = np.asarray(sub[ycol])
        yerr = np.asarray(sub[yerr_col])
        det = np.asarray(sub["DETECTED_7LINE"], dtype=bool)
        has_err = np.isfinite(yerr) & (yerr > 0)

        if np.any(det & has_err):
            m = det & has_err
            ax.errorbar(
                mid_sub[m], y[m], yerr=yerr[m],
                fmt="o-", color=EW_COLORS[token], label=label,
                capsize=3, lw=1.2,
            )
        elif np.any(det):
            ax.plot(
                mid_sub[det], y[det], "o-",
                color=EW_COLORS[token], label=label, lw=1.2,
            )
        else:
            ax.plot([], [], "o-", color=EW_COLORS[token], label=label)

        if np.any(~det):
            has_undetected = True
            ax.plot(
                mid_sub[~det], y[~det],
                marker="o", fillstyle="none", ls="--", lw=1.0,
                color=EW_COLORS[token], alpha=0.7,
            )

    ax.set_xlabel(r"log $M_\star$ [$M_\odot$]")
    ax.set_ylabel(ylab)
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D(
        [], [], color="k", marker="o", ls="-", lw=1.2,
        label="TE lines detected",
    ))
    labels.append("TE lines detected")
    if has_undetected:
        handles.append(Line2D(
            [], [], color="k", marker="o", fillstyle="none", ls="--",
            lw=1.0, alpha=0.7, label="ratio only (TE gate failed)",
        ))
        labels.append("ratio only (TE gate failed)")
    ax.legend(handles, labels, frameon=False, fontsize=7)
    if title:
        ax.set_title(title, fontsize=10)
    return True


def make_all_plots(out_tab, plot_dir):
    """Write oh_av, te_ne_hahb, obs_hahb, and doublet-ratio vs M* figures."""
    for plot_fn in (
        make_oh_av_plot, make_te_ne_hahb_plot, make_obs_hahb_plot,
        make_doublet_ratios_vs_mstar_plot, make_ne_diagnostic_plot,
    ):
        try:
            plot_fn(out_tab, plot_dir)
        except Exception as e:
            print(f"    ({plot_fn.__name__} skipped: {e})")


def make_oh_av_plot(out_tab, plot_dir):
    """12+log(O/H) and A_V vs M* (bin midpoint), one series per EW bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    good = out_tab[(out_tab["DETECTED_7LINE"]) & (out_tab["MEAN_FIT_SUCCESS"])]
    if len(good) == 0:
        print("    (no successful fits; skipping oh_av_vs_mstar)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col, ax, ylab in [
        ("TWELVE_LOG_OH", axes[0], "12 + log(O/H)"),
        ("AV", axes[1], r"$A_V$ [mag]"),
    ]:
        _plot_vs_mstar_by_ew(
            good, col, f"{col}_ERR_LO", f"{col}_ERR_HI", ylab, ax,
        )

    fig.suptitle("Direct-method stacks: trends vs stellar mass")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(plot_dir, "oh_av_vs_mstar.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


def make_te_ne_hahb_plot(out_tab, plot_dir):
    """n_e, T_e, and intrinsic Halpha/Hbeta vs M* for successful fits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    good = out_tab[(out_tab["DETECTED_7LINE"]) & (out_tab["MEAN_FIT_SUCCESS"])]
    if len(good) == 0:
        print("    (no successful fits; skipping te_ne_hahb_vs_mstar)")
        return

    dens_diag = "SII"
    if "DENSITY_DIAGNOSTIC" in good.colnames and len(good) > 0:
        uniq = np.unique(np.asarray(good["DENSITY_DIAGNOSTIC"]))
        if len(uniq) == 1:
            dens_diag = str(uniq[0])
    ne_ylab = (
        r"$n_e$ ([S II]) [cm$^{-3}$]" if dens_diag == "SII"
        else r"$n_e$ ([O II]) [cm$^{-3}$]"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for col, ax, ylab in [
        ("NE_OII", axes[0], ne_ylab),
        ("TE_OIII", axes[1], r"$T_e$ [K]"),
        ("HA_HB_INTRINSIC", axes[2], r"H$\alpha$/H$\beta$ (intrinsic)"),
    ]:
        _plot_vs_mstar_by_ew(
            good, col, f"{col}_ERR_LO", f"{col}_ERR_HI", ylab, ax,
        )

    fig.suptitle(
        f"Direct-method stacks ($n_e$ from {dens_diag}): "
        r"$n_e$, $T_e$, intrinsic Balmer ratio"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(plot_dir, "te_ne_hahb_vs_mstar.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


def make_obs_hahb_plot(out_tab, plot_dir):
    """Observed HALPHA_BOXFLUX / HBETA_BOXFLUX vs M* (all finite ratios)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if not _plot_obs_vs_mstar_by_ew(
        out_tab, "OBS_HA_HB", "OBS_HA_HB_ERR",
        r"H$\alpha$/H$\beta$ (observed boxcar)", ax,
        title="Observed Balmer decrement (mean stack)",
    ):
        plt.close(fig)
        print("    (no finite OBS_HA_HB; skipping obs_hahb_vs_mstar)")
        return

    fig.suptitle(
        "Observed Balmer decrement (mean stack); "
        "dashed = ratio measured but TE lines not all detected",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(plot_dir, "obs_hahb_vs_mstar.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


def make_doublet_ratios_vs_mstar_plot(out_tab, plot_dir):
    """Observed [S II] and [O II] doublet ratios vs M* (all finite ratios)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    has_sii = np.isfinite(out_tab["OBS_SII_DOUBLET"])
    has_oii = np.isfinite(out_tab["OBS_OII_DOUBLET"])
    if not (np.any(has_sii) or np.any(has_oii)):
        print("    (no finite doublet ratios; skipping doublet_ratios_vs_mstar)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [
        ("OBS_SII_DOUBLET", "OBS_SII_DOUBLET_ERR",
         axes[0], r"[S II] 6716/6731 (observed)"),
        ("OBS_OII_DOUBLET", "OBS_OII_DOUBLET_ERR",
         axes[1], r"[O II] 3729/3726 (observed FLUX)"),
    ]
    for ycol, yerr_col, ax, ylab in panels:
        if not _plot_obs_vs_mstar_by_ew(out_tab, ycol, yerr_col, ylab, ax):
            ax.set_title(f"{ylab} (no finite values)", fontsize=10)
            ax.axis("off")

    fig.suptitle(
        "Density-diagnostic doublet ratios (mean stack); "
        "dashed = ratio measured but TE lines not all detected",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(plot_dir, "doublet_ratios_vs_mstar.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


def make_ne_diagnostic_plot(out_tab, plot_dir):
    """Density-diagnostic ratio (y) vs electron density (x).

    Two panels ([S II] 6716/6731 and [O II] 3729/3726). Each shows the PyNeb
    model curve built from the SAME emissivity interpolators the direct-method
    fit uses (pn_functions), drawn at T_e = 1e4 K with a shaded band over
    T_e in [7000, 15000] K to expose the (weak) temperature dependence.
    Horizontal lines mark the observed doublet ratio of each detected stack
    (DETECTED_7LINE), colored by EW bin, with a faint band for the ratio error.
    Reading where a horizontal line crosses the curve gives the implied n_e.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pn_functions import O2_3726, O2_3729, S2_6716, S2_6731, den_min, den_max

    os.makedirs(plot_dir, exist_ok=True)

    ne_grid = np.logspace(np.log10(den_min), np.log10(den_max), 200)
    te_fid = 1.0e4
    te_band = (7.0e3, 1.5e4)

    def _curve(num_g, den_g, te):
        te_arr = np.full_like(ne_grid, te)
        return num_g(te_arr, ne_grid) / den_g(te_arr, ne_grid)

    # (tag, obs col, obs err col, numerator grid, denominator grid, y-label)
    panels = [
        ("SII", "OBS_SII_DOUBLET", "OBS_SII_DOUBLET_ERR",
         S2_6716, S2_6731, r"[S II] $\lambda6716/\lambda6731$"),
        ("OII", "OBS_OII_DOUBLET", "OBS_OII_DOUBLET_ERR",
         O2_3729, O2_3726, r"[O II] $\lambda3729/\lambda3726$"),
    ]

    det = np.asarray(out_tab["DETECTED_7LINE"], dtype=bool)
    sub = out_tab[det]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (tag, ycol, yerr_col, num_g, den_g, ylab) in zip(axes, panels):
        c_fid = _curve(num_g, den_g, te_fid)
        c_lo = _curve(num_g, den_g, te_band[0])
        c_hi = _curve(num_g, den_g, te_band[1])
        band_lo = np.minimum(c_lo, c_hi)
        band_hi = np.maximum(c_lo, c_hi)
        ax.fill_between(ne_grid, band_lo, band_hi, color="0.7", alpha=0.4,
                        label=r"$T_e\in[7,15]\times10^3$ K")
        ax.plot(ne_grid, c_fid, "k-", lw=1.8, label=r"PyNeb ($T_e=10^4$ K)")

        ew_present = []
        if ycol in sub.colnames:
            for token in EW_TOKENS:
                rows = sub[np.asarray(sub["EW_TOKEN"]) == token]
                drawn = False
                for row in rows:
                    val = float(row[ycol]) if np.isfinite(row[ycol]) else np.nan
                    if not np.isfinite(val):
                        continue
                    color = EW_COLORS.get(token, "0.3")
                    ax.axhline(val, color=color, lw=1.0, alpha=0.85)
                    err = (float(row[yerr_col])
                           if yerr_col in row.colnames
                           and np.isfinite(row[yerr_col]) else np.nan)
                    if np.isfinite(err) and err > 0:
                        ax.axhspan(val - err, val + err, color=color, alpha=0.10)
                    drawn = True
                if drawn:
                    ew_present.append(token)

        ax.set_xscale("log")
        ax.set_xlim(den_min, den_max)
        ax.set_xlabel(r"$n_e$ [cm$^{-3}$]")
        ax.set_ylabel(ylab)
        ax.set_title(f"{tag} density diagnostic", fontsize=10)

        handles, labels = ax.get_legend_handles_labels()
        for token in ew_present:
            handles.append(Line2D([], [], color=EW_COLORS[token], lw=1.0))
            labels.append(EW_BIN_LABELS.get(token, token))
        ax.legend(handles, labels, frameon=False, fontsize=7)

    fig.suptitle(
        "Density-diagnostic ratio vs $n_e$: PyNeb model curves and observed "
        "stack ratios (detected bins, colored by EW)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(plot_dir, "ne_diagnostic_ratio_vs_density.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    Saved {os.path.basename(out_png)}")


if __name__ == "__main__":
    sys.exit(main())
