"""
Reproduce Korhonen Cuestas+ 2025 Table 2 conversion factors C(Z*) using
BPASS v2.2.1 binary models with Chabrier IMF (M_max = 100 M_sun).

Method (paper Section 2.4):
  1. For each metallicity Z*, take the BPASS instantaneous-burst SED time
     series (51 log-spaced age bins, normalised to 10^6 M_sun).
  2. Co-add bursts weighted by bin width, for ages <= 100 Myr, to build
     the composite SED of a constant-SFR-for-100-Myr population.
  3. Integrate SED from 0 to 912 A to get the ionizing photon rate N(H0).
  4. L(Halpha) = 1.36e-12 * N(H0)        [Leitherer & Heckman 1995]
  5. C(Z*) = L(Halpha) / SFR.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot_style import apply_paper_style, make_subplots, MARGIN_LABEL, MARGIN_PAD

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# ---------------------------------------------------------------------------
# Constants (CGS)
# ---------------------------------------------------------------------------
H_PLANCK = 6.62607015e-27
C_LIGHT  = 2.99792458e10
LSUN     = 3.828e33
HC_ERG_A = H_PLANCK * C_LIGHT * 1e8            # h*c with lambda in Angstroms
L_HALPHA_PER_N_ION = 1.36e-12                  # Leitherer & Heckman 1995

# ---------------------------------------------------------------------------
# BPASS age grid: 51 bins, log(age/yr) = 6.0, 6.1, ..., 11.0
# ---------------------------------------------------------------------------
LOG_AGES   = np.arange(6.0, 11.01, 0.1)
_lo        = 10.0 ** (LOG_AGES - 0.05)
_hi        = 10.0 ** (LOG_AGES + 0.05)
_lo[0]     = 0.0                                # extend youngest bin to t=0
BIN_WIDTHS = _hi - _lo                          # years

# ---------------------------------------------------------------------------
# Inputs (edit BPASS_ROOT to point at your folder)
# ---------------------------------------------------------------------------
BPASS_ROOT = Path("/Users/virajmanwadkar/Desktop/Stanford/Research/fun_research/bpass/BPASS_to_Viraj")
CACHE_PATH = BPASS_ROOT / "cache" / "bpass_C_Zstar_t100Myr.npz"
MMAX_VARIANTS = {"100": "chab100", "300": "chab300"}

COLOR_BINARY = "#7b3294"
COLOR_SINGLE = "#008837"

SFR        = 1.0
T_CONST    = 100e6
LAMBDA_MAX = 912.0
BURST_MASS = 1.0e6
Z_SOLAR    = 0.020                              # BPASS / paper convention

# All 13 BPASS metallicities (extending below the paper's Table 2)
METALLICITIES = {
    "zem5": 1e-5, "zem4": 1e-4,
    "z001": 0.001, "z002": 0.002, "z003": 0.003, "z004": 0.004,
    "z006": 0.006, "z008": 0.008, "z010": 0.010, "z014": 0.014,
    "z020": 0.020, "z030": 0.030, "z040": 0.040,
}


def bpass_dir(mmax):
    return BPASS_ROOT / f"bpass_v2.2.1_imf_{MMAX_VARIANTS[mmax]}"


def spectrum_path(pop, mmax, tag):
    """pop: 'bin' or 'sin'; mmax: '100' or '300'."""
    return bpass_dir(mmax) / f"spectra-{pop}-imf_{MMAX_VARIANTS[mmax]}.{tag}.dat.gz"


# ---------------------------------------------------------------------------
def log_C_from_spectrum(spectrum_file):
    """log10(C(Z*)) for one BPASS spectra file."""
    data = np.loadtxt(spectrum_file)            # numpy reads .gz natively
    wl   = data[:, 0]                           # Angstroms
    sed  = data[:, 1:]                          # L_sun/A, 51 age columns

    include = LOG_AGES <= np.log10(T_CONST)
    weights = SFR * BIN_WIDTHS[include] / BURST_MASS

    f_lambda = (sed[:, include] @ weights) * LSUN   # erg/s/A

    ion   = wl < LAMBDA_MAX
    N_ion = np.trapz(f_lambda[ion] * wl[ion], wl[ion]) / HC_ERG_A

    return np.log10(L_HALPHA_PER_N_ION * N_ion / SFR)


def _compute_logC_arrays():
    tags = list(METALLICITIES.keys())
    Z_arr = np.array([METALLICITIES[t] for t in tags])
    logC_bin_100, logC_bin_300 = [], []
    logC_sin_100, logC_sin_300 = [], []
    for tag in tags:
        logC_bin_100.append(log_C_from_spectrum(spectrum_path("bin", "100", tag)))
        logC_bin_300.append(log_C_from_spectrum(spectrum_path("bin", "300", tag)))
        logC_sin_100.append(log_C_from_spectrum(spectrum_path("sin", "100", tag)))
        logC_sin_300.append(log_C_from_spectrum(spectrum_path("sin", "300", tag)))
    return {
        "Z": Z_arr,
        "log_Z_Zsol": np.log10(Z_arr / Z_SOLAR),
        "tags": np.array(tags),
        "T_CONST": np.array(T_CONST),
        "logC_bin_100": np.array(logC_bin_100),
        "logC_bin_300": np.array(logC_bin_300),
        "logC_sin_100": np.array(logC_sin_100),
        "logC_sin_300": np.array(logC_sin_300),
    }


def load_or_compute_logC():
    """Load cached C(Z*) curves or compute from BPASS spectra and save."""
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        if float(data["T_CONST"]) == T_CONST:
            return {k: data[k] for k in data.files}

    print(f"Computing C(Z*) curves (cache miss: {CACHE_PATH.name}) ...")
    arrays = _compute_logC_arrays()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, **arrays)
    print(f"Cache saved: {CACHE_PATH}")
    return arrays


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Paper Table 2 reference values
    paper_vals = {
        "z001": 41.680, "z002": 41.647, "z003": 41.619, "z004": 41.595,
        "z006": 41.544, "z008": 41.512, "z010": 41.473, "z014": 41.411,
        "z020": 41.373,
    }

    cached = load_or_compute_logC()
    tags = cached["tags"]
    Z_arr = cached["Z"]
    log_Z_Zsol = cached["log_Z_Zsol"]
    logC_bin_100 = cached["logC_bin_100"]
    logC_bin_300 = cached["logC_bin_300"]
    logC_sin_100 = cached["logC_sin_100"]
    logC_sin_300 = cached["logC_sin_300"]

    print(f"{'tag':>5}  {'Z*':>9}  {'log Z/Zsol':>11}  {'log C(Z*)':>10}  "
          f"{'Paper':>7}  {'Diff':>7}")
    print("-" * 60)
    for i, tag in enumerate(tags):
        Z = Z_arr[i]
        log_C = logC_bin_100[i]
        paper = paper_vals.get(str(tag))
        paper_s = f"{paper:7.3f}" if paper is not None else "    -- "
        diff_s  = f"{log_C - paper:+7.3f}" if paper is not None else "    -- "
        print(f"{tag:>5}  {Z:9.5f}  {log_Z_Zsol[i]:11.3f}  "
              f"{log_C:10.3f}  {paper_s}  {diff_s}")

    # -----------------------------------------------------------------------
    # C(Z*) calibration plot: bin/sin x chab100/chab300, all at 100 Myr
    # -----------------------------------------------------------------------

    apply_paper_style(usetex=False)
    fig, axes = make_subplots(
        ncol=1, nrow=1, plot_size=2.25, return_fig=True,
        row_spacing=[MARGIN_LABEL - 0.35, MARGIN_PAD],
        col_spacing=[MARGIN_LABEL - 0.3, MARGIN_PAD],
    )
    ax = axes[0]

    ax.plot(log_Z_Zsol, logC_bin_100, color=COLOR_BINARY, ls="-", lw=3.0)
    ax.plot(log_Z_Zsol, logC_bin_300, color=COLOR_BINARY, ls="-", lw=1.0)
    ax.plot(log_Z_Zsol, logC_sin_100, color=COLOR_SINGLE, ls="--", lw=3.0, alpha=0.6)
    ax.plot(log_Z_Zsol, logC_sin_300, color=COLOR_SINGLE, ls="--", lw=1.0, alpha=0.6)

    ax.axhline(41.30, color="k", ls=":", lw=1, alpha=0.5)
    ax.text(-2.175, 41.305, "KE12", color="k", alpha=0.5, fontsize=9,
            va="bottom")

    ax.set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
    ax.set_ylabel(r"$\log_{10}\,C(Z_*)$ [erg/s]")
    ax.set_xlim(-2.35, 0)
    ax.set_ylim(41.25, 42)

    for i in range(10):
        ax.fill_betweenx(y=[41.25, 41.5 + i*0.05], x1=-1.69, x2=-0.715, facecolor="k", edgecolor="none", alpha=0.01)
    

    ax.text( -1.69 - 0.135 , 41.35,r"$M_{\star}\!=\!10^6 M_{\odot}$", color="k",rotation=90,fontsize = 8)
    ax.text( -0.715 - 0.135 , 41.35,r"$10^{9.25} M_{\odot}$", color="k",rotation=90,fontsize = 8)


    legend_handles = [
        Line2D([0], [0], color=COLOR_BINARY, ls="-", lw=1.5, label="Binary"),
        Line2D([0], [0], color=COLOR_SINGLE, ls="--", lw=1.5, alpha=0.6,
               label="Single"),
        Line2D([0], [0], color="k", ls="-", lw=3.0,
               label=r"$M_\mathrm{max}=100\,M_\odot$"),
        Line2D([0], [0], color="k", ls="-", lw=1.0,
               label=r"$M_\mathrm{max}=300\,M_\odot$"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", 
    handlelength=1, handletextpad=0.5, fontsize=10, ncol=2, columnspacing=0.75)

    outpath = BPASS_ROOT / "figures" / "bpass_C_Zstar_calibration.pdf"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    print(f"\nFigure saved: {outpath}")
