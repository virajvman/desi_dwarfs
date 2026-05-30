"""
Deterministic tests of the Halpha -> SFR calibration bias, using the BPASS
binary kernel. Unlike the PSD approach, these require NO assumption about the
short-timescale SFR power spectrum -- they just ask "what does L(Halpha)/C(Z*)
report under a specified recent star-formation history?"

Two tests, each a one-parameter family:

  TEST 1 - TRUNCATION (post-burst / fading):
    Constant SFR = 1 until N Myr ago, then zero. Sweep N.
    Asks: if a dwarf stopped forming stars N Myr ago, how badly does Halpha
    under-report its true recent (~5 Myr) average SFR?

  TEST 2 - ONSET (rising / just-ignited):
    SFR = 0, then suddenly = 1 for the last D Myr (still ongoing). Sweep D.
    Asks: if a burst started D Myr ago, how does L(Halpha) build toward
    equilibrium, and how does the inferred SFR compare to the recent average?

Both reference an SFR of 1 Msun/yr, so all SFRs are in those units and
SFR_app = L(Halpha)/C(Z*) is directly interpretable. The comparison is to
the true 5 Myr-averaged SFR (Halpha is a ~10 Myr tracer; the 100 Myr
comparison is intentionally NOT included here).

Kernel intuition (K = the Halpha burst kernel, cum = its cumulative fraction):
    truncation:  SFR_app(N) ~ 1 - cum(N)     (only stars older than N remain)
    onset:       SFR_app(D) ~ cum(D)          (only stars younger than D exist)
so these curves are essentially the cumulative Halpha kernel read forwards
(onset) and backwards (truncation).
"""
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot_style import (
    apply_paper_style, make_subplots, reshape_axes,
    MARGIN_LABEL, MARGIN_PAD, MARGIN_SHARED, MARGIN_SPLIT,
)

# numpy 2.0 removed np.trapz (renamed to np.trapezoid); keep both working.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# ---------------------------------------------------------------------------
H_PLANCK = 6.62607015e-27
C_LIGHT  = 2.99792458e10
LSUN     = 3.828e33
HC_ERG_A = H_PLANCK * C_LIGHT * 1e8
L_HALPHA_PER_N_ION = 1.36e-12
LAMBDA_MAX = 912.0

LOG_AGES = np.arange(6.0, 11.01, 0.1)
N_AGES   = len(LOG_AGES)
_lo = 10.0 ** (LOG_AGES - 0.05); _lo[0] = 0.0
_hi = 10.0 ** (LOG_AGES + 0.05)

BPASS_DIR  = Path("/Users/virajmanwadkar/Desktop/Stanford/Research/fun_research/bpass/BPASS_to_Viraj/bpass_v2.2.1_imf_chab100")
BURST_MASS = 1e6
TAG, LOG_C_CONST = "z008", 41.512
C_CONST = 10.0 ** LOG_C_CONST

# Halpha contribution only from the past 100 Myr (matches how C(Z*) is defined:
# the ionizing SED has faded to negligible levels by 100 Myr even for binaries).
T_HALPHA_MAX = 100e6
HALPHA_BINS  = LOG_AGES <= np.log10(T_HALPHA_MAX)
SFR_AVG_WINDOW = 5e6  # yr; true <SFR> comparison window


def load_bpass_bin(tag):
    data = np.loadtxt(BPASS_DIR / f"spectra-bin-imf_chab100.{tag}.dat.gz")
    return data[:, 0], data[:, 1:]


def composite_sed(wl, sed, sfh, n_sub=25):
    """Composite SED [erg/s/A] for SFH sfh(t_lookback) [yr], summing only the
    past 100 Myr. = sum over age bins of (mass formed / 10^6 Msun) * SSP SED."""
    masses = np.zeros(N_AGES)
    for n in np.flatnonzero(HALPHA_BINS):
        ts = np.linspace(_lo[n], _hi[n], n_sub)
        masses[n] = np.trapz(sfh(ts), ts)
    return (sed @ (masses / BURST_MASS)) * LSUN       # erg/s/A


def L_halpha_from_sed(wl, f_lambda):
    """L(Halpha) [erg/s] from a composite SED via the ionizing photon rate."""
    ion = wl < LAMBDA_MAX
    N_ion = np.trapz(f_lambda[ion] * wl[ion], wl[ion]) / HC_ERG_A
    return L_HALPHA_PER_N_ION * N_ion


def L_halpha(wl, sed, sfh, n_sub=25):
    """Convenience wrapper: L(Halpha) [erg/s] for an SFH (builds the SED)."""
    return L_halpha_from_sed(wl, composite_sed(wl, sed, sfh, n_sub))


def mean_sfr_window(sfh, t_window, n=400):
    ts = np.linspace(0, t_window, n)
    return np.trapz(sfh(ts), ts) / t_window


# ---------------------------------------------------------------------------
# Deterministic SFHs (t = lookback time in years, t=0 is the observation epoch)
# ---------------------------------------------------------------------------
def sfh_truncation(N_yr):
    """Constant SFR=1 until N_yr ago, then 0. (SFR present for lookback >= N.)"""
    return lambda t: np.where(t >= N_yr, 1.0, 0.0)


def sfh_onset(D_yr):
    """SFR=1 for the last D_yr (lookback <= D), 0 before."""
    return lambda t: np.where(t <= D_yr, 1.0, 0.0)

from matplotlib.offsetbox import HPacker, TextArea, AnchoredOffsetbox

def colored_xlabel(ax, parts, fontsize=12, y=-0.125):
    """parts: list of (text, color). Replaces the x-axis label with
    natively-colored fragments that work on any backend."""
    ax.set_xlabel("")  # clear the normal label
    boxes = [TextArea(t, textprops=dict(color=c, fontsize=fontsize))
             for t, c in parts]
    packed = HPacker(children=boxes, align="baseline", pad=0, sep=0)
    box = AnchoredOffsetbox(loc="upper center", child=packed, pad=0,
                            frameon=False, borderpad=0,
                            bbox_to_anchor=(0.5, y),
                            bbox_transform=ax.transAxes)
    ax.add_artist(box)


# ===========================================================================
if __name__ == "__main__":
    wl, sed = load_bpass_bin(TAG)

    # ---- TEST 1: truncation sweep ----
    N_dense  = np.linspace(0, 50e6, 60)          # time since SF ceased [yr]
    N_marks  = np.array([0, 3, 5, 10, 20, 50]) * 1e6
    trunc = {"N": N_dense, "app": [], "s5": []}
    for N in N_dense:
        sfh = sfh_truncation(N)
        trunc["app"].append(L_halpha(wl, sed, sfh) / C_CONST)
        trunc["s5"].append(mean_sfr_window(sfh, SFR_AVG_WINDOW))
    for k in ("app", "s5"):
        trunc[k] = np.array(trunc[k])

    # ---- TEST 2: onset sweep ----
    D_dense  = np.linspace(0.5e6, 30e6, 60)      # time since SF began [yr]
    D_marks  = np.array([1, 3, 5, 10, 20]) * 1e6
    onset = {"D": D_dense, "app": [], "s5": []}
    for D in D_dense:
        sfh = sfh_onset(D)
        onset["app"].append(L_halpha(wl, sed, sfh) / C_CONST)
        onset["s5"].append(mean_sfr_window(sfh, SFR_AVG_WINDOW))
    for k in ("app", "s5"):
        onset[k] = np.array(onset[k])

    # ---- print tables at the marked values ----
    def row(sfh):
        a   = L_halpha(wl, sed, sfh) / C_CONST
        s5 = mean_sfr_window(sfh, SFR_AVG_WINDOW)
        d5 = np.log10(a/s5) if s5 > 1e-6 else np.nan
        return a, s5, d5

    print("TRUNCATION (constant until N Myr ago, then 0):")
    print(f"{'N[Myr]':>7} {'SFR_app':>9} {'<SFR>5':>9} {'app/5[dex]':>12}")
    for N in N_marks:
        a, s5, d5 = row(sfh_truncation(N))
        print(f"{N/1e6:7.0f} {a:9.3f} {s5:9.3f} {d5:12.2f}")
    print("\nONSET (0, then on for last D Myr):")
    print(f"{'D[Myr]':>7} {'SFR_app':>9} {'<SFR>5':>9} {'app/5[dex]':>12}")
    for D in D_marks:
        a, s5, d5 = row(sfh_onset(D))
        print(f"{D/1e6:7.0f} {a:9.3f} {s5:9.3f} {d5:12.2f}")

    # -----------------------------------------------------------------------
    # Plot: 2 rows x 2 cols.
    #   row 0: SFR curves (SFR_app vs true <SFR>_5Myr)
    #   row 1: bias in dex (SFR_app / <SFR>_5Myr)
    # NOTE: each x value is a DIFFERENT SFH (N or D), not a time axis -- these
    #       are response curves for the truncation / onset families.
    # -----------------------------------------------------------------------

    color_onset = "mediumblue"
    color_truncation = "firebrick"

    apply_paper_style(usetex=False)
    fig, axes = make_subplots(
        ncol=1, nrow=2, plot_size=2.25, return_fig=True,
        row_spacing=[MARGIN_LABEL-0.25, MARGIN_SHARED, MARGIN_PAD],
        col_spacing=[MARGIN_LABEL-0.25, MARGIN_PAD],
    )
    axes[1].sharex(axes[0])

    axes[1].tick_params(labelbottom=False)

    from matplotlib.lines import Line2D

    # (a) truncation, SFR curves  — note: labels removed from the plot calls
    axes[1].plot(N_dense/1e6, trunc["app"], color=color_truncation, ls="--", alpha=0.5, lw=1.5)
    axes[1].plot(N_dense/1e6, trunc["s5"], color=color_truncation, ls='-',  lw=2)

    # (b) onset, SFR curves
    axes[1].plot(D_dense/1e6, onset["app"], color=color_onset, ls="--", alpha=0.5, lw=1.5)
    axes[1].plot(D_dense/1e6, onset["s5"], color=color_onset, ls='-',  lw=2)

    # axes[1].plot(D_marks/1e6, [row(sfh_onset(D))[0] for D in D_marks], 'C3o', ms=5)
    axes[1].axhline(1, color='k', ls=':', alpha=0.5,lw=1)
    axes[1].text(2.725+2.5,1.05,"Constant SFR",color="k",alpha=0.5)

    axes[1].annotate(
    "",                              # no text, arrow only
    xy=(2.0+2.5, 0.99),                   # arrowhead: where it points (on the line at y=1)
    xytext=(2.7+2.5, 1.075),              # tail: left side of the text
    arrowprops=dict(
        arrowstyle="->,head_length=0.25,head_width=0.12",
        color="k", alpha=0.5, lw=0.5,
        connectionstyle="arc3,rad=0.25",
    ),)

    axes[1].set_ylabel(r"SFR  [units of steady rate]")

    # --- black proxy handles for the legend ---
    legend_handles = [
        Line2D([0], [0], color='k', ls='--', lw=1.5, alpha=0.5,
            label=r"SFR$_\mathrm{app}=L(\mathrm{H}\alpha)/C$"),
        Line2D([0], [0], color='k', ls='-', lw=2,
            label=r"true $\langle$SFR$\rangle_{5\,\mathrm{Myr}}$"),
    ]
    axes[1].legend(handles=legend_handles, loc='best', handlelength=1, handletextpad=0.5)
    axes[1].set_ylim(-0.05, 1.15)

    # (c) truncation, bias in dex
    m5 = trunc["s5"] > 1e-6
    axes[0].plot(N_dense[m5]/1e6, np.log10(trunc["app"][m5]/trunc["s5"][m5]),ls="-", lw=2.0, color=color_truncation)
    axes[0].axhline(0, color='k',ls=':',lw=1,alpha = 0.5)
    axes[0].set_ylabel(r"$\log_{10}\,$SFR$_\mathrm{app}/\langle$SFR$\rangle_{5\,\mathrm{Myr}}$")

    # (d) onset, bias in dex
    axes[0].plot(D_dense/1e6, np.log10(onset["app"]/onset["s5"]),ls="-", lw=2.0, color=color_onset)

    # axes[0].set_xlabel(r"Time since SF \textcolor{red}{ceased}/\textcolor{blue}{began} [Myr]")

    colored_xlabel(axes[0], [
    (r"Time since SF$\,$", "black"),
    (r"$\mathbf{ceased}$",         "firebrick"),
    ("/",              "black"),
    (r"$\mathbf{began}$",          "mediumblue"),
    (" [Myr]",         "black"),
])

    axes[0].set_ylim(-0.6, 0.6)
    axes[0].set_xlim([0,30])

    from matplotlib.patches import ConnectionPatch

    # ---- pick the target points on each curve (data coords) ----
    # blue / onset bias curve (the positive bump, above the zero line)
    x_blue = D_dense / 1e6
    y_blue = np.log10(onset["app"] / onset["s5"])
    i_blue = np.argmin(np.abs(x_blue - 5.0))        # ~2 Myr; change the time to move it
    pt_blue = (x_blue[i_blue], y_blue[i_blue])

    # red / truncation bias curve (the trough, below the zero line) — same mask you plotted
    x_red = N_dense[m5] / 1e6
    y_red = np.log10(trunc["app"][m5] / trunc["s5"][m5])
    i_red = np.argmin(np.abs(x_red - 5.0))          # ~6 Myr; change the time to move it
    pt_red = (x_red[i_red], y_red[i_red])

    # ---- rectangular inset axes, bounds = [x0, y0, width, height] in axes fractions ----
    inset_top = axes[0].inset_axes([0.55, 0.6, 0.38, 0.3])   # above the zero line
    inset_bot = axes[0].inset_axes([0.55, 0.1, 0.38, 0.3])   # below the zero line

    tgrid = np.arange(0,100,0.1)
    sfr_constant_grid = np.zeros(len(tgrid)) + 1

    sfr_constant_grid[tgrid < 5] = 0

    inset_bot.plot(tgrid, sfr_constant_grid, color=color_truncation, lw=1.5)
    inset_bot.axhline(1, color='k', ls=':', lw=0.5, alpha = 0.5)
    inset_bot.set_ylabel("SFR", fontsize=7, labelpad=1)
    inset_bot.set_xlabel(r"$t$", fontsize=7, labelpad=2)
    inset_bot.set_xlim([0,30])
    inset_bot.set_ylim([-0.025,1.025])

    inset_bot.set_xticks([])
    inset_bot.set_yticks([])

    ########################################################

    tgrid = np.arange(0,100,0.1)
    sfr_constant_grid = np.zeros(len(tgrid)) + 1

    sfr_constant_grid[tgrid > 5] = 0

    inset_top.plot(tgrid, sfr_constant_grid, color=color_onset, lw=1.5)
    inset_top.axhline(1, color='k', ls=':', lw=0.5, alpha = 0.5)
    inset_top.set_ylabel("SFR", fontsize=7, labelpad=1)
    inset_top.set_xlabel(r"$t$", fontsize=7, labelpad=2)
    inset_top.set_xlim([0,30])
    inset_top.set_ylim([-0.025,1.025])

    inset_top.set_xticks([])
    inset_top.set_yticks([])


    for ax_in in (inset_top, inset_bot):
        ax_in.tick_params(labelsize=7)
        # leave empty — you'll plot your own content here

    # optional: mark the points being pointed at
    axes[0].plot(*pt_blue, 'o', ms=3, color=color_onset,      zorder=6)
    axes[0].plot(*pt_red,  'o', ms=3, color=color_truncation, zorder=6)

   # two leaders from the top inset's left edge -> same point on the blue curve
    for corner in [(0.0, 1.0), (0.0, 0.0)]:          # top-left, bottom-left
        con = ConnectionPatch(
            xyA=pt_blue, coordsA=axes[0].transData,
            xyB=corner,  coordsB=inset_top.transAxes,
            arrowstyle="-", color="k", lw=1.0, alpha=0.25, zorder=5,
        )
        con.set_clip_on(False)
        axes[0].add_artist(con)


    for corner in [(0.0, 1.0), (0.0, 0.0)]:          # top-left, bottom-left
        con = ConnectionPatch(
            xyA=pt_red, coordsA=axes[0].transData,
            xyB=corner,  coordsB=inset_bot.transAxes,
            arrowstyle="-", color="k", lw=1.0, alpha=0.25, zorder=5,
        )
        con.set_clip_on(False)
        axes[0].add_artist(con)

    outpath = "/Users/virajmanwadkar/Desktop/Stanford/Research/fun_research/bpass/figures/deterministic_sfh_tests.pdf"
    fig.savefig(outpath)
    print(f"\nFigure saved: {outpath}")