"""
appendix_nebular_qa_plots.py
============================

Two catalog-paper appendix figures for "Details on Nebular Property
Estimation" (QA of the direct-Te pipeline against the population-level
prescriptions):

  1. direct_vs_strongline_zgas.pdf
     Top   : direct-Te 12+log(O/H) (TE_12_LOG_OH) vs strong-line R23+N2
             (Z_GAS_R23_N2) as a 2D density, with the 1:1 line.
             Both panels additionally require [OIII]4363 SNR > 5 (from the
             FASTSPEC HDU): threshold-level auroral detections Eddington-bias
             the direct Te high (O/H low), producing a spurious clump at low
             direct / high strong-line O/H.
     Bottom: our direct-Te vs literature direct-Te for objects cross-matched
             against three external samples -- Sui et al. 2026 DESI XMPGs
             (TARGETID join on data/table1.fits), Izotov et al. 2006 SDSS
             metal-poor galaxies (data/sdss_xmpg/, 1.5" sky match), and
             CLASSY (Berg et al. 2022; data/classy/, 1.5" sky match with a
             |dz| < 0.005 veto). Different symbols per sample; per-sample
             match counts and offset stats are printed to stdout.

  2. av_direct_vs_param.pdf
     Left : per-object direct-fit A_V (TE_AV, joint {ne, Te, Av, O+, O++}
            posterior median) vs log M*, as a 2D density, with the
            mass-based parametrization overlaid: BD = model_hahb(logM*)
            (logistic fit to stacked Halpha/Hbeta, 2.86 low-mass asymptote)
            dereddened against the intrinsic 2.79 through the CCM89 law.
            A vertical bar shows the median TE_AV_ERR of the sample.
     Right: residual TE_AV - A_V,model(logM*) vs log M*, with white running
            medians (16-84th percentile bars).

Both read only catalog columns (MAIN + SPEC_DERIVED, plus the two [OIII]4363
columns of FASTSPEC for the SNR cut), so they run anywhere astropy, matplotlib,
scipy, and cmasher are available (e.g. locally on a downloaded catalog). The
shared purple colormap, density-contour (plot_2d_dist) and alternating-line
helpers come from plot_style.py (not desi_lowz_funcs, which needs desispec).
The literature files for the cross-match panel are resolved relative to the
repo (data/); a missing file just skips that sample with a warning.
Sample: DWARF_MASKBIT == 0 plus finite values of the plotted columns; the
direct-Te subset is auroral-line selected (see the paper appendix caveat).

Usage
-----
    python appendix_nebular_qa_plots.py CATALOG.fits --outdir /path/to/figures
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import cmasher as cmr

from astropy.table import Table, join, unique
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
import astropy.units as u

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
for _p in (_CODE_DIR, os.path.join(_CODE_DIR, "nebular_stuff")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_style import (
    apply_paper_style, make_subplots,
    make_alternating_plot, plot_2d_dist,
    MARGIN_LABEL, MARGIN_PAD, MARGIN_SPLIT, MARGIN_SHARED
)

from cardelli_attenuation import k_ccm89, model_hahb, BALMER_INTRINSIC


# Purple sequential colormap shared with the other catalog-paper hist2d panels
# (explore_fastspec): cmr.gothic_r with the near-white low end clipped, and
# empty (zero-count) bins rendered transparent.
def _purple_hist_cmap():
    cmap = mcolors.ListedColormap(cmr.gothic_r(np.linspace(0.05, 1.0, 256)))
    cmap.set_bad(alpha=0)
    return cmap

PURPLE_CMAP = _purple_hist_cmap()


_DATA_DIR = os.path.join(os.path.dirname(_CODE_DIR), "data")

# Literature direct-Te cross-match settings
MATCH_RADIUS_ARCSEC = 1.5
DZ_MAX = 0.005   # |z_DESI - z_lit| veto for sky matches (where lit z exists)

# Minimum [OIII]4363 SNR for the direct-Te metallicity comparison figure.
# Threshold-level (SNR ~ 3) auroral detections are Eddington-biased: upscattered
# 4363 -> overestimated Te -> underestimated direct O/H.
AURORAL_SNR_MIN = 5.0


def read_4363_snr(cat_path):
    """[OIII]4363 SNR (flux * sqrt(ivar)) from the FASTSPEC HDU, row-matched
    to MAIN. Memory-mapped so only the two needed columns are actually read."""
    with fits.open(cat_path, memmap=True) as hdul:
        data = hdul["FASTSPEC"].data
        flux = np.asarray(data["OIII_4363_FLUX"], dtype=float)
        ivar = np.asarray(data["OIII_4363_FLUX_IVAR"], dtype=float)
    return flux * np.sqrt(np.clip(ivar, 0, np.inf))


def _load_sui26(data_dir):
    """Sui et al. 2026 DESI XMPGs (data/table1.fits): matched by TARGETID."""
    path = os.path.join(data_dir, "table1.fits")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found; skipping Sui+26")
        return None
    t = Table.read(path)
    return dict(name="Sui+26", targetid=np.asarray(t["TARGETID"]),
                oh=np.asarray(t["O_ABUNDANCE"], float),
                oh_err=np.asarray(t["O_ABUNDANCEERR"], float))


def _load_izotov06(data_dir):
    """Izotov et al. 2006 SDSS-DR3 metal-poor galaxies (data/sdss_xmpg/):
    positions from sdss.dat joined to abundances in table2.dat on Name."""
    d = os.path.join(data_dir, "sdss_xmpg")
    files = [os.path.join(d, f) for f in ("sdss.dat", "table2.dat", "ReadMe")]
    if not all(os.path.exists(f) for f in files):
        print(f"  WARNING: files missing in {d}; skipping Izotov+06")
        return None
    readme = files[2]
    pos = unique(ascii.read(files[0], readme=readme, format="cds"), keys="Name")
    ab = unique(ascii.read(files[1], readme=readme, format="cds"), keys="Name")
    t = join(pos, ab, keys="Name")
    ra = np.asarray(t["RAdeg"].filled(np.nan) if hasattr(t["RAdeg"], "filled")
                    else t["RAdeg"], float)
    dec = np.asarray(t["DEdeg"].filled(np.nan) if hasattr(t["DEdeg"], "filled")
                     else t["DEdeg"], float)
    good = np.isfinite(ra) & np.isfinite(dec)   # one object has no position
    t = t[good]
    coords = SkyCoord(ra[good] * u.deg, dec[good] * u.deg)
    return dict(name="Izotov+06", coords=coords, z=None,
                oh=np.asarray(t["12+logO/H"], float),
                oh_err=np.asarray(t["e_12+logO/H"], float))


def _load_classy(data_dir):
    """CLASSY (Berg et al. 2022, data/classy/table5.dat): direct-Te O/H."""
    d = os.path.join(data_dir, "classy")
    files = [os.path.join(d, f) for f in ("table5.dat", "ReadMe")]
    if not all(os.path.exists(f) for f in files):
        print(f"  WARNING: files missing in {d}; skipping CLASSY")
        return None
    t = ascii.read(files[0], readme=files[1], format="cds")
    ra = (np.asarray(t["RAh"], float) + np.asarray(t["RAm"], float) / 60.0
          + np.asarray(t["RAs"], float) / 3600.0) * 15.0
    sign = np.where(np.asarray(t["DE-"]) == "-", -1.0, 1.0)
    dec = sign * (np.asarray(t["DEd"], float) + np.asarray(t["DEm"], float) / 60.0
                  + np.asarray(t["DEs"], float) / 3600.0)
    return dict(name="CLASSY", coords=SkyCoord(ra * u.deg, dec * u.deg),
                z=np.asarray(t["z"], float),
                oh=np.asarray(t["Z"], float),
                oh_err=np.asarray(t["e_Z"], float))


def add_literature_comparison(ax, main, der, snr4363):
    """Cross-match the direct-Te sample against literature direct-Te catalogs
    and overplot the matches on `ax` (our value on x, literature on y).

    Match pool: DWARF_MASKBIT == 0 & finite TE_12_LOG_OH & [OIII]4363
    SNR > AURORAL_SNR_MIN. Sui+26 is joined on TARGETID; Izotov+06 and CLASSY
    are matched lit -> nearest pool object within MATCH_RADIUS_ARCSEC (each
    literature object appears at most once), with a |dz| < DZ_MAX veto where
    the literature provides a redshift.
    """
    pool = ((np.asarray(main["DWARF_MASKBIT"]) == 0)
            & np.isfinite(np.asarray(der["TE_12_LOG_OH"]))
            & (snr4363 > AURORAL_SNR_MIN))
    pm, pd = main[pool], der[pool]
    pool_oh = np.asarray(pd["TE_12_LOG_OH"], float)
    pool_oh_err = np.asarray(pd["TE_12_LOG_OH_ERR"], float)
    pool_coords = SkyCoord(np.asarray(pm["RA_TARGET"], float) * u.deg,
                           np.asarray(pm["DEC_TARGET"], float) * u.deg)

    styles = {
        "Sui+26":    dict(marker="o", color="orchid"),
        "Izotov+06": dict(marker="s", color="dodgerblue"),
        "CLASSY":    dict(marker="D", color="goldenrod"),
    }

    samples = [_load_sui26(_DATA_DIR), _load_izotov06(_DATA_DIR),
               _load_classy(_DATA_DIR)]
    for lit in samples:
        if lit is None:
            continue
        if "targetid" in lit:
            pool_tg = np.asarray(pm["TARGETID"])
            order = np.argsort(pool_tg)
            pos = np.searchsorted(pool_tg, lit["targetid"], sorter=order)
            pos = np.clip(pos, 0, len(pool_tg) - 1)
            sel = pool_tg[order[pos]] == lit["targetid"]
            idx = order[pos[sel]]
        else:
            near, sep, _ = lit["coords"].match_to_catalog_sky(pool_coords)
            sel = sep.arcsec < MATCH_RADIUS_ARCSEC
            if lit["z"] is not None:
                dz_ok = (np.abs(lit["z"] - np.asarray(pm["Z"], float)[near])
                         < DZ_MAX)
                n_veto = int((sel & ~dz_ok).sum())
                if n_veto:
                    print(f"  {lit['name']}: {n_veto} match(es) rejected by "
                          f"|dz| > {DZ_MAX}")
                sel &= dz_ok
            idx = near[sel]

        x, xerr = pool_oh[idx], pool_oh_err[idx]
        y, yerr = lit["oh"][sel], lit["oh_err"][sel]
        resid = x - y
        print(f"  lit-compare {lit['name']}: N={len(x)}, "
              f"median offset (direct - lit)={np.median(resid):+.3f} dex, "
              f"NMAD={nmad(resid):.3f} dex")

        st = styles[lit["name"]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=st["marker"],
                    mfc="none", mec=st["color"], mew=0.9, ms=4.5,
                    ecolor="0.6", elinewidth=0.6, capsize=0, alpha=0.85,
                    ls="none", zorder=5,
                    label=f"{lit['name']} (N={len(x)})")

    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.5,
              fontsize=8, frameon=False)


def nmad(x):
    """Normalised median absolute deviation (robust sigma)."""
    x = np.asarray(x, dtype=float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def av_from_bd(bd):
    """A_V implied by an observed Balmer decrement through the pipeline
    convention: CCM89 (R_V=3.1), dereddened against the intrinsic 2.79.
    E(B-V) = 2.5 log10(BD/2.79) / (k_Hb - k_Ha);  A_V = E(B-V) * k(5500)."""
    ebv = 2.5 * np.log10(np.asarray(bd, dtype=float) / BALMER_INTRINSIC) \
        / (k_ccm89(4861.0) - k_ccm89(6563.0))
    return np.clip(ebv, 0, np.inf) * k_ccm89(5500.0)


def running_median(x, y, edges, min_n=20):
    """Median + 16/84th percentiles of y in bins of x (skip bins with < min_n)."""
    cen, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x < b)
        if sel.sum() < min_n:
            continue
        cen.append(0.5 * (a + b))
        med.append(np.median(y[sel]))
        lo.append(np.percentile(y[sel], 16))
        hi.append(np.percentile(y[sel], 84))
    cen, med, lo, hi = map(np.asarray, (cen, med, lo, hi))
    return cen, med, med - lo, hi - med


def add_median_points(ax, x, y, edges, min_n=20):
    cen, med, elo, ehi = running_median(x, y, edges, min_n=min_n)
    ax.errorbar(cen, med, yerr=[elo, ehi], fmt="o", color="w",
                mec="k", mew=0.8, ecolor="k", elinewidth=1.0,
                capsize=0, ms=5, zorder=5)


def two_panel_axes():
    """Standard two-panel layout used by the other comparison figures."""
    fig, axes = make_subplots(
        ncol=2, nrow=1, plot_size=2.25, return_fig=True,
        row_spacing=[MARGIN_LABEL - 0.4, MARGIN_PAD],
        col_spacing=[MARGIN_LABEL - 0.2, MARGIN_SPLIT, MARGIN_PAD],
    )
    return fig, axes


def plot_direct_vs_strongline(main, der, snr4363, outpath):
    base = (
        (np.asarray(main["DWARF_MASKBIT"]) == 0)
        & np.isfinite(np.asarray(der["TE_12_LOG_OH"]))
        & np.isfinite(np.asarray(der["Z_GAS_R23_N2"]))
    )
    ok = base & (snr4363 > AURORAL_SNR_MIN)
    print(f"  [OIII]4363 SNR > {AURORAL_SNR_MIN:g} cut: "
          f"{base.sum()} -> {ok.sum()} objects")


    strong = np.asarray(der["Z_GAS_R23_N2"])[ok]
    direct = np.asarray(der["TE_12_LOG_OH"])[ok]
    resid = direct - strong

    med_off, sig = np.median(resid), nmad(resid)
    print(f"  direct-vs-strongline: N={ok.sum()}, "
          f"median offset={med_off:+.3f} dex, NMAD={sig:.3f} dex")

    fig, ax = make_subplots(
        ncol=1, nrow=2, plot_size=2.25, return_fig=True,
        row_spacing=[MARGIN_LABEL - 0.4, MARGIN_SHARED+0.1, MARGIN_PAD],
        col_spacing=[MARGIN_LABEL - 0.2, MARGIN_PAD],
    )

    lim = (7.0, 8.75)
    ax[1].hist2d(direct, strong, bins=[30, 30],
               range=[lim, lim], norm=LogNorm(), cmap=PURPLE_CMAP, rasterized=True)

    ax[1].set_ylabel(r"$12 + \log_{10}(\mathrm{O/H})$  [$R_{23}$+$N2$]")
    ax[1].set_xticklabels([])

    ax[0].set_ylabel(r"$12 + \log_{10}(\mathrm{O/H})$  [direct $T_e$, lit.]")
    ax[0].set_xlabel(r"$12 + \log_{10}(\mathrm{O/H})$  [direct $T_e$]")

    add_literature_comparison(ax[0], main, der, snr4363)

    # alternating yellowgreen/black 1:1 line (shared convention with the other
    # catalog-paper comparison figures), on both the density and lit. panels
    line_grid = np.linspace(lim[0], lim[1], 50)
    for axi in ax:
        axi.set_yticks([7,7.5,8,8.5,9])
        axi.set_xticks([7,7.5,8,8.5,9])
        axi.set_xlim(lim); axi.set_ylim(lim)
        make_alternating_plot(axi, line_grid, line_grid, dash_len=1,
                              color_1="yellowgreen", color_2="k", lw=1)
        axi.collections[-1].set_zorder(4)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  saved {outpath}")

def plot_av_direct_vs_param(main, der, outpath):
    ok = (
        (np.asarray(main["DWARF_MASKBIT"]) == 0)
        & np.isfinite(np.asarray(der["TE_AV"]))
        & (np.asarray(der["TE_AV_ERR"]) < 0.2)
        & (der["TE_FIT_SUCCESS"] == True )
        & np.isfinite(np.asarray(main["LOG_MSTAR_M24"]))
        & (np.asarray(der["TE_CHI2_AV_ML"] < 5))
    )
    logm = np.asarray(main["LOG_MSTAR_M24"])[ok]
    av = np.asarray(der["TE_AV"])[ok]
    av_err = np.asarray(der["TE_AV_ERR"])[ok]
    av_model = av_from_bd(model_hahb(logm))
    resid = av - av_model
    med_err = np.nanmedian(av_err)

    #print the median te_chi2_av
    print("Median AV CHI2 = ", np.median( np.asarray(der["TE_CHI2_AV_ML"])[ok] )  )
    print("16per AV CHI2 = ", np.percentile( np.asarray(der["TE_CHI2_AV_ML"])[ok], 16 )  )
    print("84per AV CHI2 = ", np.percentile( np.asarray(der["TE_CHI2_AV_ML"])[ok], 84 )  )

    print(f"  av-direct-vs-param: N={ok.sum()}, median TE_AV_ERR={med_err:.3f}, "
          f"median resid={np.median(resid):+.3f}, NMAD={nmad(resid):.3f}")

    fig, ax = make_subplots(
        ncol=1, nrow=1, plot_size=2.25, return_fig=True,
        row_spacing=[MARGIN_LABEL - 0.4, MARGIN_PAD],
        col_spacing=[MARGIN_LABEL - 0.2, MARGIN_PAD],
    )

    mlim = (6.5, 9.25)
    alim = (0.0, 1.6)
    ax[0].hist2d(logm, av, bins=[30, 30],
               range=[mlim, alim], norm=LogNorm(), cmap=PURPLE_CMAP, rasterized=True)
    # 1- and 2-sigma density contours (matches the M*-SFR/sSFR panel in
    # explore_fastspec, which also uses plot_2d_dist)
    plot_2d_dist(logm, av, 50, 50,
                 cmin=1.e-4, cmax=1.0, smooth=10, clevs=[0, 0.68, 0.95], ax=ax[0],
                 bounds=np.array([mlim[0], mlim[1], alim[0], alim[1]]),
                 cmap=None, color="k", filled=False, label=None,
                 cmap_alpha=1, lw_scale=0.75, alternating_contours=False)
    mgrid = np.linspace(*mlim, 200)
    ax[0].plot(mgrid, av_from_bd(model_hahb(mgrid)), color="darkorange", lw=2.0, ls="-",
             zorder=6, label=r"mass-based model")
    # median per-object A_V uncertainty, shown as a single vertical bar
    ax[0].errorbar([mlim[0] + 0.25], [alim[1] - 0.35], yerr=[med_err],
                 fmt="none", ecolor="k", elinewidth=1.2, capsize=2.5, zorder=5)
    ax[0].text(mlim[0] + 0.33, alim[1] - 0.35, "median\nerror",
             fontsize=8, ha="left", va="center")
    ax[0].legend(loc="upper right", handlelength=1.2)
    ax[0].set_xlim(mlim); ax[0].set_ylim(alim)
    ax[0].set_xlabel(r"$\log_{10}\,M_\star/M_\odot$")
    ax[0].set_ylabel(r"$A_V$ (nebular)")

    fig.savefig(outpath)
    plt.close(fig)
    print(f"  saved {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cat_path", help="Multi-extension dwarf catalog FITS file.")
    ap.add_argument("--outdir", default=os.path.expanduser("~/Downloads"),
                    help="Directory for the output PDFs (default: ~/Downloads).")
    args = ap.parse_args()

    apply_paper_style()

    print(f"Reading {args.cat_path}")
    main_tab = Table.read(args.cat_path, hdu="MAIN")
    der_tab = Table.read(args.cat_path, hdu="SPEC_DERIVED")
    snr4363 = read_4363_snr(args.cat_path)

    os.makedirs(args.outdir, exist_ok=True)
    plot_direct_vs_strongline(
        main_tab, der_tab, snr4363,
        os.path.join(args.outdir, "direct_vs_strongline_zgas.pdf"))

    plot_av_direct_vs_param(
        main_tab, der_tab,
        os.path.join(args.outdir, "av_direct_vs_param.pdf"))


if __name__ == "__main__":
    main()
