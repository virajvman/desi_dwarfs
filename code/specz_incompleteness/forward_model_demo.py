'''
Forward-modeling demo for the DESI dwarfs talk -- and a working miniature of
the injection-recovery validation planned for the specz incompleteness
pipeline.

Takes two noiseless desisim BGS basis-template spectra (one strong-emission,
one passive), places them at z=0.035, scales each to a ladder of r fiber
magnitudes, "observes" them at nominal BGS Bright depth (180s) with the
feasibgs/specsim forward model (many noise realizations each), runs redrock,
and shows where the redshift machinery loses each spectral type.

Outputs (paper style, plot_style.py) go to
~/DESI2_LOWZ/quenched_fracs_nbs/incompleteness_plots/:
  fwdmodel_montage.{png,pdf}   spectra ladder w/ redrock verdicts
  fwdmodel_recovery.{png,pdf}  success + Dchi2 vs fiber mag

Run (login node is fine):
  source /global/cfs/cdirs/desi/software/desi_environment.sh main
  python forward_model_demo.py sim      # generate + observe (~minutes)
  rrdesi -i <demo dir>/fwd_demo_SF.fits -o <demo dir>/rr_fwd_demo_SF.fits --mp 32
  rrdesi -i <demo dir>/fwd_demo_QU.fits -o <demo dir>/rr_fwd_demo_QU.fits --mp 32
  python forward_model_demo.py plot
'''

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") )
sys.path.append('/global/u1/v/virajvm/DESI2_LOWZ/feasiBGS')

import numpy as np
from astropy.io import fits
from astropy.table import Table

DEMO_DIR = "/pscratch/sd/v/virajvm/specz_incompleteness/fwd_demo"
OUT_DIR = ("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/"
           "incompleteness_plots")

Z_DEMO = 0.035
EXPTIME = 180.0                      # nominal BGS Bright dark-equivalent
MAGS = np.arange(19.5, 23.30, 0.25)  # r fiber-mag ladder
N_REAL = 20                          # noise realizations per mag
MONTAGE_MAGS = [20.0, 21.5, 22.75]   # rows of the montage figure
SEED0 = 20260715
CATASTROPHIC = 0.0033

C_SF, C_QU = "#2a6fbb", "#c23b22"


# ---------------------------------------------------------------------------
def get_filter():
    import speclite.filters
    for name in ("decamDR1noatm-r", "decam2014-r"):
        try:
            return speclite.filters.load_filters(name), name
        except Exception:
            continue
    raise RuntimeError("no r filter found")


def synth_rmag(wave, flux_1e17):
    """AB mag of f_lambda [1e-17 erg/s/cm2/A] spectra through the r filter."""
    import astropy.units as u
    filt, _ = get_filter()
    f = np.atleast_2d(flux_1e17) * 1e-17 * u.erg / (u.cm ** 2 * u.s * u.Angstrom)
    m = filt.get_ab_maggies(f, wave * u.Angstrom, axis=-1, mask_invalid=True)
    return -2.5 * np.log10(np.asarray(m[filt.names[0]]))


FSPS_DIR = "/global/u1/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/fsps_model_spectra"
FSPS_SF = f"{FSPS_DIR}/spectra_emi_mlogz_0.2_mlogu_1.5_tau_2_age_2.00.txt"
FSPS_QU = f"{FSPS_DIR}/spectra_cont_mlogz_0.5_mlogu_1.0_tau_1_age_10.00.txt"
# median-EW dwarf (EW(Halpha) ~ 21 A, near the real BGS-dwarf median)
FSPS_MED = f"{FSPS_DIR}/spectra_emi_mlogz_1.2_mlogu_3.5_tau_2_age_8.00.txt"


def pick_templates():
    """The user's own FSPS models (quenched_fracs_nbs era, as in the slides):
    a young tau-model with nebular emission (SF) and an old continuum-only
    model (quenched). Two-column text: wave [A], L_lambda. Absolute
    normalization is irrelevant -- each mock is rescaled to a target fiber
    mag -- so the two spectra are returned on a common wavelength grid.
    """
    spec = {}
    for name, path in (("SF", FSPS_SF), ("QU", FSPS_QU), ("MED", FSPS_MED)):
        w, f = np.loadtxt(path, unpack=True)
        spec[name] = (w, f)
        print(f"{name}: {os.path.basename(path)} "
              f"({len(w)} pix, {w.min():.0f}-{w.max():.0f} A rest)")
    # common grid = SF grid (they share the FSPS grid, but be safe)
    wave = spec["SF"][0]
    out = {name: np.interp(wave, w, f) for name, (w, f) in spec.items()}
    return wave, out


def scores_template():
    """Any real iron coadd provides the SCORES HDU redrock insists on."""
    pats = glob.glob("/global/cfs/cdirs/desi/spectro/redux/iron/healpix/"
                     "main/bright/*/*/coadd-*.fits")
    return pats[0]


def add_scores_hdu(file_path, n_rows, template_fits):
    with fits.open(file_path, mode="update") as hdulist:
        if "SCORES" in [h.name for h in hdulist]:
            return
        tab = fits.open(template_fits)["SCORES"]
        new = np.resize(tab.data, n_rows)
        hdulist.append(fits.BinTableHDU(data=new, header=tab.header,
                                        name="SCORES"))
        hdulist.flush()


def fix_fibermap(file_path):
    """feasibgs leaves OBJTYPE='' -- current redrock then flags every fiber
    NODATA|BAD_TARGET and returns z=0 without fitting. Same patch is needed
    in the production mock_observe backend."""
    with fits.open(file_path, mode="update") as hdulist:
        fm = hdulist["FIBERMAP"].data
        fm["OBJTYPE"] = "TGT"
        for col in ("FIBERSTATUS", "COADD_FIBERSTATUS"):
            if col in fm.columns.names:
                fm[col] = 0
        hdulist.flush()


# ---------------------------------------------------------------------------
def simulate():
    os.makedirs(DEMO_DIR, exist_ok=True)
    from feasibgs import forwardmodel as FM
    fdesi = FM.fakeDESIspec()

    wave_rest, tmpl = pick_templates()
    # spans desimodel ccd limits; step must divide the 0.8 A output pixel
    wave_obs = np.arange(3530.0, 9922.0, 0.4)

    sc_tmpl = scores_template()
    print("SCORES donor:", sc_tmpl)

    for k, (name, frest) in enumerate(tmpl.items()):
        if (os.path.exists(f"{DEMO_DIR}/fwd_demo_{name}.fits")
                and os.path.exists(f"{DEMO_DIR}/fwd_demo_{name}_truth.fits")):
            print(f"{name}: exists, skipping")
            continue
        # redshift, resample, and normalize once at mag=20, then scale
        fobs = np.interp(wave_obs, wave_rest * (1 + Z_DEMO), frest / (1 + Z_DEMO))
        m0 = synth_rmag(wave_obs, fobs)[0]

        rows, truth = [], []
        for mag in MAGS:
            scaled = fobs * 10 ** (-0.4 * (mag - m0))
            for r in range(N_REAL):
                rows.append(scaled)
                truth.append((name, mag, r))
        rows = np.array(rows)

        out = f"{DEMO_DIR}/fwd_demo_{name}.fits"
        fdesi.simExposure(wave=wave_obs, flux=rows, airmass=1.0,
                          exptime=EXPTIME, seeing=1.1,
                          seed=SEED0 + k, filename=out)
        add_scores_hdu(out, len(rows), sc_tmpl)
        fix_fibermap(out)

        tt = Table(rows=truth, names=("TYPE", "RFIB", "REALIZATION"))
        tt["Z_TRUE"] = Z_DEMO
        tt.write(f"{DEMO_DIR}/fwd_demo_{name}_truth.fits", overwrite=True)
        print(f"wrote {out} ({len(rows)} spectra) + truth")

    print("\nNow run redrock (login node OK):")
    for name in tmpl:
        print(f"  rrdesi -i {DEMO_DIR}/fwd_demo_{name}.fits "
              f"-o {DEMO_DIR}/rr_fwd_demo_{name}.fits --mp 32")


# ---------------------------------------------------------------------------
def load_results(name):
    # NOTE: simExposure per-camera grids are not aligned to a common 0.8 A
    # lattice, so modern desispec coadd_cameras() refuses them -- plot the
    # b/r/z cameras individually instead (same latent issue affects
    # measure_features.py; fix there is a common output grid).
    import desispec.io
    truth = Table.read(f"{DEMO_DIR}/fwd_demo_{name}_truth.fits")
    spec = desispec.io.read_spectra(f"{DEMO_DIR}/fwd_demo_{name}.fits")
    rr = Table.read(f"{DEMO_DIR}/rr_fwd_demo_{name}.fits", hdu="REDSHIFTS")
    rr = rr[np.argsort(np.asarray(rr["TARGETID"]))]     # row order = input order
    assert len(rr) == len(truth)
    return truth, spec, rr


def plot_figures():
    import matplotlib.pyplot as plt
    from plot_style import (apply_paper_style, make_subplots, reshape_axes,
                            MARGIN_LABEL, MARGIN_TICKS, MARGIN_PAD,
                            MARGIN_SHARED, MARGIN_SPLIT)
    apply_paper_style()

    res = {name: load_results(name) for name in ("SF", "QU")}

    # ------------------------------------------------- Fig 1: montage
    nrow, ncol = len(MONTAGE_MAGS), 2
    fig, flat = make_subplots(ncol=ncol, nrow=nrow, plot_size=2.9,
                              col_spacing=[MARGIN_LABEL, MARGIN_SHARED, MARGIN_PAD],
                              row_spacing=[MARGIN_LABEL] + [MARGIN_SHARED] * (nrow - 1) + [0.45],
                              return_fig=True)
    axes = reshape_axes(flat, nrow, ncol)

    for col, (name, color, title) in enumerate(
            [("SF", C_SF, "star-forming dwarf (strong lines)"),
             ("QU", C_QU, "quenched dwarf (continuum only)")]):
        truth, spec, rr = res[name]
        for row, mag in enumerate(MONTAGE_MAGS):
            ax = axes[row, col]
            i = np.flatnonzero((np.abs(truth["RFIB"] - mag) < 0.01)
                               & (truth["REALIZATION"] == 0))[0]
            k = np.ones(15) / 15                     # boxcar smooth for the eye
            allfl = []
            for band in spec.bands:
                fl = spec.flux[band][i]
                ax.plot(spec.wave[band], fl, color="0.78", lw=0.3, rasterized=True)
                ax.plot(spec.wave[band], np.convolve(fl, k, mode="same"),
                        color=color, lw=0.8)
                allfl.append(fl)
            fl = np.concatenate(allfl)

            zrr, dchi2, zw = rr["Z"][i], rr["DELTACHI2"][i], rr["ZWARN"][i]
            ok = (abs(zrr - Z_DEMO) / (1 + Z_DEMO) < CATASTROPHIC) and zw == 0
            verdict = ("correct z" if ok else "WRONG z")
            ax.text(0.03, 0.92,
                    rf"$r_{{\rm fib}}$ = {mag:.2f}",
                    transform=ax.transAxes, va="top", fontsize=10)
            ax.text(0.97, 0.92,
                    rf"redrock: {verdict}" + "\n"
                    + rf"$\Delta\chi^2$ = {dchi2:.0f}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=9,
                    color=("k" if ok else "crimson"))
            ax.set_xlim(3600, 9800)
            lim = np.percentile(fl, [1, 99])
            ax.set_ylim(lim[0] - 0.15 * (lim[1] - lim[0]), lim[1] * 1.45)
            if row == 0:
                ax.set_title(title, fontsize=11, color=color)
            if row == nrow - 1:
                ax.set_xlabel(r"observed wavelength [$\mathrm{\AA}$]")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(r"$f_\lambda$ [$10^{-17}\,{\rm erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]")
            else:
                ax.set_yticklabels([])
    fig.text(0.5, 0.975,
             f"same model dwarfs, dimmed and re-observed at BGS depth "
             rf"({EXPTIME:.0f}s, $z_{{\rm true}}$ = {Z_DEMO})",
             ha="center", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_DIR}/fwdmodel_montage.{ext}", dpi=250,
                    bbox_inches="tight")
        print(f"wrote {OUT_DIR}/fwdmodel_montage.{ext}")
    plt.close(fig)

    # ------------------------------------------------- Fig 2: recovery curves
    fig, flat = make_subplots(ncol=2, nrow=1, plot_size=2.9,
                              col_spacing=[MARGIN_LABEL, MARGIN_SPLIT, MARGIN_PAD],
                              row_spacing=[MARGIN_LABEL, 0.45],
                              return_fig=True)
    ax_s, ax_d = flat[0], flat[1]

    for name, color, label in [("SF", C_SF, "star-forming"),
                               ("QU", C_QU, "quenched")]:
        truth, spec, rr = res[name]
        zrr = np.asarray(rr["Z"]); zw = np.asarray(rr["ZWARN"])
        dchi2 = np.asarray(rr["DELTACHI2"], dtype=float)
        ok = (np.abs(zrr - Z_DEMO) / (1 + Z_DEMO) < CATASTROPHIC) & (zw == 0)
        ok40 = ok & (dchi2 > 40)
        mags = np.asarray(truth["RFIB"])

        succ = [ok[mags == m].mean() for m in MAGS]
        succ40 = [ok40[mags == m].mean() for m in MAGS]
        ax_s.plot(MAGS, succ, "-o", color=color, ms=3.5,
                  label=f"{label}, ZWARN=0")
        ax_s.plot(MAGS, succ40, "--s", color=color, ms=3.5, alpha=0.75,
                  label=rf"{label}, + $\Delta\chi^2>40$")

        med = [np.median(dchi2[(mags == m)].clip(1e-2)) for m in MAGS]
        lo = [np.percentile(dchi2[(mags == m)].clip(1e-2), 16) for m in MAGS]
        hi = [np.percentile(dchi2[(mags == m)].clip(1e-2), 84) for m in MAGS]
        ax_d.plot(MAGS, med, "-o", color=color, ms=3.5, label=label)
        ax_d.fill_between(MAGS, lo, hi, color=color, alpha=0.18, lw=0)

    ax_s.set_xlabel(r"$r$ fiber magnitude")
    ax_s.set_ylabel("redshift recovery fraction")
    ax_s.set_ylim(-0.03, 1.06)
    ax_s.legend(fontsize=8.5, loc="lower left")

    ax_d.axhline(40, color="0.4", ls=":", lw=1)
    ax_d.text(MAGS[0] + 0.08, 48, r"$\Delta\chi^2 = 40$ cut", fontsize=9, color="0.4")
    ax_d.set_yscale("log")
    ax_d.set_xlabel(r"$r$ fiber magnitude")
    ax_d.set_ylabel(r"redrock $\Delta\chi^2$ (median, 16-84%)")
    ax_d.legend(fontsize=9, loc="upper right")

    fig.text(0.5, 0.96,
             rf"forward-modeled recovery: {N_REAL} noise realizations / mag, "
             rf"BGS depth ({EXPTIME:.0f}s), $z_{{\rm true}}$ = {Z_DEMO}",
             ha="center", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_DIR}/fwdmodel_recovery.{ext}", dpi=250,
                    bbox_inches="tight")
        print(f"wrote {OUT_DIR}/fwdmodel_recovery.{ext}")


def plot_recovery_single():
    """Single-panel recovery figure: one curve per type, catalog-style cut
    (ZWARN=0 & correct & Dchi2>40) only. EW(Halpha) of each template in the
    legend."""
    import matplotlib.pyplot as plt
    from plot_style import apply_paper_style, make_subplots, MARGIN_LABEL, MARGIN_PAD
    from dchi2_ew_scan import true_halpha_ew
    apply_paper_style()

    ews = {}
    for name, path in (("SF", FSPS_SF), ("QU", FSPS_QU), ("MED", FSPS_MED)):
        w, f = np.loadtxt(path, unpack=True)
        ews[name] = true_halpha_ew(w, f)

    fig, flat = make_subplots(ncol=1, nrow=1, plot_size=2.5,
                              col_spacing=[MARGIN_LABEL, MARGIN_PAD],
                              row_spacing=[MARGIN_LABEL, MARGIN_PAD],
                              return_fig=True)
    ax = flat[0]
    for name, color, label in [
            ("SF", C_SF, r"H$\alpha$ EW $\sim$ 100"),
            ("MED", "#3d9970", r"H$\alpha$ EW $\sim$ 20"),
            ("QU", C_QU, "quenched")]:
        truth, spec, rr = load_results(name)
        zrr = np.asarray(rr["Z"]); zw = np.asarray(rr["ZWARN"])
        dchi2 = np.asarray(rr["DELTACHI2"], dtype=float)
        ok = ((np.abs(zrr - Z_DEMO) / (1 + Z_DEMO) < CATASTROPHIC)
              & (zw == 0) & (dchi2 > 40))
        mags = np.asarray(truth["RFIB"])
        succ = [ok[mags == m].mean() for m in MAGS]
        ax.plot(MAGS, succ, "-o", color=color, ms=4, lw=1.8, label=label)

    ax.set_xlabel(r"$r$ fiber magnitude")
    ax.set_ylabel("redshift recovery fraction")
    ax.set_xlim(19.5, 23.25)
    ax.set_ylim(0, 1)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_DIR}/fwdmodel_recovery_single.{ext}", dpi=250,
                    bbox_inches="tight")
        print(f"wrote {OUT_DIR}/fwdmodel_recovery_single.{ext}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    if mode == "sim":
        simulate()
    elif mode == "plot":
        plot_figures()
    elif mode == "plot-single":
        plot_recovery_single()
    else:
        raise SystemExit("usage: forward_model_demo.py [sim|plot|plot-single]")
