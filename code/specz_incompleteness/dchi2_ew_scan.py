'''
Simulated Dchi2 vs true Halpha EW at realistic BGS Bright depths, overlaid on
the real DESI dwarf locus -- the modern re-make of the old slides' Dchi2-EW
comparison (DESI_completeness_model.pdf p.28ff), now with effective exposure
times SAMPLED from the real BGS Bright EFFTIME distribution instead of a
single nominal exptime.

Mocks: the user's FSPS models (quenched_fracs_nbs), emission + continuum,
selected to span EW(Halpha) ~ 0-60 A; each observed at quantile-sampled
EFFTIME_BRIGHT (= 0.1400 * TSNR2_BGS, Guy et al. 2023) drawn from the real
Iron BGS Bright catalog, at fixed fiber mag in the middle of the comparison
slice. NOTE: exptime = efftime with calibration factor 1.0 -- the old slides
showed this underpredicts Dchi2 (the pipeline's calibrate_efftime.py
milestone exists to fix the scale); interpret offsets with that in mind.

Real locus: DR1 dwarf catalog BGS_BRIGHT members (fastspec HALPHA_EW).

Run:
  python dchi2_ew_scan.py sim     # generate + observe (writes N_EFF files)
  python dchi2_ew_scan.py rr      # run redrock on all scan files (serial)
  python dchi2_ew_scan.py plot    # figure -> incompleteness_plots/
'''

import os
import sys
import glob
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append('/global/u1/v/virajvm/DESI2_LOWZ/feasiBGS')

import numpy as np
from astropy.table import Table, vstack

from forward_model_demo import (FSPS_DIR, add_scores_hdu, fix_fibermap,
                                scores_template, synth_rmag)

SCAN_DIR = "/pscratch/sd/v/virajvm/specz_incompleteness/ew_scan"
OUT_DIR = ("/global/homes/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs/"
           "incompleteness_plots")

BGS_CATALOG = "/pscratch/sd/v/virajvm/catalog/Iron_bgs_bright_all_phot_final_filter.fits"
DWARF_V5 = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v5.fits"

TSNR2_BGS_TO_EFFTIME = 0.1400
RFIB_SLICE = (20.5, 21.0)      # real-data fiber mag slice; mocks at midpoint
N_MODELS = 30                  # FSPS models spanning EW (incl. continuum-only)
N_EFF = 8                      # efftime quantiles sampled
Z_RANGE = (0.02, 0.12)
SEED = 20260715


# ---------------------------------------------------------------------------
def true_halpha_ew(wave_rest, flux):
    """Rest-frame EW(Halpha) from the noiseless model; sidebands avoid [NII]."""
    line = (wave_rest > 6554.6) & (wave_rest < 6574.6)
    side = ((wave_rest > 6500) & (wave_rest < 6540)) | \
           ((wave_rest > 6600) & (wave_rest < 6640))
    cont = np.median(flux[side])
    if cont <= 0 or line.sum() < 2:
        return 0.0
    dl = np.gradient(wave_rest[line])
    return float(np.sum((flux[line] / cont - 1.0) * dl))


def load_fsps_models():
    """All emission + continuum FSPS models with their true EW(Halpha)."""
    files = (sorted(glob.glob(f"{FSPS_DIR}/spectra_emi_*.txt"))
             + sorted(glob.glob(f"{FSPS_DIR}/spectra_cont_*.txt")))
    out = []
    for fn in files:
        w, f = np.loadtxt(fn, unpack=True)
        out.append((os.path.basename(fn), w, f, true_halpha_ew(w, f)))
    return out


def select_span(models, n=N_MODELS):
    """Subset spanning the EW range ~uniformly (plus the EW~0 floor)."""
    ews = np.array([m[3] for m in models])
    order = np.argsort(ews)
    idx = order[np.unique(np.linspace(0, len(order) - 1, n).astype(int))]
    return [models[i] for i in idx]


def sample_efftimes():
    """Quantile-sample the real BGS Bright EFFTIME distribution."""
    t = Table.read(BGS_CATALOG)
    eff = TSNR2_BGS_TO_EFFTIME * np.asarray(t["TSNR2_BGS"], dtype=float)
    eff = eff[np.isfinite(eff) & (eff > 0)]
    qs = np.linspace(0.08, 0.92, N_EFF)
    return np.quantile(eff, qs), eff


# ---------------------------------------------------------------------------
def simulate():
    os.makedirs(SCAN_DIR, exist_ok=True)
    from feasibgs import forwardmodel as FM
    fdesi = FM.fakeDESIspec()
    rng = np.random.default_rng(SEED)

    models = select_span(load_fsps_models())
    print(f"{len(models)} models, EW range "
          f"{models[0][3]:.1f} .. {max(m[3] for m in models):.1f} A "
          f"(sorted: {sorted(round(m[3],1) for m in models)})")

    efftimes, _ = sample_efftimes()
    print("efftime quantiles [s]:", np.round(efftimes, 0))

    wave_obs = np.arange(3530.0, 9922.0, 0.4)
    rfib = 0.5 * (RFIB_SLICE[0] + RFIB_SLICE[1])
    sc_tmpl = scores_template()

    for j, eff in enumerate(efftimes):
        rows, truth = [], []
        for name, w, f, ew in models:
            z = rng.uniform(*Z_RANGE)
            fobs = np.interp(wave_obs, w * (1 + z), f / (1 + z))
            m0 = synth_rmag(wave_obs, fobs)[0]
            rows.append(fobs * 10 ** (-0.4 * (rfib - m0)))
            truth.append((name, ew, z, eff))
        rows = np.array(rows)

        out = f"{SCAN_DIR}/ew_scan_eff{j:02d}.fits"
        fdesi.simExposure(wave=wave_obs, flux=rows, airmass=1.0,
                          exptime=float(eff), seeing=1.1,
                          seed=SEED + 1000 + j, filename=out)
        add_scores_hdu(out, len(rows), sc_tmpl)
        fix_fibermap(out)
        Table(rows=truth, names=("MODEL", "EW_TRUE", "Z_TRUE", "EFFTIME")).write(
            f"{SCAN_DIR}/ew_scan_eff{j:02d}_truth.fits", overwrite=True)
        print(f"wrote {out} (efftime {eff:.0f}s, {len(rows)} spectra)")


def run_redrock():
    for f in sorted(glob.glob(f"{SCAN_DIR}/ew_scan_eff??.fits")):
        out = f.replace("ew_scan_", "rr_ew_scan_")
        if os.path.exists(out):
            print("exists:", out); continue
        print("redrock:", f)
        subprocess.run(["rrdesi", "-i", f, "-o", out, "--mp", "16"], check=True)


# ---------------------------------------------------------------------------
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from plot_style import (apply_paper_style, make_subplots,
                            MARGIN_LABEL, MARGIN_CBAR, MARGIN_PAD)
    apply_paper_style()

    # simulated points
    sims = []
    for f in sorted(glob.glob(f"{SCAN_DIR}/rr_ew_scan_eff??.fits")):
        rr = Table.read(f, hdu="REDSHIFTS")
        rr = rr[np.argsort(np.asarray(rr["TARGETID"]))]
        tr = Table.read(f.replace("rr_ew_scan_", "ew_scan_").replace(
            ".fits", "_truth.fits"))
        assert len(rr) == len(tr)
        tr["DELTACHI2"] = np.asarray(rr["DELTACHI2"], dtype=float)
        tr["Z_RR"] = np.asarray(rr["Z"], dtype=float)
        tr["ZWARN"] = np.asarray(rr["ZWARN"], dtype=int)
        sims.append(tr)
    sims = vstack(sims)
    correct = (np.abs(sims["Z_RR"] - sims["Z_TRUE"]) / (1 + sims["Z_TRUE"])
               < 0.0033) & (sims["ZWARN"] == 0)
    print(f"sim points: {len(sims)}, correct-z fraction {np.mean(correct):.2f}")

    # real dwarf locus
    cat = Table.read(DWARF_V5)
    m = np.char.strip(np.asarray(cat["SAMPLE"]).astype(str)) == "BGS_BRIGHT"
    rf = np.asarray(cat["FIBERMAG_R"], dtype=float)
    m &= (rf > RFIB_SLICE[0]) & (rf < RFIB_SLICE[1])
    m &= (np.asarray(cat["Z"]) > 0.001) & (np.asarray(cat["Z"]) < 0.15)
    ew_r = np.asarray(cat["HALPHA_EW"], dtype=float)
    dchi2_r = np.asarray(cat["DELTACHI2"], dtype=float)
    m &= np.isfinite(ew_r) & (dchi2_r > 0)
    print(f"real BGS Bright dwarfs in slice: {m.sum()}")

    fig, flat = make_subplots(ncol=1, nrow=1, plot_size=3.6,
                              col_spacing=[MARGIN_LABEL, MARGIN_CBAR],
                              row_spacing=[MARGIN_LABEL, 0.62],
                              return_fig=True)
    ax = flat[0]

    hb = ax.hexbin(np.log10(dchi2_r[m].clip(1e-2)), ew_r[m],
                   gridsize=55, extent=[0, 4, -6, 64], cmap="Greys",
                   norm=LogNorm(), mincnt=1, rasterized=True)

    ok = np.asarray(correct)
    inrange = np.asarray(sims["EW_TRUE"]) < 64      # drop off-axis EW>64 models
    ldchi2 = np.log10(np.asarray(sims["DELTACHI2"]).clip(1e-2))
    ews = np.asarray(sims["EW_TRUE"])
    sc = ax.scatter(ldchi2[ok & inrange], ews[ok & inrange],
                    c=np.asarray(sims["EFFTIME"])[ok & inrange], cmap="viridis",
                    s=16, marker="o", edgecolors="k", linewidths=0.3,
                    label="FSPS mock (correct z)", zorder=5)
    ax.scatter(ldchi2[~ok & inrange], ews[~ok & inrange],
               c="crimson", s=22, marker="x", linewidths=0.9,
               label="FSPS mock (wrong z)", zorder=6)

    ax.axvline(np.log10(40), color="0.35", ls=":", lw=1)
    ax.text(np.log10(40) + 0.04, 30, r"$\Delta\chi^2=40$", fontsize=8,
            color="0.35", rotation=90, va="center")
    ax.set_xlim(0, 4)
    ax.set_ylim(-6, 64)
    ax.set_xlabel(r"$\log_{10}\,\Delta\chi^2$")
    ax.set_ylabel(r"EW(H$\alpha$) [$\mathrm{\AA}$]  (true for mocks, fastspec for data)")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.03, 0.82, "mock exptime = EFFTIME (uncalibrated, factor 1.0)\n"
            "leftward mock offset = the calibration milestone",
            transform=ax.transAxes, fontsize=7.5, color="0.45", va="top")

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"sampled EFFTIME$_{\rm BGS}$ [s]", fontsize=9)

    ax.set_title(rf"mocks at $r_{{\rm fib}}={np.mean(RFIB_SLICE):.2f}$ vs real "
                 rf"BGS Bright dwarfs (${RFIB_SLICE[0]}<r_{{\rm fib}}<{RFIB_SLICE[1]}$, "
                 r"$z<0.15$)", fontsize=9)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_DIR}/dchi2_vs_ew_scan.{ext}", dpi=250,
                    bbox_inches="tight")
        print(f"wrote {OUT_DIR}/dchi2_vs_ew_scan.{ext}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    {"sim": simulate, "rr": run_redrock, "plot": plot}[mode]()
