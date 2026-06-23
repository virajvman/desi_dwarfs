"""
Validate the position-angle (PA) conventions used in the dwarf catalog against
the SGA-2020 ground-truth PA.

Background
----------
SHAPE_PARAMS[:, 1] (the stored PA) is built two different ways depending on
MAG_TYPE (see consolidate_photometry.py:consolidate_positions_and_shapes):

  * TRACTOR_OG / clean (~95% of catalog):  PA = rad2deg(arctan2(e2, e1) / 2)
        -> Tractor ellipticity convention, range [-90, 90]
  * COG / SIMPLE / TRACTOR_BASED (aperture-reprocessed): PA = 90 + degrees(theta)
        -> photutils pixel-frame orientation, range [0, 180]

Legacy Surveys' own convention (legacypipe/doc/nb/dr8-lslga.ipynb) for the
on-sky, East-of-North PA is:

        pa_LS = 180 - ( -rad2deg(arctan2(e2, e1) / 2) )
              = 180 + rad2deg(arctan2(e2, e1) / 2)        (== our Tractor phi, mod 180)

SGA-2020 'PA' is a bona-fide East-of-North position angle in degrees, [0, 180).
SGA galaxies in our catalog are (almost all) MAG_TYPE == TRACTOR_OG, and the
intermediate SGA catalogs carry BOTH SGA_PA (truth) and SHAPE_E1/E2 (for the
Tractor PA) and the aperture fit (APER_PARAMS), so we can pin down the exact
correct transform for *both* code paths empirically.

This script tests several candidate transforms for each path and reports the
median absolute circular residual (mod 180, folded to [0, 90]) vs SGA_PA.
The transform with a small residual (a few deg) is the correct one; a wrong
sign shows up as a large residual that *correlates with SGA_PA* (diagnostic
plot saved to figs/).

Run on NERSC (the catalogs live on $PSCRATCH).
"""

import os
import numpy as np
from astropy.table import Table

# --- file locations -------------------------------------------------------
CAT_DIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs"
# Has SGA_PA, SGA_BA, SHAPE_E1, SHAPE_E2, TARGETID (from process_sga_matches):
SGA_REPROCESS = f"{CAT_DIR}/iron_desi_SGA_matched_dwarfs_REPROCESS_V2.fits"
# Has the aperture fit (APER_PARAMS*) + TARGETID, post photo-pipeline:
SGA_WAPER = f"{CAT_DIR}/iron_photometry/iron_SGA_sga_catalog_w_aper_mags.fits"

FIG_DIR = "/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/figs"


def circ_resid_180(a, b):
    """Absolute difference of two angles (deg) modulo 180, folded to [0, 90]."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 180.0
    return np.minimum(d, 180.0 - d)


def summarize(name, cand, truth, good):
    """Print residual stats for a candidate PA transform vs the truth PA."""
    r = circ_resid_180(cand[good], truth[good])
    print(
        f"  {name:<34s}  median={np.nanmedian(r):6.2f}  "
        f"mean={np.nanmean(r):6.2f}  frac<10deg={np.mean(r < 10):5.2f}  "
        f"frac<20deg={np.mean(r < 20):5.2f}"
    )
    return r


def get_aper_theta(tab):
    """Return the aperture orientation (radians) per row, or None if absent.

    Prefer the consolidated *_FINAL column written by consolidate_cog_photo;
    fall back to the raw APER_PARAMS. APER_PARAMS = [semimajor_pix, b/a, theta_rad].
    """
    for col in ("APER_PARAMS_FINAL", "APER_PARAMS"):
        if col in tab.colnames:
            arr = np.asarray(tab[col])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, 2], col
    return None, None


def main():
    print(f"Reading {SGA_REPROCESS}")
    rep = Table.read(SGA_REPROCESS)
    print(f"  N = {len(rep)};  columns of interest present:",
          [c for c in ("SGA_PA", "SGA_BA", "SHAPE_E1", "SHAPE_E2", "PHI", "BA", "TARGETID")
           if c in rep.colnames])

    # --- ground truth ---
    sga_pa_raw = np.asarray(rep["SGA_PA"], float)
    sga_pa = sga_pa_raw % 180.0                              # fold to [0,180)
    # SGA-2020 PA is in [0,180); missing entries are typically a negative sentinel.
    valid_truth = np.isfinite(sga_pa_raw) & (sga_pa_raw >= 0) & (sga_pa_raw <= 180)

    # --- the current-code Tractor PA (T1) ---
    # Prefer raw SHAPE_E1/E2 (exactly what construct_dwarf_galaxy_catalogs.py:673 uses);
    # fall back to the already-derived PHI column, which IS T1 by construction.
    if {"SHAPE_E1", "SHAPE_E2"}.issubset(rep.colnames):
        e1 = np.asarray(rep["SHAPE_E1"], float)
        e2 = np.asarray(rep["SHAPE_E2"], float)
        phi_user = np.rad2deg(np.arctan2(e2, e1) * 0.5)      # current code (signed, [-90,90])
        shape_ok = np.isfinite(e1) & np.isfinite(e2) & ((e1 != 0) | (e2 != 0))
        print("  Tractor PA derived from SHAPE_E1/E2.")
    elif "PHI" in rep.colnames:
        phi_user = np.asarray(rep["PHI"], float)
        shape_ok = np.isfinite(phi_user)
        print("  SHAPE_E1/E2 absent -> using stored PHI column directly as T1.")
    else:
        raise KeyError("Neither SHAPE_E1/E2 nor PHI present; cannot test Tractor path.")

    good = valid_truth & shape_ok
    print(f"  Usable rows (finite SGA_PA + Tractor shape): {good.sum()} / {len(rep)}")

    # ---------------------------------------------------------------
    # TRACTOR-PATH candidates (this is what TRACTOR_OG / clean store)
    # ---------------------------------------------------------------
    print("\nTRACTOR path  (SHAPE_E1/E2 -> PA),  vs SGA_PA:")
    summarize("T1  rad2deg(atan2(e2,e1)/2)   [current]", phi_user, sga_pa, good)
    summarize("T2  -rad2deg(atan2(e2,e1)/2)  [EllipseE]", -phi_user, sga_pa, good)
    summarize("T3  (180 + T1) % 180  [LS East-of-N]", (180 + phi_user) % 180, sga_pa, good)
    summarize("T4  (90  + T1) % 180", (90 + phi_user) % 180, sga_pa, good)
    summarize("T5  (90  - T1) % 180", (90 - phi_user) % 180, sga_pa, good)

    # ---------------------------------------------------------------
    # APERTURE-PATH candidates (COG/SIMPLE/TRACTOR_BASED store these).
    # We need the aperture theta; cross-match SGA_PA onto the w_aper file.
    # ---------------------------------------------------------------
    if os.path.exists(SGA_WAPER):
        print(f"\nReading {SGA_WAPER}")
        wap = Table.read(SGA_WAPER)
        theta, used_col = get_aper_theta(wap)
        if theta is not None and "TARGETID" in wap.colnames and "TARGETID" in rep.colnames:
            print(f"  aperture theta from {used_col} (radians)")
            # join SGA_PA (from rep) onto wap by TARGETID
            pa_by_tid = {int(t): p for t, p in zip(rep["TARGETID"], rep["SGA_PA"])}
            wtid = np.asarray(wap["TARGETID"]).astype(np.int64)
            sga_pa_w = np.array([pa_by_tid.get(int(t), np.nan) for t in wtid]) % 180.0
            gw = np.isfinite(sga_pa_w) & np.isfinite(theta)
            print(f"  matched aperture rows with SGA_PA: {gw.sum()} / {len(wap)}")
            adeg = np.degrees(theta)
            print("\nAPERTURE path  (photutils theta -> PA),  vs SGA_PA:")
            summarize("A1  90 + deg(theta)  [current]", (90 + adeg) % 180, sga_pa_w, gw)
            summarize("A2  90 - deg(theta)  [sign flip]", (90 - adeg) % 180, sga_pa_w, gw)
            summarize("A3  deg(theta) % 180", adeg % 180, sga_pa_w, gw)
            summarize("A4  -deg(theta) % 180", (-adeg) % 180, sga_pa_w, gw)

            # diagnostic plot: residual vs SGA_PA reveals a sign error as a slope
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                ax[0].scatter(sga_pa[good], circ_resid_180((180 + phi_user) % 180, sga_pa)[good], s=2)
                ax[0].set(title="Tractor T3 residual vs SGA_PA", xlabel="SGA_PA [deg]", ylabel="|Δ| mod180 [deg]")
                ax[1].scatter(sga_pa_w[gw], circ_resid_180((90 + adeg) % 180, sga_pa_w)[gw], s=2)
                ax[1].set(title="Aperture A1 residual vs SGA_PA", xlabel="SGA_PA [deg]", ylabel="|Δ| mod180 [deg]")
                os.makedirs(FIG_DIR, exist_ok=True)
                out = os.path.join(FIG_DIR, "pa_convention_validation.png")
                fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight")
                print(f"\nSaved diagnostic plot -> {out}")
            except Exception as exc:  # plotting is optional
                print(f"(plot skipped: {exc})")
        else:
            print("  Could not find aperture theta or TARGETID; skipping aperture check.")
    else:
        print(f"\n(aperture file not found: {SGA_WAPER}; skipping aperture check)")

    print("\nInterpretation:")
    print("  * The transform with the SMALL median residual (a few deg) is correct.")
    print("  * A wrong sign gives a LARGE residual whose value tracks SGA_PA in the plot.")
    print("  * Expect T1/T3 to match (they are equal mod 180). The aperture winner")
    print("    (A1 vs A2) tells you whether '90 + deg(theta)' needs flipping to '90 - deg(theta)'.")


if __name__ == "__main__":
    main()
