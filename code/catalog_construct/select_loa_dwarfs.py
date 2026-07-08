#!/usr/bin/env python
"""
select_loa_dwarfs.py
====================

Standalone dwarf-galaxy selector for the DESI "loa" healpix zcatalog (v1).

The loa counterpart of `select_matterhorn_bgs_lowz.py`. Selects BGS_BRIGHT +
BGS_FAINT + LOW_Z (spare-fiber secondary) targets across main + SV1/2/3, applies
the redshift-robustness + science cuts, dereddens g/r/z, computes the FIDUCIAL
stellar mass (de los Reyes+2024 Eq. 13, via `get_stellar_mass_mia`), and keeps
only DWARFS (log Mstar < --logmstar-max). Writes a single FITS catalog with
per-sample boolean flags and everything a later Tractor cross-match needs.

Why this differs from the matterhorn selector
----------------------------------------------
loa is still HEALPIX (not the unipix layout matterhorn uses), and its zcatalog is
in the SAME format as iron/DR1: ONE row per file, NO base/imaging/-extra split and
NO `_BEST` column suffixes. So compared to select_matterhorn_bgs_lowz.py:

  * one file read (`zall-{group}-loa.fits`, ext ZCATALOG), not three;
  * redshift columns are Z / ZWARN / DELTACHI2 / SPECTYPE (no `_BEST`);
  * FLUX_G/R/Z + FLUX_IVAR_G/R/Z live in the SAME file (they are dropped from the
    matterhorn base file into a separate imaging file) -- so photometry for the
    mass estimate needs no cross-match here.

Photometry: what is and isn't in the loa zcatalog
-------------------------------------------------
Present (LS "tractor input", targeting):  FLUX_G/R/Z, FLUX_IVAR_G/R/Z,
  FIBERFLUX_*, EBV, PHOTSYS, RELEASE, BRICKID, BRICK_OBJID, BRICKNAME, MASKBITS.
NOT present (Tractor-catalog only), and therefore deferred to a cross-match:
  * FRACFLUX_G/R/Z (+ FRACMASKED/FRACIN/RCHISQ/NOBS/SHAPE_*/SERSIC) -- needed for
    the shred / FRACFLUX < 0.35 cut.
  * MW_TRANSMISSION_* -- but only EBV + PHOTSYS are needed to deredden, so we do
    that here with desiutil.dust.mwdust_transmission (match_legacy_surveys=True).

To make the FRACFLUX cross-match a deterministic key join with no re-read, this
catalog carries RELEASE, BRICKID, BRICK_OBJID, BRICKNAME, PHOTSYS, EBV plus the
raw FLUX_*/FLUX_IVAR_*. Feed the output straight into
`crossmatch_tractorphot.py` (stage 2), whose gather_tractorphot pass adds
FRACFLUX_* etc. and applies the FRACFLUX cut -- loa has no public
lsdr9-photometry VAC, so gather_tractorphot (reading the DR9 sweeps directly) is
the production-independent way to get FRACFLUX.

Stellar mass / dwarf cut
------------------------
log Mstar comes from `get_stellar_mass_mia` (desi_lowz_funcs), the project's
fiducial de los Reyes+2024 Eq. 13 estimator (valid for Mstar < 1e10, i.e. dwarfs).
Inputs are the MW-dereddened g-r colour and g magnitude and the redrock redshift
(input_zred=True). Per the request, NO k-correction handling beyond what the
function does internally, and NO nebular (emission-line) correction to the
broadband mags -- these are the raw dereddened Legacy-Surveys mags. For the very
nearest objects a flow-model distance (as in the main pipeline's Z_CMB /
DIST_MPC_FIDU) would be better than luminosity_distance(z); that refinement is
out of scope here. Rows with non-positive flux (undefined mag/mass) are dropped
by the dwarf cut.

Cuts applied (all tunable via CLI)
----------------------------------
  targeting   : (BGS_BRIGHT | BGS_FAINT | LOW_Z_TIER*) over main + SV1/2/3
  redshift    : GOOD_SPEC & (ZWARN == 0) & (DELTACHI2 > --deltachi2-min)
  spectype    : SPECTYPE == 'GALAXY'             (toggle: --no-require-galaxy)
  redshift z  : --zmin < Z < --zmax              (defaults 0.001 .. 0.2)
  uniqueness  : ZCAT_PRIMARY == True             (toggle: --no-primary-only)
  dwarf       : LOGM_M24 < --logmstar-max        (default 9.25)

Memory / where to run
---------------------
`zall-pix-loa.fits` is ~a couple GB; a single-pass column read holds a few GB in
RAM. Run on an interactive/compute node (e.g. `salloc ...`), NOT a login node.

Examples
--------
  python select_loa_dwarfs.py
  python select_loa_dwarfs.py --logmstar-max 9.0 --zmax 0.1
  python select_loa_dwarfs.py --group tilecumulative --output /pscratch/sd/v/virajvm/loa/loa_dwarfs.fits

Author: Viraj Manwadkar (desi_dwarfs)
"""

import os
import sys
import argparse

import numpy as np
import fitsio
from astropy.table import Table

from desiutil.log import get_logger
from desiutil.dust import mwdust_transmission

# Target-selection bitmasks: import by NAME (not hardcoded bit numbers) so any
# main-vs-SV bit differences are handled automatically.
from desitarget.targetmask import bgs_mask, scnd_mask
from desitarget.sv1.sv1_targetmask import bgs_mask as sv1_bgs_mask, scnd_mask as sv1_scnd_mask
from desitarget.sv2.sv2_targetmask import bgs_mask as sv2_bgs_mask, scnd_mask as sv2_scnd_mask
from desitarget.sv3.sv3_targetmask import bgs_mask as sv3_bgs_mask, scnd_mask as sv3_scnd_mask

# Fiber-status quality bits (for GOOD_SPEC, matching desispec.validredshifts).
from desispec.maskbits import fibermask

# Fiducial stellar-mass estimator lives in the top-level code dir.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from desi_lowz_funcs import get_stellar_mass_mia  # noqa: E402

log = get_logger()

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_INPUT_DIR = "/global/cfs/cdirs/desi/spectro/redux/loa/zcatalog/v1"
DEFAULT_SPECPROD = "loa"
DEFAULT_OUTPUT_DIR = "/pscratch/sd/v/virajvm/loa"

# (survey_prefix, bgs_mask, scnd_mask). '' == main survey (unprefixed columns).
SURVEY_MASKS = [
    ("", bgs_mask, scnd_mask),
    ("SV1_", sv1_bgs_mask, sv1_scnd_mask),
    ("SV2_", sv2_bgs_mask, sv2_scnd_mask),
    ("SV3_", sv3_bgs_mask, sv3_scnd_mask),
]

BANDS = ("G", "R", "Z")

# Columns we must have (selection + photometry for the mass + carried join keys).
REQUIRED = [
    "TARGETID", "TARGET_RA", "TARGET_DEC",
    "Z", "ZWARN", "DELTACHI2", "SPECTYPE",
    "ZCAT_PRIMARY", "COADD_FIBERSTATUS", "OBJTYPE",
    "DESI_TARGET", "BGS_TARGET", "SCND_TARGET",
    "FLUX_G", "FLUX_R", "FLUX_Z",
    "FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z",
    "EBV", "PHOTSYS",
    "RELEASE", "BRICKID", "BRICK_OBJID", "BRICKNAME",
]
# Columns we use if present (SV targeting bits + provenance + extras).
OPTIONAL = [
    "SURVEY", "PROGRAM", "HEALPIX", "MASKBITS", "MORPHTYPE",
    "SV1_BGS_TARGET", "SV1_SCND_TARGET",
    "SV2_BGS_TARGET", "SV2_SCND_TARGET",
    "SV3_BGS_TARGET", "SV3_SCND_TARGET",
    "FIBERFLUX_G", "FIBERFLUX_R", "FIBERFLUX_Z",
]


# --------------------------------------------------------------------------- #
# Helpers (shared style with select_matterhorn_bgs_lowz.py)
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Select BGS_BRIGHT/BGS_FAINT/LOW_Z dwarf galaxies from the loa zcatalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                   help="Directory holding the zall-{group}-{specprod}.fits file.")
    p.add_argument("--specprod", default=DEFAULT_SPECPROD,
                   help="Spectro production name in the filename.")
    p.add_argument("--group", default="pix", choices=["pix", "tilecumulative"],
                   help="Which zall grouping to read (loa is healpix -> 'pix').")
    p.add_argument("--output", default=None,
                   help="Output FITS path. Default: "
                        f"{DEFAULT_OUTPUT_DIR}/loa_<group>_bgs_lowz_dwarfs.fits")
    p.add_argument("--zmin", type=float, default=0.001, help="Lower Z bound (exclusive).")
    p.add_argument("--zmax", type=float, default=0.2, help="Upper Z bound (exclusive).")
    p.add_argument("--deltachi2-min", type=float, default=40.0,
                   help="DELTACHI2 threshold for a robust redshift.")
    p.add_argument("--logmstar-max", type=float, default=9.25,
                   help="Keep only dwarfs with LOGM_M24 < this (log10 Msun).")
    p.add_argument("--require-galaxy", action=argparse.BooleanOptionalAction, default=True,
                   help="Require SPECTYPE == 'GALAXY'.")
    p.add_argument("--primary-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep only ZCAT_PRIMARY == True (one row per TARGETID).")
    return p.parse_args()


def norm_str(arr):
    """Normalize a FITS string/bytes column to a stripped unicode array."""
    return np.char.strip(np.asarray(arr).astype(str))


def or_bits(mask, names):
    """OR together the integer values of `names` that exist in BitMask `mask`."""
    available = set(mask.names())
    val = 0
    for n in names:
        if n in available:
            val |= int(mask[n])
    return val


def col_bit_match(rec, colname, bitval):
    """Boolean array: (rec[colname] & bitval) != 0, robust to a missing column / zero mask."""
    n = rec.shape[0]
    if bitval == 0 or colname not in rec.dtype.names:
        return np.zeros(n, dtype=bool)
    return (rec[colname] & bitval) != 0


def open_table_hdu(fits_obj, extname):
    """Return the requested EXTNAME HDU, falling back to HDU 1."""
    try:
        return fits_obj[extname]
    except (OSError, ValueError, KeyError):
        return fits_obj[1]


def read_subset(path, extname, required, optional, rows=None):
    """Read `required` + present `optional` columns from the named HDU."""
    with fitsio.FITS(path) as f:
        hdu = open_table_hdu(f, extname)
        avail = set(hdu.get_colnames())
        missing_req = [c for c in required if c not in avail]
        if missing_req:
            raise KeyError(f"{os.path.basename(path)} [{extname}] is missing required "
                           f"columns: {missing_req}")
        missing_opt = [c for c in optional if c not in avail]
        if missing_opt:
            log.warning(f"{os.path.basename(path)}: optional columns absent (skipped): {missing_opt}")
        use = required + [c for c in optional if c in avail]
        data = hdu.read(columns=use, rows=rows)
    return data


def build_target_masks(rec):
    """Return (is_bgs_bright, is_bgs_faint, is_lowz) boolean arrays over main + SV."""
    n = rec.shape[0]
    is_bb = np.zeros(n, dtype=bool)
    is_bf = np.zeros(n, dtype=bool)
    is_lowz = np.zeros(n, dtype=bool)

    for prefix, bmask, smask in SURVEY_MASKS:
        bgs_col = f"{prefix}BGS_TARGET"
        scnd_col = f"{prefix}SCND_TARGET"

        is_bb |= col_bit_match(rec, bgs_col, or_bits(bmask, ["BGS_BRIGHT"]))
        is_bf |= col_bit_match(rec, bgs_col, or_bits(bmask, ["BGS_FAINT"]))

        lowz_names = [nm for nm in smask.names() if nm.startswith("LOW_Z")]
        is_lowz |= col_bit_match(rec, scnd_col, or_bits(smask, lowz_names))
        if lowz_names and scnd_col in rec.dtype.names:
            log.info(f"  {prefix or 'MAIN'}: LOW_Z tiers used = {lowz_names}")

    return is_bb, is_bf, is_lowz


def compute_good_spec(rec):
    """GOOD_SPEC recomputed from the zcatalog exactly as desispec.validredshifts:
    good fiber hardware status (tolerating only RESTRICTED and VARIABLE bits) AND
    a science target (OBJTYPE == 'TGT')."""
    okmask = fibermask.mask("RESTRICTED|VARIABLE")
    good_fiber = (rec["COADD_FIBERSTATUS"] & okmask) == rec["COADD_FIBERSTATUS"]
    is_tgt = norm_str(rec["OBJTYPE"]) == "TGT"
    return good_fiber & is_tgt


def raw_mag(flux):
    """22.5 - 2.5*log10(flux); NaN where flux <= 0."""
    flux = np.asarray(flux, dtype=np.float64)
    mag = np.full(flux.shape, np.nan, dtype=np.float64)
    good = flux > 0
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    return mag


def mw_transmission(ebv, photsys, band):
    """MW dust transmission (0-1) from EBV + PHOTSYS, Legacy-Surveys-matched.

    The loa zcatalog carries EBV + PHOTSYS but NOT MW_TRANSMISSION_*, so we
    recompute it. Rows whose PHOTSYS is not 'N'/'S' (should be none for real
    targets) get transmission=1 (left uncorrected)."""
    ebv = np.asarray(ebv, dtype=np.float64)
    ps = norm_str(photsys)
    trans = np.ones(ebv.shape, dtype=np.float64)
    valid = np.isin(ps, ["N", "S"])
    if valid.any():
        trans[valid] = mwdust_transmission(ebv[valid], band, ps[valid],
                                            match_legacy_surveys=True)
    if not valid.all():
        log.warning(f"MW_TRANSMISSION_{band}: {int((~valid).sum()):,} rows have "
                    f"PHOTSYS not in {{N,S}}; left uncorrected (transmission=1).")
    return trans


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    in_path = os.path.join(args.input_dir, f"zall-{args.group}-{args.specprod}.fits")
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    if args.output is None:
        out_path = os.path.join(DEFAULT_OUTPUT_DIR,
                                f"loa_{args.group}_bgs_lowz_dwarfs.fits")
    else:
        out_path = args.output

    log.info(f"input  : {in_path}")
    log.info(f"output : {out_path}")

    # ---- 1. Read columns (one pass) ---------------------------------------- #
    log.info("Reading zcatalog columns (single-pass over the loa zall file)...")
    rec = read_subset(in_path, "ZCATALOG", REQUIRED, OPTIONAL)
    n_total = rec.shape[0]
    log.info(f"  rows: {n_total:,}")

    # ---- 2. Target-selection masks (main + SV1/2/3) ------------------------ #
    log.info("Building target-selection masks (main + SV1/2/3)...")
    is_bb, is_bf, is_lowz = build_target_masks(rec)
    is_target = is_bb | is_bf | is_lowz
    log.info(f"  BGS_BRIGHT={is_bb.sum():,}  BGS_FAINT={is_bf.sum():,}  "
             f"LOW_Z={is_lowz.sum():,}  (union={is_target.sum():,})")

    # ---- 3. Quality + science cuts ----------------------------------------- #
    good_spec = compute_good_spec(rec)
    good_z = good_spec & (rec["ZWARN"] == 0) & (rec["DELTACHI2"] > args.deltachi2_min)

    z = rec["Z"]
    z_in_range = (z > args.zmin) & (z < args.zmax)

    is_galaxy = norm_str(rec["SPECTYPE"]) == "GALAXY"
    is_primary = rec["ZCAT_PRIMARY"].astype(bool)

    presel = is_target & good_z & z_in_range
    if args.require_galaxy:
        presel &= is_galaxy
    if args.primary_only:
        presel &= is_primary

    log.info("Cut cascade (cumulative, on target sample):")
    log.info(f"  targeting            : {is_target.sum():,}")
    log.info(f"  + GOOD_SPEC&ZWARN&dX2: {(is_target & good_z).sum():,}")
    log.info(f"  + {args.zmin} < z < {args.zmax}      : {(is_target & good_z & z_in_range).sum():,}")
    if args.require_galaxy:
        log.info(f"  + SPECTYPE=='GALAXY'  : {(is_target & good_z & z_in_range & is_galaxy).sum():,}")
    if args.primary_only:
        log.info(f"  + ZCAT_PRIMARY        : {presel.sum():,}")

    idx = np.where(presel)[0]
    log.info(f"Pre-mass selection: {idx.size:,} / {n_total:,}")

    # Subset now, so dereddening + the mass estimate touch only survivors.
    rec = rec[idx]
    sel_bb, sel_bf, sel_lowz = is_bb[idx], is_bf[idx], is_lowz[idx]
    sel_good_spec = good_spec[idx]

    # ---- 4. Deredden g/r/z, compute fiducial stellar mass, cut to dwarfs --- #
    trans = {b: mw_transmission(rec["EBV"], rec["PHOTSYS"], b) for b in BANDS}
    mag_raw = {b: raw_mag(rec[f"FLUX_{b}"]) for b in BANDS}
    with np.errstate(invalid="ignore", divide="ignore"):
        flux_dered = {b: np.where(trans[b] > 0, rec[f"FLUX_{b}"] / trans[b], np.nan)
                      for b in BANDS}
    mag_dered = {b: raw_mag(flux_dered[b]) for b in BANDS}

    gr_dered = mag_dered["G"] - mag_dered["R"]
    logm = np.asarray(get_stellar_mass_mia(gr_dered, mag_dered["G"], rec["Z"],
                                           input_zred=True), dtype=np.float64)

    with np.errstate(invalid="ignore"):
        is_dwarf = logm < args.logmstar_max      # NaN mass (bad flux) -> False -> dropped
    n_nan = int(np.isnan(logm).sum())
    log.info(f"Stellar mass: {n_nan:,} rows have undefined mass (non-positive flux); dropped.")
    log.info(f"Dwarf cut (LOGM_M24 < {args.logmstar_max}): "
             f"{int(is_dwarf.sum()):,} / {idx.size:,} kept.")

    keep = np.where(is_dwarf)[0]
    rec = rec[keep]
    sel_bb, sel_bf, sel_lowz = sel_bb[keep], sel_bf[keep], sel_lowz[keep]
    sel_good_spec = sel_good_spec[keep]
    logm = logm[keep]
    gr_dered = gr_dered[keep]
    trans = {b: trans[b][keep] for b in BANDS}
    mag_raw = {b: mag_raw[b][keep] for b in BANDS}
    mag_dered = {b: mag_dered[b][keep] for b in BANDS}

    if rec.shape[0] == 0:
        log.warning("No dwarfs passed the cuts -- writing an empty catalog.")

    # ---- 5. Assemble output ------------------------------------------------ #
    out = Table()
    out["TARGETID"] = rec["TARGETID"]
    out["TARGET_RA"] = rec["TARGET_RA"]
    out["TARGET_DEC"] = rec["TARGET_DEC"]
    for c in ("SURVEY", "PROGRAM"):
        if c in rec.dtype.names:
            out[c] = norm_str(rec[c])
    if "HEALPIX" in rec.dtype.names:
        out["HEALPIX"] = rec["HEALPIX"]

    # Per-sample membership flags (an object can belong to more than one).
    out["IS_BGS_BRIGHT"] = sel_bb
    out["IS_BGS_FAINT"] = sel_bf
    out["IS_LOWZ"] = sel_lowz

    # Raw targeting-bit columns that exist.
    for c in ("DESI_TARGET", "BGS_TARGET", "SCND_TARGET",
              "SV1_BGS_TARGET", "SV1_SCND_TARGET",
              "SV2_BGS_TARGET", "SV2_SCND_TARGET",
              "SV3_BGS_TARGET", "SV3_SCND_TARGET"):
        if c in rec.dtype.names:
            out[c] = rec[c]

    # Redshift + quality.
    out["Z"] = rec["Z"]
    out["ZWARN"] = rec["ZWARN"]
    out["DELTACHI2"] = rec["DELTACHI2"]
    out["SPECTYPE"] = norm_str(rec["SPECTYPE"])
    out["ZCAT_PRIMARY"] = rec["ZCAT_PRIMARY"].astype(bool)
    out["COADD_FIBERSTATUS"] = rec["COADD_FIBERSTATUS"]
    out["GOOD_SPEC"] = sel_good_spec

    # Photometry: raw fluxes (from zcatalog) + derived transmission + mags.
    for b in BANDS:
        out[f"FLUX_{b}"] = rec[f"FLUX_{b}"]
        out[f"FLUX_IVAR_{b}"] = rec[f"FLUX_IVAR_{b}"]
        out[f"MW_TRANSMISSION_{b}"] = trans[b]          # derived from EBV+PHOTSYS
        out[f"MAG_{b}"] = mag_raw[b]                     # raw (not dereddened)
        out[f"MAG_{b}_DERED"] = mag_dered[b]             # MW-dereddened
    for c in ("FIBERFLUX_G", "FIBERFLUX_R", "FIBERFLUX_Z"):
        if c in rec.dtype.names:
            out[c] = rec[c]

    # Fiducial stellar mass (de los Reyes+2024 Eq.13; dereddened g-r, g; redrock z).
    out["GR_DERED"] = gr_dered
    out["LOGM_M24"] = logm

    # Join keys + metadata for the stage-2 Tractor cross-match (FRACFLUX etc).
    for c in ("RELEASE", "BRICKID", "BRICK_OBJID", "BRICKNAME", "PHOTSYS",
              "EBV", "MASKBITS", "MORPHTYPE"):
        if c in rec.dtype.names:
            out[c] = norm_str(rec[c]) if rec[c].dtype.kind in ("S", "U") else rec[c]

    # ---- 6. Provenance header + write -------------------------------------- #
    out.meta["EXTNAME"] = "BGS_LOWZ_DWARFS"
    out.meta["SPECPROD"] = args.specprod
    out.meta["ZGROUP"] = args.group
    out.meta["INFILE"] = os.path.basename(in_path)
    out.meta["NIN"] = n_total
    out.meta["NSEL"] = int(rec.shape[0])
    out.meta["ZMIN"] = args.zmin
    out.meta["ZMAX"] = args.zmax
    out.meta["DCHI2MIN"] = args.deltachi2_min
    out.meta["LOGMMAX"] = args.logmstar_max
    out.meta["REQGAL"] = args.require_galaxy
    out.meta["PRIMONLY"] = args.primary_only
    out.meta["COMMENT"] = (
        "BGS_BRIGHT/BGS_FAINT/LOW_Z dwarf selection from loa zall (main+SV1/2/3). "
        "LOGM_M24 = get_stellar_mass_mia (de los Reyes+2024 Eq.13) from MW-dereddened "
        "g-r, g and redrock z (no k-corr beyond the estimator; no nebular corr). "
        "FRACFLUX cut NOT applied -- run crossmatch_tractorphot.py (stage 2) to attach "
        "FRACFLUX_* via gather_tractorphot and cut. FLUX_* are raw; MW_TRANSMISSION_* "
        "were derived from EBV+PHOTSYS."
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.write(out_path, format="fits", overwrite=True)
    log.info(f"Wrote {len(out):,} dwarfs -> {out_path}")
    log.info("NEXT: attach FRACFLUX_* (+ FRACMASKED/RCHISQ/NOBS/MW_TRANSMISSION) with")
    log.info(f"  python crossmatch_tractorphot.py --input {out_path} \\")
    log.info("      --output <loa_dwarfs_clean.fits>   # gather_tractorphot on DR9 sweeps")


if __name__ == "__main__":
    main()
