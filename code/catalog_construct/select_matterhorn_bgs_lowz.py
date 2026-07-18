#!/usr/bin/env python
"""
select_matterhorn_bgs_lowz.py
=============================

Standalone selector for the DESI Y5 "matterhorn" zall zcatalog (v2).

Selects BGS_BRIGHT + BGS_FAINT + LOW_Z (spare-fiber secondary) targets across
main + SV1/2/3, applies redshift-robustness + basic science cuts, computes
RAW g/r/z magnitudes, and writes a single FITS catalog with per-sample boolean
flags. This is a clean, reusable selector -- it is NOT wired into the main
dwarf-catalog pipeline.

Matterhorn format notes (v2 zcatalog, different from iron/DR1)
-------------------------------------------------------------
The zall catalog is split into THREE row-matched files per group (`pix` or
`tilecumulative`); the three share identical row order (TARGETID is asserted
equal across them by the producing code, desispec/scripts/zcatalog.py):

  zall-{group}-matterhorn.fits          ext ZCATALOG          redshift + targeting
  zall-{group}-matterhorn-imaging.fits  ext ZCATALOG_IMAGING  photometry
  zall-{group}-matterhorn-extra.fits    ext ZCATALOG_EXTRA    everything else (~50 GB)

Redshift columns are renamed with a `_BEST` suffix (Z_BEST, ZWARN_BEST,
DELTACHI2_BEST, SPECTYPE_BEST). This script reads ONLY the base + imaging
files; it never touches the giant `-extra` file (GOOD_SPEC is recomputed from
COADD_FIBERSTATUS + OBJTYPE using the exact desispec.validredshifts logic).

Two things deliberately NOT done here (not available in these files)
--------------------------------------------------------------------
  * FRACFLUX_GRZ < 0.35 cut   -- FRACFLUX is NOT in the zcatalog (Tractor-only).
  * Galactic-extinction dered -- MW_TRANSMISSION is NOT in the zcatalog; only
                                 EBV + PHOTSYS are. Magnitudes here are RAW.
Both come for free from a single later cross-match to the Tractor catalogs --
e.g. `desispec.io.photo.gather_tractorphot` (or `gather_targetphot`), whose
output includes FRACFLUX_*, FRACMASKED_*, FRACIN_*, RCHISQ_*, AND
MW_TRANSMISSION_*. To make that a deterministic key join (no positional match,
no re-read here), this catalog carries RELEASE, BRICKID, BRICK_OBJID, BRICKNAME,
PHOTSYS, EBV, and the raw FLUX_*/FLUX_IVAR_* alongside TARGETID/RA/DEC.

Cuts applied (all tunable via CLI)
----------------------------------
  targeting   : (BGS_BRIGHT | BGS_FAINT | LOW_Z_TIER*) over main + SV1/2/3
                (--bgs-bright-only restricts this to BGS_BRIGHT alone)
  redshift    : GOOD_SPEC & (ZWARN_BEST == 0) & (DELTACHI2_BEST > --deltachi2-min)
                (this equals the official GOOD_Z_BGS for BGS objects, and applies
                 the same galaxy-quality definition to LOW_Z secondaries)
  spectype    : SPECTYPE_BEST == 'GALAXY'        (toggle: --no-require-galaxy)
  redshift z  : --zmin < Z_BEST < --zmax         (defaults 0.001 .. 0.2)
  uniqueness  : ZCAT_PRIMARY == True             (toggle: --no-primary-only)

Memory / where to run
---------------------
A full pass over the base file (~30 GB) holds a few GB of columns in RAM. Run on
an interactive/compute node (e.g. `salloc ...`), NOT a NERSC login node.

Examples
--------
  python select_matterhorn_bgs_lowz.py
  python select_matterhorn_bgs_lowz.py --zmax 0.5 --output /pscratch/sd/v/virajvm/mh_bgs_lowz.fits
  python select_matterhorn_bgs_lowz.py --group tilecumulative --no-require-galaxy

Author: Viraj Manwadkar (desi_dwarfs)
"""

import os
import argparse

import numpy as np
import fitsio
from astropy.table import Table

from desiutil.log import get_logger

# Target-selection bitmasks: import by NAME (not hardcoded bit numbers) so that
# any main-vs-SV bit differences are handled automatically.
from desitarget.targetmask import bgs_mask, scnd_mask
from desitarget.sv1.sv1_targetmask import bgs_mask as sv1_bgs_mask, scnd_mask as sv1_scnd_mask
from desitarget.sv2.sv2_targetmask import bgs_mask as sv2_bgs_mask, scnd_mask as sv2_scnd_mask
from desitarget.sv3.sv3_targetmask import bgs_mask as sv3_bgs_mask, scnd_mask as sv3_scnd_mask

# Fiber-status quality bits (for GOOD_SPEC, matching desispec.validredshifts).
from desispec.maskbits import fibermask

log = get_logger()

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_INPUT_DIR = "/global/cfs/cdirs/desi/spectro/redux/matterhorn/zcatalog/v2/zall"
DEFAULT_SPECPROD = "matterhorn"
DEFAULT_OUTPUT_DIR = "/pscratch/sd/v/virajvm/matterhorn"

# (survey_prefix, bgs_mask, scnd_mask). '' == main survey (unprefixed columns).
SURVEY_MASKS = [
    ("", bgs_mask, scnd_mask),
    ("SV1_", sv1_bgs_mask, sv1_scnd_mask),
    ("SV2_", sv2_bgs_mask, sv2_scnd_mask),
    ("SV3_", sv3_bgs_mask, sv3_scnd_mask),
]

# Base-file columns we must have (selection + carried metadata).
BASE_REQUIRED = [
    "TARGETID", "TARGET_RA", "TARGET_DEC",
    "Z_BEST", "ZWARN_BEST", "DELTACHI2_BEST", "SPECTYPE_BEST",
    "ZCAT_PRIMARY", "COADD_FIBERSTATUS", "OBJTYPE",
    "DESI_TARGET", "BGS_TARGET", "SCND_TARGET",
]
# Base-file columns we use if present (SV targeting bits + survey provenance).
BASE_OPTIONAL = [
    "SURVEY", "PROGRAM",
    "SV1_BGS_TARGET", "SV1_SCND_TARGET",
    "SV2_BGS_TARGET", "SV2_SCND_TARGET",
    "SV3_BGS_TARGET", "SV3_SCND_TARGET",
]

# Imaging-file columns.
IMG_REQUIRED = [
    "TARGETID",
    "FLUX_G", "FLUX_R", "FLUX_Z",
    "FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z",
]
# RELEASE/BRICKID/BRICK_OBJID/BRICKNAME are carried so a later Tractor cross-match
# (e.g. desispec.io.photo.gather_tractorphot) is a deterministic key join for
# FRACFLUX + MW_TRANSMISSION, with no need to re-read this catalog.
IMG_OPTIONAL = [
    "RELEASE", "BRICKID", "BRICK_OBJID", "BRICKNAME", "PHOTSYS",
    "EBV", "MASKBITS", "MORPHTYPE",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Select BGS_BRIGHT/BGS_FAINT/LOW_Z targets from the matterhorn zall catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                   help="Directory holding the zall-{group}-{specprod}[-imaging].fits files.")
    p.add_argument("--specprod", default=DEFAULT_SPECPROD,
                   help="Spectro production name in the filenames.")
    p.add_argument("--group", default="pix", choices=["pix", "tilecumulative"],
                   help="Which zall grouping to read.")
    p.add_argument("--output", default=None,
                   help="Output FITS path. Default: "
                        f"{DEFAULT_OUTPUT_DIR}/matterhorn_<group>_bgs_lowz_raw.fits")
    p.add_argument("--zmin", type=float, default=0.001, help="Lower Z_BEST bound (exclusive).")
    p.add_argument("--zmax", type=float, default=0.2, help="Upper Z_BEST bound (exclusive).")
    p.add_argument("--deltachi2-min", type=float, default=40.0,
                   help="DELTACHI2_BEST threshold for a robust redshift.")
    p.add_argument("--require-galaxy", action=argparse.BooleanOptionalAction, default=True,
                   help="Require SPECTYPE_BEST == 'GALAXY'.")
    p.add_argument("--primary-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep only ZCAT_PRIMARY == True (one row per TARGETID).")
    p.add_argument("--bgs-bright-only", action="store_true", default=False,
                   help="Restrict the target sample to BGS_BRIGHT only (drop BGS_FAINT and "
                        "LOW_Z). Handy for a volume-limited massive-galaxy tracer sample; "
                        "e.g. combine with a larger --zmax for an LSS backdrop.")
    return p.parse_args()


def print_followup_banner():
    """Loud reminder of what was deliberately skipped and must come from Tractor."""
    line = "=" * 78
    log.info(line)
    log.info("NOTE: magnitudes are RAW (no Galactic-extinction correction), and the")
    log.info("      FRACFLUX_GRZ < 0.35 cut was NOT applied -- neither EBV-dereddening")
    log.info("      inputs nor FRACFLUX live in the matterhorn zcatalog.")
    log.info("      When you cross-match to the Tractor sweeps / tractorphot VAC, save:")
    log.info("        FRACFLUX_G/R/Z, FRACMASKED_G/R/Z, FRACIN_G/R/Z, RCHISQ_G/R/Z")
    log.info("        MW_TRANSMISSION_G/R/Z   (or deredden via")
    log.info("        desiutil.dust.mwdust_transmission(EBV, band, PHOTSYS,")
    log.info("                                          match_legacy_surveys=True))")
    log.info("      EBV, PHOTSYS, FLUX_*, FLUX_IVAR_* are already carried in the output.")
    log.info(line)


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
    """
    Read `required` + present `optional` columns (optionally only `rows`) from
    the named HDU. Errors if any required column is absent.
    """
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


def build_target_masks(base):
    """Return (is_bgs_bright, is_bgs_faint, is_lowz) boolean arrays over main + SV."""
    n = base.shape[0]
    is_bb = np.zeros(n, dtype=bool)
    is_bf = np.zeros(n, dtype=bool)
    is_lowz = np.zeros(n, dtype=bool)

    for prefix, bmask, smask in SURVEY_MASKS:
        bgs_col = f"{prefix}BGS_TARGET"
        scnd_col = f"{prefix}SCND_TARGET"

        is_bb |= col_bit_match(base, bgs_col, or_bits(bmask, ["BGS_BRIGHT"]))
        is_bf |= col_bit_match(base, bgs_col, or_bits(bmask, ["BGS_FAINT"]))

        lowz_names = [nm for nm in smask.names() if nm.startswith("LOW_Z")]
        is_lowz |= col_bit_match(base, scnd_col, or_bits(smask, lowz_names))
        if lowz_names and scnd_col in base.dtype.names:
            log.info(f"  {prefix or 'MAIN'}: LOW_Z tiers used = {lowz_names}")

    return is_bb, is_bf, is_lowz


def compute_good_spec(base):
    """
    GOOD_SPEC, recomputed from base-file columns exactly as in
    desispec.validredshifts: good fiber hardware status (tolerating only the
    RESTRICTED and VARIABLE bits) AND a science target (OBJTYPE == 'TGT').
    """
    okmask = fibermask.mask("RESTRICTED|VARIABLE")
    good_fiber = (base["COADD_FIBERSTATUS"] & okmask) == base["COADD_FIBERSTATUS"]
    is_tgt = norm_str(base["OBJTYPE"]) == "TGT"
    return good_fiber & is_tgt


def raw_mag(flux):
    """22.5 - 2.5*log10(flux); NaN where flux <= 0. NO extinction correction."""
    flux = np.asarray(flux, dtype=np.float64)
    mag = np.full(flux.shape, np.nan, dtype=np.float64)
    good = flux > 0
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    return mag


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    base_path = os.path.join(args.input_dir, f"zall-{args.group}-{args.specprod}.fits")
    img_path = os.path.join(args.input_dir, f"zall-{args.group}-{args.specprod}-imaging.fits")
    for pth in (base_path, img_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(pth)

    if args.output is None:
        out_path = os.path.join(DEFAULT_OUTPUT_DIR,
                                f"matterhorn_{args.group}_bgs_lowz_raw.fits")
    else:
        out_path = args.output

    print_followup_banner()
    log.info(f"base    : {base_path}")
    log.info(f"imaging : {img_path}")
    log.info(f"output  : {out_path}")

    # ---- 1. Read base columns (one pass) and build the selection mask -------- #
    log.info("Reading base columns (this scans the ~30 GB base file)...")
    base = read_subset(base_path, "ZCATALOG", BASE_REQUIRED, BASE_OPTIONAL)
    n_total = base.shape[0]
    log.info(f"  base rows: {n_total:,}")

    log.info("Building target-selection masks (main + SV1/2/3)...")
    is_bb, is_bf, is_lowz = build_target_masks(base)
    if args.bgs_bright_only:
        is_target = is_bb.copy()
        log.info("  --bgs-bright-only: restricting the sample to BGS_BRIGHT (dropping BGS_FAINT + LOW_Z)")
    else:
        is_target = is_bb | is_bf | is_lowz
    log.info(f"  BGS_BRIGHT={is_bb.sum():,}  BGS_FAINT={is_bf.sum():,}  "
             f"LOW_Z={is_lowz.sum():,}  (selected={is_target.sum():,})")

    # ---- 2. Quality + science cuts ----------------------------------------- #
    good_spec = compute_good_spec(base)
    good_z = good_spec & (base["ZWARN_BEST"] == 0) & (base["DELTACHI2_BEST"] > args.deltachi2_min)

    z = base["Z_BEST"]
    z_in_range = (z > args.zmin) & (z < args.zmax)

    is_galaxy = norm_str(base["SPECTYPE_BEST"]) == "GALAXY"
    is_primary = base["ZCAT_PRIMARY"].astype(bool)

    final = is_target & good_z & z_in_range
    if args.require_galaxy:
        final &= is_galaxy
    if args.primary_only:
        final &= is_primary

    log.info("Cut cascade (cumulative, on target sample):")
    log.info(f"  targeting            : {is_target.sum():,}")
    log.info(f"  + GOOD_SPEC&ZWARN&dX2: {(is_target & good_z).sum():,}")
    log.info(f"  + {args.zmin} < z < {args.zmax}      : {(is_target & good_z & z_in_range).sum():,}")
    if args.require_galaxy:
        log.info(f"  + SPECTYPE=='GALAXY'  : {(is_target & good_z & z_in_range & is_galaxy).sum():,}")
    if args.primary_only:
        log.info(f"  + ZCAT_PRIMARY        : {final.sum():,}")
    log.info(f"FINAL selected: {final.sum():,} / {n_total:,}")

    idx = np.where(final)[0]
    if idx.size == 0:
        log.warning("No objects passed the cuts -- writing an empty catalog.")

    # Subset base in memory; keep the per-row sample flags for the selection.
    base_sel = base[idx]
    sel_bb, sel_bf, sel_lowz = is_bb[idx], is_bf[idx], is_lowz[idx]
    sel_good_spec = good_spec[idx]
    del base, is_bb, is_bf, is_lowz, is_target, good_spec, good_z, is_galaxy, is_primary

    # ---- 3. Read imaging for ONLY the selected rows (row-matched) ----------- #
    log.info(f"Reading imaging columns for {idx.size:,} selected rows...")
    img_sel = read_subset(img_path, "ZCATALOG_IMAGING", IMG_REQUIRED, IMG_OPTIONAL, rows=idx)

    # Safety: confirm the two files really are row-matched on the selection.
    if not np.array_equal(base_sel["TARGETID"], img_sel["TARGETID"]):
        raise RuntimeError("TARGETID mismatch between base and imaging on the selected rows -- "
                           "the files are not row-matched as expected; aborting.")

    # ---- 4. Assemble output ------------------------------------------------- #
    out = Table()
    out["TARGETID"] = base_sel["TARGETID"]
    out["TARGET_RA"] = base_sel["TARGET_RA"]
    out["TARGET_DEC"] = base_sel["TARGET_DEC"]
    for c in ("SURVEY", "PROGRAM"):
        if c in base_sel.dtype.names:
            out[c] = norm_str(base_sel[c])

    # Per-sample membership flags (an object can belong to more than one).
    out["IS_BGS_BRIGHT"] = sel_bb
    out["IS_BGS_FAINT"] = sel_bf
    out["IS_LOWZ"] = sel_lowz

    # Carry the raw targeting-bit columns that exist.
    for c in ("DESI_TARGET", "BGS_TARGET", "SCND_TARGET",
              "SV1_BGS_TARGET", "SV1_SCND_TARGET",
              "SV2_BGS_TARGET", "SV2_SCND_TARGET",
              "SV3_BGS_TARGET", "SV3_SCND_TARGET"):
        if c in base_sel.dtype.names:
            out[c] = base_sel[c]

    # Redshift + quality.
    out["Z_BEST"] = base_sel["Z_BEST"]
    out["ZWARN_BEST"] = base_sel["ZWARN_BEST"]
    out["DELTACHI2_BEST"] = base_sel["DELTACHI2_BEST"]
    out["SPECTYPE_BEST"] = norm_str(base_sel["SPECTYPE_BEST"])
    out["ZCAT_PRIMARY"] = base_sel["ZCAT_PRIMARY"].astype(bool)
    out["COADD_FIBERSTATUS"] = base_sel["COADD_FIBERSTATUS"]
    out["GOOD_SPEC"] = sel_good_spec

    # Photometry: raw fluxes + RAW magnitudes (no dereddening).
    for b in ("G", "R", "Z"):
        out[f"FLUX_{b}"] = img_sel[f"FLUX_{b}"]
        out[f"FLUX_IVAR_{b}"] = img_sel[f"FLUX_IVAR_{b}"]
    out["MAG_G"] = raw_mag(img_sel["FLUX_G"])
    out["MAG_R"] = raw_mag(img_sel["FLUX_R"])
    out["MAG_Z"] = raw_mag(img_sel["FLUX_Z"])

    for c in ("RELEASE", "BRICKID", "BRICK_OBJID", "BRICKNAME", "PHOTSYS",
              "EBV", "MASKBITS", "MORPHTYPE"):
        if c in img_sel.dtype.names:
            out[c] = norm_str(img_sel[c]) if img_sel[c].dtype.kind in ("S", "U") else img_sel[c]

    # ---- 5. Provenance header + write -------------------------------------- #
    out.meta["EXTNAME"] = "BGS_LOWZ"
    out.meta["SPECPROD"] = args.specprod
    out.meta["ZGROUP"] = args.group
    out.meta["BASEFILE"] = os.path.basename(base_path)
    out.meta["IMGFILE"] = os.path.basename(img_path)
    out.meta["NIN"] = n_total
    out.meta["NSEL"] = int(final.sum())
    out.meta["ZMIN"] = args.zmin
    out.meta["ZMAX"] = args.zmax
    out.meta["DCHI2MIN"] = args.deltachi2_min
    out.meta["REQGAL"] = args.require_galaxy
    out.meta["PRIMONLY"] = args.primary_only
    out.meta["BBONLY"] = args.bgs_bright_only
    out.meta["COMMENT"] = (
        "BGS_BRIGHT/BGS_FAINT/LOW_Z selection from matterhorn zall (main+SV1/2/3). "
        "MAG_* are RAW (not dereddened); FRACFLUX cut NOT applied -- both require a "
        "Tractor/tractorphot cross-match (see EBV, PHOTSYS for dereddening)."
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.write(out_path, format="fits", overwrite=True)
    log.info(f"Wrote {len(out):,} rows -> {out_path}")
    print_followup_banner()


if __name__ == "__main__":
    main()
