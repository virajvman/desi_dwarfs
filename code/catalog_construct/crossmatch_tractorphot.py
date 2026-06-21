#!/usr/bin/env python
"""
crossmatch_tractorphot.py
=========================

Stage 2 of the matterhorn BGS/LOW_Z catalog build. Takes the output of
`select_matterhorn_bgs_lowz.py`, cross-matches it to the Legacy Surveys Tractor
catalogs to recover the columns the zcatalog does NOT carry, applies the
FRACFLUX cut, and adds Galactic-extinction-corrected magnitudes.

Why this step exists
--------------------
The matterhorn zcatalog has FLUX_G/R/Z + EBV but NOT FRACFLUX (a Tractor-only
contamination diagnostic) and NOT MW_TRANSMISSION. Both come for free from the
Tractor catalogs. There is no matterhorn `tractorphot` VAC, so we gather the
photometry on the fly with `desispec.io.photo.gather_tractorphot`, which:
  * matches deterministically on RELEASE + BRICKID + BRICK_OBJID (carried by the
    stage-1 catalog), falling back to a 1" positional match, and
  * returns the full Tractor row, incl. FRACFLUX_*, FRACMASKED_*, FRACIN_*,
    RCHISQ_*, NOBS_*, and MW_TRANSMISSION_*.

Cost scales with the number of UNIQUE BRICKS the sample touches (one Tractor
file read per brick), so this script parallelizes over bricks with
multiprocessing (`--nproc`) and prints the unique-brick count up front.

What it produces
----------------
Adds to the stage-1 catalog: FRACFLUX_{G,R,Z}, FRACMASKED_{G,R,Z},
FRACIN_{G,R,Z}, RCHISQ_{G,R,Z}, NOBS_{G,R,Z}, MW_TRANSMISSION_{G,R,Z},
dereddened MAG_{G,R,Z}_DERED, and the booleans TRACTORPHOT_MATCH / FRACFLUX_PASS.
By default it then WRITES ONLY rows that matched AND pass the FRACFLUX cut
(use --keep-all to write every row with the flags instead).

FRACFLUX cut (this build): keep only objects with FRACFLUX_G < f AND
FRACFLUX_R < f AND FRACFLUX_Z < f (all three bands; f = --fracflux-max, def 0.35).
Unmatched objects get FRACFLUX = NaN and therefore fail the cut.

Dereddening: MAG_X_DERED = 22.5 - 2.5*log10(FLUX_X / MW_TRANSMISSION_X), using
the Tractor MW_TRANSMISSION (identical to what the iron pipeline divided by).

Examples
--------
  python crossmatch_tractorphot.py                       # defaults (pix, nproc 128)
  python crossmatch_tractorphot.py --nproc 1             # serial (small samples)
  python crossmatch_tractorphot.py --keep-all            # don't drop; just flag
  python crossmatch_tractorphot.py --legacysurveydir /global/cfs/cdirs/desi/external/legacysurvey/dr10

Author: Viraj Manwadkar (desi_dwarfs)
"""

import os
import argparse
from functools import partial
from multiprocessing import Pool

import numpy as np
from astropy.table import Table, vstack

from desiutil.log import get_logger
from desiutil.brick import brickname as radec_to_brickname
from desispec.io.photo import gather_tractorphot

log = get_logger()

DEFAULT_INPUT = "/pscratch/sd/v/virajvm/matterhorn/matterhorn_pix_bgs_lowz_raw.fits"
DEFAULT_OUTPUT = "/pscratch/sd/v/virajvm/matterhorn/matterhorn_pix_bgs_lowz_clean.fits"

# Tractor columns to carry into the final catalog (besides the match keys).
FRAC_BANDS = ("G", "R", "Z")
TRACTOR_FLOAT_COLS = (
    [f"FRACFLUX_{b}" for b in FRAC_BANDS]
    + [f"FRACMASKED_{b}" for b in FRAC_BANDS]
    + [f"FRACIN_{b}" for b in FRAC_BANDS]
    + [f"RCHISQ_{b}" for b in FRAC_BANDS]
    + [f"MW_TRANSMISSION_{b}" for b in FRAC_BANDS]
)
TRACTOR_INT_COLS = [f"NOBS_{b}" for b in FRAC_BANDS]

# Columns gather_tractorphot wants for an exact (non-positional) match.
MATCH_COLS = ["TARGETID", "TARGET_RA", "TARGET_DEC",
              "RELEASE", "BRICKID", "BRICK_OBJID", "BRICKNAME", "PHOTSYS"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-match the matterhorn BGS/LOW_Z selection to Tractor photometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="Stage-1 selection FITS.")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output cleaned FITS.")
    p.add_argument("--legacysurveydir", default=None,
                   help="Legacy Surveys Tractor dir. Default: "
                        "$DESI_ROOT/external/legacysurvey/dr9 (resolved by desispec).")
    p.add_argument("--nproc", type=int, default=128,
                   help="Worker processes (parallelize over bricks). 1 = serial.")
    p.add_argument("--fracflux-max", type=float, default=0.35,
                   help="Keep objects with FRACFLUX < this in ALL of g,r,z.")
    p.add_argument("--keep-all", action="store_true",
                   help="Write every row with flag columns instead of dropping "
                        "unmatched / FRACFLUX-failing rows.")
    return p.parse_args()


def normalize_str_col(tab, col):
    """Ensure a character column is unicode (FITS round-trips can yield bytes)."""
    if col in tab.colnames and tab[col].dtype.kind == "S":
        tab[col] = np.char.decode(np.asarray(tab[col]), "utf-8")


def filled_bricknames(cat):
    """Brick name per row, computing blanks from RA/Dec (for the brick split)."""
    n = len(cat)
    if "BRICKNAME" in cat.colnames:
        bn = np.asarray(cat["BRICKNAME"]).astype(str)
        bn = np.char.strip(bn)
    else:
        bn = np.full(n, "", dtype="<U8")
    blank = bn == ""
    if blank.any():
        bn = bn.astype("<U8")
        bn[blank] = radec_to_brickname(np.asarray(cat["TARGET_RA"])[blank],
                                       np.asarray(cat["TARGET_DEC"])[blank])
    return bn


def split_indices_by_brick(bricknames, nproc):
    """Partition row indices into <=nproc groups, each holding whole bricks."""
    uniq = np.unique(bricknames)
    brick_bin = {b: i % nproc for i, b in enumerate(uniq)}
    binid = np.fromiter((brick_bin[b] for b in bricknames), dtype=np.int64,
                        count=len(bricknames))
    groups = [np.where(binid == k)[0] for k in range(nproc)]
    return [g for g in groups if g.size > 0], len(uniq)


def _gather_worker(input_sub, legacysurveydir):
    """Top-level so it is picklable; returns Tractor photometry for input_sub."""
    return gather_tractorphot(input_sub, legacysurveydir=legacysurveydir, verbose=False)


def gather_all(input_tab, legacysurveydir, nproc):
    """Run gather_tractorphot over the whole input, parallelized over bricks.

    Returns a photometry Table row-aligned to `input_tab`.
    """
    if nproc <= 1 or len(input_tab) < 2:
        phot = gather_tractorphot(input_tab, legacysurveydir=legacysurveydir, verbose=False)
        return phot

    bn = filled_bricknames(input_tab)
    groups, _ = split_indices_by_brick(bn, nproc)
    sub_tabs = [input_tab[idx] for idx in groups]

    with Pool(processes=min(nproc, len(sub_tabs))) as pool:
        results = pool.map(partial(_gather_worker, legacysurveydir=legacysurveydir), sub_tabs)

    phot_concat = vstack(results, join_type="exact")
    order = np.concatenate(groups)              # cat-row index for each phot_concat row
    phot = phot_concat[np.argsort(order)]       # back to input_tab order
    return phot


def raw_mag(flux):
    flux = np.asarray(flux, dtype=np.float64)
    mag = np.full(flux.shape, np.nan)
    good = flux > 0
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    return mag


def main():
    args = parse_args()
    log.info(f"input  : {args.input}")
    log.info(f"output : {args.output}")

    cat = Table.read(args.input)
    n_in = len(cat)
    log.info(f"Loaded {n_in:,} selected objects.")
    if n_in == 0:
        raise SystemExit("Empty input catalog; nothing to do.")

    for c in ("BRICKNAME", "PHOTSYS"):
        normalize_str_col(cat, c)

    # Build the (copied) match input for gather_tractorphot.
    have = [c for c in MATCH_COLS if c in cat.colnames]
    for req in ("TARGETID", "TARGET_RA", "TARGET_DEC"):
        if req not in have:
            raise KeyError(f"Input catalog is missing required column {req}.")
    input_tab = cat[have].copy()

    bn = filled_bricknames(cat)
    n_bricks = np.unique(bn).size
    log.info(f"Sample touches {n_bricks:,} unique Legacy Surveys bricks "
             f"(one Tractor file read each); nproc={args.nproc}.")

    # ---- gather Tractor photometry ----------------------------------------- #
    phot = gather_all(input_tab, args.legacysurveydir, args.nproc)
    if not np.array_equal(np.asarray(phot["TARGETID"]), np.asarray(cat["TARGETID"])):
        raise RuntimeError("Tractor photometry is not row-aligned to the input; aborting.")

    # Matched = a real DR9/DR10 Tractor source was found.
    matched = np.asarray(phot["RELEASE"]) > 0
    log.info(f"Tractor matches: {matched.sum():,} / {n_in:,} "
             f"({100*matched.sum()/n_in:.1f}%); unmatched: {(~matched).sum():,}")

    # ---- attach Tractor columns (NaN/-1 where unmatched) ------------------- #
    for col in TRACTOR_FLOAT_COLS:
        vals = np.asarray(phot[col], dtype=np.float64).copy()
        vals[~matched] = np.nan
        cat[col] = vals
    for col in TRACTOR_INT_COLS:
        vals = np.asarray(phot[col]).astype(np.int32).copy()
        vals[~matched] = -1
        cat[col] = vals

    # ---- dereddened magnitudes (Tractor MW_TRANSMISSION) ------------------- #
    for b in FRAC_BANDS:
        flux = np.asarray(cat[f"FLUX_{b}"], dtype=np.float64)
        mwt = np.asarray(cat[f"MW_TRANSMISSION_{b}"], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            dered_flux = np.where(mwt > 0, flux / mwt, np.nan)
        cat[f"MAG_{b}_DERED"] = raw_mag(dered_flux)

    # ---- FRACFLUX cut: all three bands < fmax (NaN -> fails) --------------- #
    fmax = args.fracflux_max
    with np.errstate(invalid="ignore"):
        fracflux_pass = (
            (np.asarray(cat["FRACFLUX_G"]) < fmax)
            & (np.asarray(cat["FRACFLUX_R"]) < fmax)
            & (np.asarray(cat["FRACFLUX_Z"]) < fmax)
        )
    cat["TRACTORPHOT_MATCH"] = matched
    cat["FRACFLUX_PASS"] = fracflux_pass
    log.info(f"FRACFLUX (all g,r,z < {fmax}) pass: {fracflux_pass.sum():,}; "
             f"matched & pass: {(matched & fracflux_pass).sum():,}")

    # ---- write ------------------------------------------------------------- #
    if args.keep_all:
        out = cat
        log.info(f"--keep-all: writing all {len(out):,} rows (with flag columns).")
    else:
        keep = matched & fracflux_pass
        out = cat[keep]
        log.info(f"Writing {len(out):,} cleaned rows "
                 f"(dropped {(~keep).sum():,}: unmatched or FRACFLUX-failing).")

    out.meta["EXTNAME"] = "BGS_LOWZ_CLEAN"
    out.meta["NIN"] = n_in
    out.meta["NMATCH"] = int(matched.sum())
    out.meta["NOUT"] = len(out)
    out.meta["FFLUXMAX"] = fmax
    out.meta["FFLUXDEF"] = "all g,r,z < FFLUXMAX"
    out.meta["LSDIR"] = args.legacysurveydir or "default:dr9"
    out.meta["COMMENT"] = (
        "Stage-2 Tractor cross-match of the matterhorn BGS/LOW_Z selection. "
        "FRACFLUX/MW_TRANSMISSION from gather_tractorphot; MAG_*_DERED dereddened "
        "with the Tractor MW_TRANSMISSION."
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.write(args.output, format="fits", overwrite=True)
    log.info(f"Wrote {len(out):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
