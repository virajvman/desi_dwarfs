#!/usr/bin/env python
"""
crossmatch_tractorphot.py
=========================

Stage 2 of the matterhorn BGS/LOW_Z catalog build. Takes the output of
`select_matterhorn_bgs_lowz.py`, attaches the photometric columns the zcatalog
does NOT carry (FRACFLUX, MW_TRANSMISSION, ...), applies the FRACFLUX cut, and
adds Galactic-extinction-corrected magnitudes. Writes TWO catalogs:

  *_phot.fits   ALL z<0.2 selected objects with photometry + flags (no cut)
  *_clean.fits  the subset that is matched AND passes the FRACFLUX cut

Two photometry sources, by target class
---------------------------------------
* BGS objects (BGS_BRIGHT or BGS_FAINT, including objects that are ALSO LOW_Z):
  `desispec.io.photo.gather_tractorphot` against the Legacy Surveys Tractor
  catalogs (matched on RELEASE+BRICKID+BRICK_OBJID, 1" positional fallback).

* LOW_Z-only objects (IS_LOWZ and NOT BGS): the LOW_Z target selection was done
  separately, so their photometry comes from Elise's DR9 LOW_Z target catalogs
  (north + south), combined with the exact footprint logic from
  construct_dwarf_galaxy_catalogs.py and matched positionally at 1". ALL
  photometry (FLUX/FLUX_IVAR/FIBERFLUX/MW_TRANSMISSION/FRACFLUX/...) is taken
  from Elise for these objects. LOW_Z-only objects that miss Elise fall back to
  gather_tractorphot (recorded as PHOT_SOURCE='tractorphot_fallback').

Provenance: PHOT_SOURCE in {'tractorphot','lowz_target','tractorphot_fallback',
'none'}; PHOT_MATCH = (PHOT_SOURCE != 'none').

FRACFLUX cut (this build): keep only objects with FRACFLUX_G < f AND
FRACFLUX_R < f AND FRACFLUX_Z < f (all three; f = --fracflux-max, def 0.35).
Unmatched objects get FRACFLUX=NaN and therefore fail.

Dereddening: MAG_X_DERED = 22.5 - 2.5*log10(FLUX_X / MW_TRANSMISSION_X).
Raw MAG_X are recomputed from the final (source) FLUX so mags and FLUX agree.

Cost scales with the number of UNIQUE BRICKS the BGS/fallback set touches (one
Tractor file read each); parallelized over bricks with `--nproc`, with progress.

Examples
--------
  python crossmatch_tractorphot.py
  python crossmatch_tractorphot.py --nproc 1
  python crossmatch_tractorphot.py --output /path/mh_clean.fits --phot-output /path/mh_phot.fits

Author: Viraj Manwadkar (desi_dwarfs)
"""

import os
import time
import argparse
from functools import partial
from multiprocessing import Pool

import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

from desiutil.log import get_logger
from desiutil.brick import brickname as radec_to_brickname
from desispec.io.photo import gather_tractorphot

log = get_logger()

DEFAULT_INPUT = "/pscratch/sd/v/virajvm/matterhorn/matterhorn_pix_bgs_lowz_raw.fits"
DEFAULT_CLEAN = "/pscratch/sd/v/virajvm/matterhorn/matterhorn_pix_bgs_lowz_clean.fits"
DEFAULT_NORTH = "/pscratch/sd/v/virajvm/target/dr9_north_lowz_targets_no_rfib_cut.fits"
DEFAULT_SOUTH = "/pscratch/sd/v/virajvm/target/dr9_south_lowz_targets_no_rfib_cut_dec20.fits"

BANDS = ("G", "R", "Z")
# Photometry columns already present in the stage-1 catalog (from the imaging
# file): kept for unmatched rows, overwritten by the matched source otherwise.
EXISTING_FLOAT = [f"FLUX_{b}" for b in BANDS] + [f"FLUX_IVAR_{b}" for b in BANDS]
EXISTING_INT = ["MASKBITS", "RELEASE", "BRICKID", "BRICK_OBJID"]
# New columns supplied by the photometry match (NaN / -1 where unmatched).
NEW_FLOAT = (
    [f"FIBERFLUX_{b}" for b in BANDS]
    + [f"MW_TRANSMISSION_{b}" for b in BANDS]
    + [f"FRACFLUX_{b}" for b in BANDS]
    + [f"FRACMASKED_{b}" for b in BANDS]
    + [f"FRACIN_{b}" for b in BANDS]
    + [f"RCHISQ_{b}" for b in BANDS]
)
NEW_INT = [f"NOBS_{b}" for b in BANDS]
WANT_COLS = EXISTING_FLOAT + EXISTING_INT + NEW_FLOAT + NEW_INT

# Columns gather_tractorphot wants for an exact (non-positional) match.
MATCH_COLS = ["TARGETID", "TARGET_RA", "TARGET_DEC",
              "RELEASE", "BRICKID", "BRICK_OBJID", "BRICKNAME", "PHOTSYS"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Attach Tractor / LOW_Z-target photometry to the matterhorn selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="Stage-1 selection FITS.")
    p.add_argument("--output", default=DEFAULT_CLEAN, help="Cleaned (FRACFLUX-cut) FITS.")
    p.add_argument("--phot-output", default=None,
                   help="Full (no-cut) photometry FITS. Default: --output with "
                        "'_clean'->'_phot' (or '_phot' appended).")
    p.add_argument("--north-targets", default=DEFAULT_NORTH, help="Elise DR9 north LOW_Z targets.")
    p.add_argument("--south-targets", default=DEFAULT_SOUTH, help="Elise DR9 south LOW_Z targets.")
    p.add_argument("--legacysurveydir", default=None,
                   help="Legacy Surveys Tractor dir. Default: "
                        "$DESI_ROOT/external/legacysurvey/dr9 (resolved by desispec).")
    p.add_argument("--nproc", type=int, default=128,
                   help="Worker processes for the Tractor gather. 1 = serial.")
    p.add_argument("--fracflux-max", type=float, default=0.35,
                   help="Keep objects with FRACFLUX < this in ALL of g,r,z.")
    return p.parse_args()


def derive_phot_output(clean_path):
    base, ext = os.path.splitext(clean_path)
    if "_clean" in base:
        return base.replace("_clean", "_phot") + ext
    return base + "_phot" + ext


# --------------------------------------------------------------------------- #
# Tractor gather (BGS + LOW_Z fallback), parallelized over bricks
# --------------------------------------------------------------------------- #
def filled_bricknames(input_tab):
    n = len(input_tab)
    if "BRICKNAME" in input_tab.colnames:
        bn = np.char.strip(np.asarray(input_tab["BRICKNAME"]).astype(str))
    else:
        bn = np.full(n, "", dtype="<U8")
    blank = bn == ""
    if blank.any():
        bn = bn.astype("<U8")
        bn[blank] = radec_to_brickname(np.asarray(input_tab["TARGET_RA"])[blank],
                                       np.asarray(input_tab["TARGET_DEC"])[blank])
    return bn


def make_brick_batches(input_tab, nproc):
    """Group rows into ~8x nproc batches of whole bricks (for progress + parallelism)."""
    bn = filled_bricknames(input_tab)
    uniq_bricks = np.unique(bn)
    n_bricks = uniq_bricks.size
    n_batches = max(1, min(n_bricks, nproc * 8))
    brick_to_batch = {b: i for i, arr in enumerate(np.array_split(uniq_bricks, n_batches))
                      for b in arr}
    batch_id = np.fromiter((brick_to_batch[b] for b in bn), dtype=np.int64, count=len(bn))
    batches = []
    for i in range(n_batches):
        idx = np.where(batch_id == i)[0]
        if idx.size:
            batches.append((idx, input_tab[idx]))
    return batches, n_bricks


def _gather_worker(batch, legacysurveydir):
    rowidx, sub_tab = batch
    return rowidx, gather_tractorphot(sub_tab, legacysurveydir=legacysurveydir, verbose=False)


def gather_all(input_tab, legacysurveydir, nproc, label=""):
    """gather_tractorphot over input_tab, batched over bricks, with progress/ETA."""
    batches, n_bricks = make_brick_batches(input_tab, nproc)
    n_batches = len(batches)
    log.info(f"[{label}] Gathering {len(input_tab):,} sources across {n_bricks:,} bricks "
             f"in {n_batches} batches on {nproc} process(es)...")
    worker = partial(_gather_worker, legacysurveydir=legacysurveydir)
    log_every = max(1, n_batches // 50)
    t0 = time.monotonic()
    idx_parts, phot_parts = [], []

    def _record(done, rowidx, phot_sub):
        idx_parts.append(rowidx)
        phot_parts.append(phot_sub)
        if done % log_every == 0 or done == n_batches:
            el = time.monotonic() - t0
            frac = done / n_batches
            eta = el * (1 - frac) / frac if frac > 0 else 0.0
            log.info(f"  [{label}] progress: {done}/{n_batches} batches ({100*frac:.1f}%), "
                     f"elapsed {el/60:.1f} min, ETA ~{eta/60:.1f} min")

    if nproc <= 1:
        for done, batch in enumerate(batches, start=1):
            rowidx, phot_sub = worker(batch)
            _record(done, rowidx, phot_sub)
    else:
        with Pool(processes=min(nproc, n_batches)) as pool:
            for done, (rowidx, phot_sub) in enumerate(pool.imap_unordered(worker, batches), start=1):
                _record(done, rowidx, phot_sub)

    log.info(f"  [{label}] gather finished in {(time.monotonic() - t0)/60:.1f} min.")
    phot_concat = vstack(phot_parts, join_type="exact")
    order = np.concatenate(idx_parts)
    return phot_concat[np.argsort(order)]


def _photsys_str(input_tab):
    n = len(input_tab)
    if "PHOTSYS" in input_tab.colnames:
        return np.char.strip(np.asarray(input_tab["PHOTSYS"]).astype(str))
    return np.full(n, "", dtype="<U1")


def _gather_unique(uniq_tab, legacysurveydir, nproc, label):
    """Gather photometry for a set of UNIQUE sources, SPLIT BY PHOTSYS region.

    gather_tractorphot asserts a single PHOTSYS per brick (desispec.io.photo
    line ~978), but an N/S-overlap brick can legitimately hold both north and
    south sources. Splitting the gather by PHOTSYS guarantees every brick passed
    to gather_tractorphot is single-region. Returns photometry aligned to uniq_tab.
    """
    ps = _photsys_str(uniq_tab)
    parts_idx, parts_phot = [], []
    for region in np.unique(ps):
        sel = np.where(ps == region)[0]
        tag = region if region else "pos"
        parts_phot.append(gather_all(uniq_tab[sel], legacysurveydir, nproc,
                                     label=f"{label}:{tag}"))
        parts_idx.append(sel)
    phot_concat = vstack(parts_phot, join_type="exact")
    order = np.concatenate(parts_idx)
    return phot_concat[np.argsort(order)]


def gather_dedup(input_tab, legacysurveydir, nproc, label=""):
    """Gather Tractor photometry, deduplicating by photometric SOURCE first.

    Several DESI TARGETIDs (main + SV, primary + secondary) point at the same
    Legacy Surveys source. Inside a brick gather_tractorphot matches purely on
    BRICK_OBJID and assumes one PHOTSYS, so the identity it actually uses is
    (PHOTSYS, BRICKID, BRICK_OBJID) -- NOT RELEASE, which can differ spuriously
    between target lists for the same physical source and would otherwise leave
    duplicate BRICK_OBJID within a brick. We dedup on that key, gather once per
    unique source (split by PHOTSYS region), and broadcast back. Returns
    photometry row-aligned to input_tab.
    """
    n = len(input_tab)

    def _icol(name):
        return (np.asarray(input_tab[name], dtype=np.int64)
                if name in input_tab.colnames else np.zeros(n, dtype=np.int64))

    brickid, objid = _icol("BRICKID"), _icol("BRICK_OBJID")
    ps = _photsys_str(input_tab)
    valid = (brickid > 0) & (objid > 0) & (ps != "")

    # Robust string key (no bit-packing overflow). Invalid rows (positional
    # matching) each get a unique key so they are never collapsed together.
    keys = np.char.add(np.char.add(np.char.add(ps, "|"),
                                   np.char.add(brickid.astype("U12"), "|")),
                       objid.astype("U12"))
    inv = ~valid
    keys[inv] = np.char.add("u", np.arange(n)[inv].astype("U12"))

    uniq, uniq_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)
    if uniq.size < n:
        log.info(f"[{label}] {n:,} rows -> {uniq.size:,} unique photometric sources "
                 f"({n - uniq.size:,} TARGETIDs share a source).")
    phot_uniq = _gather_unique(input_tab[uniq_idx], legacysurveydir, nproc, label)
    return phot_uniq[inverse]


# --------------------------------------------------------------------------- #
# LOW_Z-only photometry from Elise's DR9 LOW_Z target catalogs
# --------------------------------------------------------------------------- #
def _galactic_b(ra, dec):
    return SkyCoord(ra=np.asarray(ra) * u.degree,
                    dec=np.asarray(dec) * u.degree, frame="icrs").galactic.b.value


def remove_south_lowz(data):
    """North target catalog footprint (mirrors construct_dwarf_galaxy_catalogs.py)."""
    b = _galactic_b(data["RA"], data["DEC"])
    dec = np.asarray(data["DEC"])
    return data[(b > 0) & (dec > 32.375) & (np.abs(b) > 15.0)]


def clean_south_lowz(data):
    """South target catalog footprint (mirrors construct_dwarf_galaxy_catalogs.py).

    NB: faithfully reproduces the original, where the |b|>15 line is overwritten,
    so the effective selection is (DEC < 32.375) | (b < 0).
    """
    b = _galactic_b(data["RA"], data["DEC"])
    dec = np.asarray(data["DEC"])
    return data[(dec < 32.375) | (b < 0)]


def _match_sky(ra1, dec1, ra2, dec2):
    c1 = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree)
    c2 = SkyCoord(ra=ra2 * u.degree, dec=dec2 * u.degree)
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    return idx, d2d.arcsec


def match_lowz_targets(sub, north_path, south_path):
    """Positionally match LOW_Z-only objects (1") to Elise's combined north+south
    target catalog, with a south-only fallback for the overlap region (mirrors
    construct_dwarf_galaxy_catalogs.get_lowz_catalogs).

    Returns (src_table aligned to `sub`, matched bool array).
    """
    log.info(f"[lowz] reading Elise LOW_Z target catalogs:\n  {north_path}\n  {south_path}")
    north = Table.read(north_path)
    south = Table.read(south_path)
    total = vstack([remove_south_lowz(north), clean_south_lowz(south)])

    keep = ["RA", "DEC"] + [c for c in WANT_COLS if c in total.colnames and c in south.colnames]
    total = total[keep]
    south = south[keep]

    ra = np.asarray(sub["TARGET_RA"], dtype=float)
    dec = np.asarray(sub["TARGET_DEC"], dtype=float)

    idx, sep = _match_sky(ra, dec, np.asarray(total["RA"], float), np.asarray(total["DEC"], float))
    matched = sep <= 1.0
    src = total[idx]                                  # aligned to sub
    log.info(f"[lowz] {matched.sum():,}/{len(sub):,} matched to combined catalog at <=1\".")

    nm = np.where(~matched)[0]
    if nm.size:
        idx2, sep2 = _match_sky(ra[nm], dec[nm],
                                np.asarray(south["RA"], float), np.asarray(south["DEC"], float))
        ok2 = sep2 <= 1.0
        if ok2.any():
            src[nm[ok2]] = south[idx2[ok2]]
            matched[nm[ok2]] = True
            log.info(f"[lowz] south-fallback recovered {ok2.sum():,} more matches.")
    return src, matched


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def raw_mag(flux):
    flux = np.asarray(flux, dtype=np.float64)
    mag = np.full(flux.shape, np.nan)
    good = flux > 0
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    return mag


def fill_from_source(phot, global_idx, src, matched):
    """Write src's WANT_COLS into the phot dict at global_idx[matched]."""
    sel = global_idx[matched]
    for c in EXISTING_FLOAT + NEW_FLOAT:
        if c in src.colnames:
            phot[c][sel] = np.asarray(src[c], dtype=np.float64)[matched]
    for c in EXISTING_INT + NEW_INT:
        if c in src.colnames:
            phot[c][sel] = np.asarray(src[c])[matched].astype(np.int64)


def main():
    args = parse_args()
    phot_output = args.phot_output or derive_phot_output(args.output)
    log.info(f"input       : {args.input}")
    log.info(f"phot output : {phot_output}")
    log.info(f"clean output: {args.output}")

    cat = Table.read(args.input)
    n = len(cat)
    log.info(f"Loaded {n:,} selected objects.")
    if n == 0:
        raise SystemExit("Empty input catalog; nothing to do.")

    for c in ("BRICKNAME", "PHOTSYS"):
        if c in cat.colnames and cat[c].dtype.kind == "S":
            cat[c] = np.char.decode(np.asarray(cat[c]), "utf-8")

    for req in ("TARGETID", "TARGET_RA", "TARGET_DEC",
                "IS_BGS_BRIGHT", "IS_BGS_FAINT", "IS_LOWZ"):
        if req not in cat.colnames:
            raise KeyError(f"Input catalog is missing required column {req}.")
    input_tab = cat[[c for c in MATCH_COLS if c in cat.colnames]].copy()

    # ---- initialize the unified photometry arrays -------------------------- #
    phot = {}
    for c in EXISTING_FLOAT:
        phot[c] = (np.asarray(cat[c], dtype=np.float64) if c in cat.colnames
                   else np.full(n, np.nan))
    for c in NEW_FLOAT:
        phot[c] = np.full(n, np.nan)
    for c in EXISTING_INT:
        phot[c] = (np.asarray(cat[c]).astype(np.int64) if c in cat.colnames
                   else np.full(n, -1, np.int64))
    for c in NEW_INT:
        phot[c] = np.full(n, -1, np.int64)
    phot_source = np.full(n, "none", dtype="<U24")

    # ---- partition --------------------------------------------------------- #
    is_bgs = np.asarray(cat["IS_BGS_BRIGHT"], bool) | np.asarray(cat["IS_BGS_FAINT"], bool)
    lowz_only = np.asarray(cat["IS_LOWZ"], bool) & ~is_bgs
    bgs_idx = np.where(is_bgs)[0]
    lowz_idx = np.where(lowz_only)[0]
    log.info(f"Partition: BGS (gather_tractorphot) = {bgs_idx.size:,}; "
             f"LOW_Z-only (Elise) = {lowz_idx.size:,}.")

    # ---- 1) BGS via gather_tractorphot ------------------------------------- #
    if bgs_idx.size:
        src = gather_dedup(input_tab[bgs_idx], args.legacysurveydir, args.nproc, label="bgs")
        m = np.asarray(src["RELEASE"]) > 0
        fill_from_source(phot, bgs_idx, src, m)
        phot_source[bgs_idx[m]] = "tractorphot"
        log.info(f"[bgs] matched {m.sum():,}/{bgs_idx.size:,}.")

    # ---- 2) LOW_Z-only via Elise target catalogs (+ tractorphot fallback) -- #
    if lowz_idx.size:
        src, m = match_lowz_targets(cat[lowz_idx], args.north_targets, args.south_targets)
        fill_from_source(phot, lowz_idx, src, m)
        phot_source[lowz_idx[m]] = "lowz_target"

        miss = lowz_idx[~m]
        if miss.size:
            log.info(f"[lowz] {miss.size:,} LOW_Z-only objects missed Elise; "
                     f"falling back to gather_tractorphot.")
            src2 = gather_dedup(input_tab[miss],
                                args.legacysurveydir, args.nproc, label="lowz-fallback")
            m2 = np.asarray(src2["RELEASE"]) > 0
            fill_from_source(phot, miss, src2, m2)
            phot_source[miss[m2]] = "tractorphot_fallback"
            log.info(f"[lowz-fallback] recovered {m2.sum():,}/{miss.size:,}.")

    # ---- derived quantities ------------------------------------------------ #
    matched = phot_source != "none"
    rel = phot["RELEASE"].astype(np.int64)
    bid = phot["BRICKID"].astype(np.int64)
    oid = phot["BRICK_OBJID"].astype(np.int64)
    ls_id = np.where(rel > 0, (rel << 40) | (bid << 16) | oid, 0)

    for b in BANDS:
        cat[f"FLUX_{b}"] = phot[f"FLUX_{b}"]
        cat[f"FLUX_IVAR_{b}"] = phot[f"FLUX_IVAR_{b}"]
        cat[f"MAG_{b}"] = raw_mag(phot[f"FLUX_{b}"])
        with np.errstate(invalid="ignore", divide="ignore"):
            dered = np.where(phot[f"MW_TRANSMISSION_{b}"] > 0,
                             phot[f"FLUX_{b}"] / phot[f"MW_TRANSMISSION_{b}"], np.nan)
        cat[f"MAG_{b}_DERED"] = raw_mag(dered)
    for c in NEW_FLOAT + NEW_INT + ["MASKBITS", "RELEASE", "BRICKID", "BRICK_OBJID"]:
        cat[c] = phot[c]
    cat["LS_ID"] = ls_id
    cat["PHOT_SOURCE"] = phot_source
    cat["PHOT_MATCH"] = matched

    fmax = args.fracflux_max
    with np.errstate(invalid="ignore"):
        fracflux_pass = ((phot["FRACFLUX_G"] < fmax) & (phot["FRACFLUX_R"] < fmax)
                         & (phot["FRACFLUX_Z"] < fmax))
    cat["FRACFLUX_PASS"] = fracflux_pass

    # ---- report ------------------------------------------------------------ #
    log.info("PHOT_SOURCE breakdown:")
    for src_name in ("tractorphot", "lowz_target", "tractorphot_fallback", "none"):
        log.info(f"  {src_name:22s}: {(phot_source == src_name).sum():,}")
    keep = matched & fracflux_pass
    log.info(f"matched={matched.sum():,}; FRACFLUX(all g,r,z<{fmax}) pass={fracflux_pass.sum():,}; "
             f"clean (matched & pass)={keep.sum():,}")

    # ---- write both catalogs ----------------------------------------------- #
    def _meta(tab, ncut_desc, nout):
        tab.meta["NIN"] = n
        tab.meta["NMATCH"] = int(matched.sum())
        tab.meta["NOUT"] = int(nout)
        tab.meta["FFLUXMAX"] = fmax
        tab.meta["FFLUXDEF"] = "all g,r,z < FFLUXMAX"
        tab.meta["COMMENT"] = (
            "matterhorn BGS/LOW_Z photometry: BGS via gather_tractorphot, "
            "LOW_Z-only via Elise DR9 LOW_Z targets (PHOT_SOURCE). " + ncut_desc)

    os.makedirs(os.path.dirname(os.path.abspath(phot_output)), exist_ok=True)
    cat.meta["EXTNAME"] = "BGS_LOWZ_PHOT"
    _meta(cat, "No FRACFLUX cut applied (full z<0.2 sample with photometry).", n)
    cat.write(phot_output, format="fits", overwrite=True)
    log.info(f"Wrote {n:,} rows -> {phot_output}")

    clean = cat[keep]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    clean.meta["EXTNAME"] = "BGS_LOWZ_CLEAN"
    _meta(clean, "FRACFLUX cut applied; matched sources only.", len(clean))
    clean.write(args.output, format="fits", overwrite=True)
    log.info(f"Wrote {len(clean):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
