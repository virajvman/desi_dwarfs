"""add_psfsize.py -- attach native per-band PSF sizes + brickname to the crops h5.

Joins a FITS catalog (with a TARGETID column) onto the STAGE-4b light-centered
crops (make_crops.py output) by TARGETID and writes per-object arrays into the
crops h5, row-aligned to its top-level `/targetid`:

  - `psfsize_g`, `psfsize_r`, `psfsize_z` (native Legacy coadd PSF FWHM, arcsec) --
    the values the galaxy_prior_proj homogenization step reads to build each
    object's Gaussian matching kernel (sigma_match = sqrt(sigma_t^2 - sigma_b^2)).
  - `brickname` (str) -- the Legacy brick the object sits in; together with the
    light-center RA/Dec in /index it lets you pull the matching real-data cutout.

Both also satisfy the psfsize_g/r/z + brickname parts of repack_scarlet_clean.py's
input contract.

These are PER OBJECT (each galaxy sits in a different brick / seeing), so they are
length-N datasets, NOT file attributes. Row i is the value for the galaxy at
targetid[i].

The crops h5 is edited IN PLACE and the write is idempotent: existing
psfsize_g/r/z + brickname datasets are removed and rewritten, so re-running is
safe. Only those datasets (+ a few provenance attrs) are touched; images/targetid/
index are untouched.

Join policy (default, graceful):
  - a crop targetid absent from the catalog     -> NaN for that row (homogenize.py
    treats a non-finite PSF as un-homogenizable and skips it -- the right sentinel)
  - duplicate TARGETID rows in the catalog      -> first occurrence wins
  - counts of both (with a few example IDs) are printed at the end
Pass --strict to abort instead of tolerating either.

Column names default to the Legacy Surveys standard TARGETID + PSFSIZE_G/R/Z
(matched case-insensitively); override with --targetid-col / --psfsize-cols if
your catalog differs. Values are assumed to be FWHM in arcsec (no conversion).

Usage::

    python -m recon_vi_scarlet.add_psfsize \
        --crops scarlet_crops128.h5 \
        --catalog /path/psfsize_catalog.fits \
        [--hdu 1] [--targetid-col TARGETID] \
        [--psfsize-cols PSFSIZE_G PSFSIZE_R PSFSIZE_Z] \
        [--brickname-col BRICKNAME] [--strict]

Needs numpy/h5py/astropy.
"""

import os
import sys
import time
import argparse

import numpy as np


def _to_str(v):
    """Decode a FITS/numpy string scalar (bytes or str) to a plain str."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip()
    return str(v).strip()


def _resolve_col(colnames, want, kind):
    """Case-insensitive column lookup; raises a clear error if absent."""
    upmap = {c.upper(): c for c in colnames}
    if want.upper() in upmap:
        return upmap[want.upper()]
    raise KeyError(
        "catalog has no {} column {!r} (available: {}...)".format(
            kind, want, ", ".join(colnames[:12])))


def main(argv=None):
    import h5py
    from astropy.table import Table

    p = argparse.ArgumentParser(
        description="Attach native psfsize_g/r/z to the crops h5 by TARGETID join.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--crops", required=True,
                   help="scarlet_crops128.h5 from make_crops.py (edited IN PLACE)")
    p.add_argument("--catalog", required=True,
                   help="FITS catalog with TARGETID + PSF-size columns")
    p.add_argument("--hdu", type=int, default=1, help="FITS table HDU to read")
    p.add_argument("--targetid-col", default="TARGETID",
                   help="TARGETID column name in the catalog (case-insensitive)")
    p.add_argument("--psfsize-cols", nargs=3, metavar=("G", "R", "Z"),
                   default=["PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_Z"],
                   help="g/r/z PSF-size column names (FWHM arcsec, case-insensitive)")
    p.add_argument("--brickname-col", default="BRICKNAME",
                   help="Legacy brick-name column in the catalog (case-insensitive)")
    p.add_argument("--strict", action="store_true",
                   help="abort on any missing crop targetid OR duplicate catalog "
                        "TARGETID instead of NaN-filling / first-wins")
    args = p.parse_args(argv)

    if not os.path.exists(args.crops):
        sys.exit("ERROR: crops h5 not found: {}".format(args.crops))

    # --- load the catalog join columns -------------------------------------
    print("reading catalog {} (hdu={}) ...".format(args.catalog, args.hdu), flush=True)
    cat = Table.read(args.catalog, hdu=args.hdu)
    tid_col = _resolve_col(cat.colnames, args.targetid_col, "TARGETID")
    psf_cols = [_resolve_col(cat.colnames, c, "PSF-size") for c in args.psfsize_cols]
    brick_col = _resolve_col(cat.colnames, args.brickname_col, "brick-name")
    print("  using columns: {} + {} + {}".format(tid_col, psf_cols, brick_col),
          flush=True)

    cat_tid = np.asarray(cat[tid_col]).astype(np.int64)
    cat_psf = np.stack([np.asarray(cat[c], dtype=np.float64) for c in psf_cols], axis=1)
    cat_brick = np.asarray(cat[brick_col])   # often FITS bytes (|S8); decoded below

    # unique TARGETID -> first-occurrence row index (uniq is sorted, so searchsorted
    # below is exact); n_dup counts collapsed duplicate rows.
    uniq, first_idx = np.unique(cat_tid, return_index=True)
    n_dup = len(cat_tid) - len(uniq)

    # --- read the crop row order -------------------------------------------
    with h5py.File(args.crops, "r") as f:
        if "targetid" not in f:
            sys.exit("ERROR: {} has no top-level /targetid dataset (is this a "
                     "make_crops.py output?)".format(args.crops))
        crop_tid = np.asarray(f["targetid"][:]).astype(np.int64)
    n = len(crop_tid)

    # --- join: crop targetid -> catalog row (missing -> matched=False) ------
    pos = np.searchsorted(uniq, crop_tid)
    pos_clip = np.clip(pos, 0, len(uniq) - 1)
    matched = uniq[pos_clip] == crop_tid
    rows = first_idx[pos_clip]                       # valid index; masked where !matched

    psf_out = np.full((n, 3), np.nan, dtype=np.float32)
    psf_out[matched] = cat_psf[rows[matched]].astype(np.float32)

    # brickname: decode FITS bytes -> str; missing targetid -> "" (empty)
    brick_out = np.full(n, "", dtype=object)
    brick_out[matched] = np.array([_to_str(cat_brick[r]) for r in rows[matched]],
                                  dtype=object)

    n_missing = int((~matched).sum())
    n_nonfinite = int(matched.sum() - np.isfinite(psf_out[matched]).all(axis=1).sum())

    if args.strict and (n_missing or n_dup):
        sys.exit("ERROR (--strict): {} crop targetid(s) missing from catalog, "
                 "{} duplicate catalog TARGETID row(s). Examples missing: {}".format(
                     n_missing, n_dup, sorted(crop_tid[~matched].tolist())[:10]))

    # --- write the arrays in place (idempotent) ----------------------------
    bands = ("g", "r", "z")
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.crops, "a") as f:
        for bi, b in enumerate(bands):
            name = "psfsize_" + b
            if name in f:
                del f[name]
            f.create_dataset(name, data=psf_out[:, bi])
        if "brickname" in f:
            del f["brickname"]
        f.create_dataset("brickname", data=brick_out, dtype=str_dt)
        f.attrs["psf_homogenized"] = False
        f.attrs["psfsize_source_catalog"] = os.path.abspath(args.catalog)
        f.attrs["psfsize_units"] = "arcsec_fwhm"
        f.attrs["psfsize_added"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- report -------------------------------------------------------------
    print("Done: wrote psfsize_g/r/z + brickname for {} crops -> {}".format(
        n, args.crops))
    print("  matched {}/{} to catalog; {} missing (psfsize NaN, brickname \"\")".format(
        n - n_missing, n, n_missing))
    if n_dup:
        print("  NOTE: catalog had {} duplicate TARGETID row(s); first occurrence "
              "used".format(n_dup))
    if n_missing:
        ex = sorted(crop_tid[~matched].tolist())[:10]
        print("  missing example targetid(s): {}{}".format(
            ex, " ..." if n_missing > 10 else ""))
    if n_nonfinite:
        print("  NOTE: {} matched object(s) have a non-finite PSF in some band "
              "(no coverage; homogenization will skip them)".format(n_nonfinite))


if __name__ == "__main__":
    main()
