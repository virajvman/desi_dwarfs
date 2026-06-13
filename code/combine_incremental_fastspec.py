"""
Vstack incremental fastspecfit rows onto an existing merged multi-HDU catalog.

Reads only the per-healpix outputs touched by the incremental run (from the
prep manifest), keeps rows for missing TARGETIDs only, and appends them to the
existing METADATA / SPECPHOT / FASTSPEC tables.

Usage:
    python combine_incremental_fastspec.py --manifest path/to/sample.manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from desiutil.depend import getdep, hasdep


DEFAULT_FASTSPEC_MERGED = (
    "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/"
    "iron/catalogs/fastspec-iron-dr1-dwarfs.fits"
)
DEFAULT_OUT_MERGED = (
    "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/"
    "iron/catalogs/fastspec-iron-dr1-dwarfs-v2.fits"
)


def _read_merged_tables(path):
    with fits.open(path) as hdul:
        meta = Table(hdul["METADATA"].data)
        specphot = Table(hdul["SPECPHOT"].data)
        fastspec = Table(hdul["FASTSPEC"].data)
        primhdr = hdul[0].header.copy()
    return meta, specphot, fastspec, primhdr


def _resolve_existing(path):
    """On-disk path, tolerating .fits vs .fits.gz.

    fastspecfit 3.4.3 always writes per-healpix outputs gzipped (.fits.gz), but
    prep records the plain .fits path for cold healpix (the file did not exist at
    prep time). Re-resolve here so cold-healpix outputs are not silently dropped.
    """
    if os.path.isfile(path):
        return path
    if path.endswith(".fits") and os.path.isfile(path + ".gz"):
        return path + ".gz"
    if path.endswith(".fits.gz") and os.path.isfile(path[:-3]):
        return path[:-3]
    return None


def _read_per_healpix_rows(path, missing_tids):
    """Read one per-healpix file; return rows with TARGETID in missing_tids."""
    from fastspecfit.mpi import read_to_merge_one

    meta, specphot, fastfit = read_to_merge_one(path, fastphot=False)
    tids = meta["TARGETID"].astype(np.int64)
    keep = np.isin(tids, missing_tids)
    if not np.any(keep):
        return None, None, None
    return meta[keep], specphot[keep], fastfit[keep]


def _header_deps(primhdr):
    deps = {
        "INPUTZ": primhdr.get("INPUTZ", False),
        "INPUTS": primhdr.get("INPUTS", False),
        "CONSAGE": primhdr.get("CONSAGE", False),
        "USEQNET": primhdr.get("USEQNET", True),
        "NMONTE": primhdr.get("NMONTE", 50),
        "SEED": primhdr.get("SEED", 1),
        "NOSCORR": primhdr.get("NOSCORR", False),
        "NOPHOTO": primhdr.get("NOPHOTO", False),
        "BRDLFIT": primhdr.get("BRDLFIT", True),
        "UFLOOR": primhdr.get("UFLOOR", 0.01),
        "SNRBBALM": primhdr.get("SNRBBALM", 2.5),
    }
    deps2 = {}
    for key in ("FPHOTO_FILE", "FTEMPLATES_FILE", "EMLINES_FILE", "CONSTRAINTS_FILE"):
        if hasdep(primhdr, key):
            deps2[key] = getdep(primhdr, key)
        else:
            deps2[key] = None
    return deps, deps2


def argument_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="JSON manifest from prepare_incremental_fastspec_sample.py",
    )
    p.add_argument(
        "--fastspec-merged",
        default=None,
        help="Existing merged catalog (default: manifest fastspec_merged).",
    )
    p.add_argument(
        "--out-merged",
        default=DEFAULT_OUT_MERGED,
        help="Output merged catalog path (written fresh; does not overwrite input).",
    )
    p.add_argument(
        "--replace-original",
        action="store_true",
        help="After writing --out-merged, replace fastspec-merged with it.",
    )
    return p


def main(argv=None):
    args = argument_parser().parse_args(argv)

    with open(args.manifest) as f:
        manifest = json.load(f)

    merged_in = args.fastspec_merged or manifest["fastspec_merged"]
    missing_tids = np.array(manifest["missing_targetids"], dtype=np.int64)
    n_missing = len(missing_tids)

    if n_missing == 0:
        print("combine_incremental_fastspec: 0 missing TARGETIDs; nothing to do.")
        return 0

    meta_old, specphot_old, fastspec_old, primhdr = _read_merged_tables(merged_in)
    old_tids = meta_old["TARGETID"].astype(np.int64)

    overlap = np.isin(missing_tids, old_tids)
    if np.any(overlap):
        dup = missing_tids[overlap][:10]
        raise RuntimeError(
            f"{np.sum(overlap)} missing TARGETIDs already in merged catalog "
            f"(first few: {dup.tolist()}). Aborting to avoid duplicates."
        )

    # Collect per-healpix outputs for healpix touched by the incremental run.
    per_healpix_files = []
    for entry in manifest["healpix"]:
        resolved = _resolve_existing(entry["outfile"])
        if resolved is not None:
            per_healpix_files.append(resolved)

    if not per_healpix_files:
        raise RuntimeError(
            "No per-healpix fastspec outputs found for manifest healpix entries. "
            "Run the incremental fit job first."
        )

    new_meta_parts, new_specphot_parts, new_fastspec_parts = [], [], []
    found_tids = set()

    for path in sorted(set(per_healpix_files)):
        m, s, f = _read_per_healpix_rows(path, missing_tids)
        if m is None:
            continue
        new_meta_parts.append(m)
        new_specphot_parts.append(s)
        new_fastspec_parts.append(f)
        found_tids.update(m["TARGETID"].astype(np.int64).tolist())

    if not new_meta_parts:
        raise RuntimeError(
            "Per-healpix files exist but none contain rows for missing TARGETIDs."
        )

    not_found = set(missing_tids.tolist()) - found_tids
    if not_found:
        sample = sorted(not_found)[:10]
        raise RuntimeError(
            f"{len(not_found)} missing TARGETIDs not found in per-healpix outputs "
            f"(first few: {sample}). Fit may have failed for some healpix."
        )

    meta_new = vstack(new_meta_parts)
    specphot_new = vstack(new_specphot_parts)
    fastspec_new = vstack(new_fastspec_parts)

    meta_out = vstack([meta_old, meta_new])
    specphot_out = vstack([specphot_old, specphot_new])
    fastspec_out = vstack([fastspec_old, fastspec_new])

    out_tids = meta_out["TARGETID"].astype(np.int64)
    if len(out_tids) != len(np.unique(out_tids)):
        raise RuntimeError("Duplicate TARGETIDs in merged output.")

    n_old = len(meta_old)
    n_out = len(meta_out)
    if n_out != n_old + n_missing:
        raise RuntimeError(
            f"Row count mismatch: expected {n_old + n_missing}, got {n_out}."
        )

    deps, deps2 = _header_deps(primhdr)
    from fastspecfit.io import write_fastspecfit

    os.makedirs(os.path.dirname(os.path.abspath(args.out_merged)) or ".", exist_ok=True)
    write_fastspecfit(
        meta_out,
        specphot_out,
        fastspec_out,
        modelspectra=None,
        outfile=args.out_merged,
        specprod=primhdr.get("SPECPROD", manifest.get("specprod", "iron")),
        coadd_type=primhdr.get("COADDTYP", "healpix"),
        fphotofile=deps2.get("FPHOTO_FILE"),
        template_file=deps2.get("FTEMPLATES_FILE"),
        emlinesfile=deps2.get("EMLINES_FILE"),
        constraintsfile=deps2.get("CONSTRAINTS_FILE"),
        inputz=deps["INPUTZ"],
        ignore_photometry=deps["NOPHOTO"],
        broadlinefit=deps["BRDLFIT"],
        constrain_age=deps["CONSAGE"],
        use_quasarnet=deps["USEQNET"],
        no_smooth_continuum=deps["NOSCORR"],
        nmonte=deps["NMONTE"],
        seed=deps["SEED"],
        uncertainty_floor=deps["UFLOOR"],
        minsnr_balmer_broad=deps["SNRBBALM"],
    )

    B, E = "\033[1m", "\033[0m"
    bar = "=" * 72
    print(B + bar + E)
    print(B + "INCREMENTAL FASTSPEC COMBINE" + E)
    print(B + f"  input merged:  {merged_in}" + E)
    print(B + f"  rows before:   {n_old:,d}" + E)
    print(B + f"  rows added:    {n_missing:,d}" + E)
    print(B + f"  rows after:    {n_out:,d}" + E)
    print(B + f"  output ->      {args.out_merged}" + E)
    print(B + bar + E)

    if args.replace_original:
        import shutil
        shutil.copy2(args.out_merged, merged_in)
        print(f"Replaced {merged_in} with {args.out_merged}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
