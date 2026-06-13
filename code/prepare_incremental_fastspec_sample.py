"""
Build an incremental mpi-fastspecfit sample from dwarf catalog MAIN vs merged fastspec.

The incremental fit writes into a SEPARATE scratch --outdir-data (not the canonical
450k tree). Because that scratch tree starts empty, every healpix is "cold" from
mpi-fastspecfit's point of view, so nothing is ever skipped and the canonical tree
is never touched. The run sample therefore contains ONLY the missing TARGETIDs --
no neighbor re-fitting, no surgical deletes, no disk/catalog divergence.

The script still reports, for information only, how many missing objects fall in
healpix that the canonical run never processed ("new region"); this no longer
changes behavior.

Usage:
    python prepare_incremental_fastspec_sample.py --dry-run
    python prepare_incremental_fastspec_sample.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table, unique


REQUIRED_COLS = ("SURVEY", "PROGRAM", "HEALPIX", "TARGETID")

DEFAULT_DWARF_CATALOG = (
    "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/"
    "desi_dr1_dwarf_catalog.fits"
)
DEFAULT_FASTSPEC_MERGED = (
    "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/"
    "iron/catalogs/fastspec-iron-dr1-dwarfs.fits"
)
DEFAULT_OUTDIR_DATA = (
    "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/"
)
DEFAULT_INCREMENTAL_OUTDIR = (
    "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_incremental_run/"
)
DEFAULT_OUT_SAMPLE = (
    "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/"
    "desi_dr1_dwarfs_fastspec_incremental.fits"
)


def _decode_str(arr):
    if arr.dtype.kind in ("S", "O"):
        return np.array([
            x.decode("ascii") if isinstance(x, bytes) else str(x)
            for x in arr
        ])
    return np.asarray(arr, dtype=str)


def _healpix_outfile(outdir_data, specprod, survey, program, healpix, gzip=False):
    """Mirror fastspecfit.mpi.findfiles output path layout."""
    h = int(healpix)
    suffix = "fits.gz" if gzip else "fits"
    return os.path.join(
        outdir_data,
        specprod,
        "healpix",
        str(survey),
        str(program),
        str(h // 100),
        str(h),
        f"fastspec-{survey}-{program}-{h}.{suffix}",
    )


def _resolve_outfile(outdir_data, specprod, survey, program, healpix):
    """Return per-healpix fastspec path, preferring an existing .fits or .fits.gz."""
    plain = _healpix_outfile(outdir_data, specprod, survey, program, healpix, gzip=False)
    if os.path.isfile(plain):
        return plain
    gz = _healpix_outfile(outdir_data, specprod, survey, program, healpix, gzip=True)
    if os.path.isfile(gz):
        return gz
    return plain


def _read_dwarf_main(path, hdu):
    with fits.open(path) as hdul:
        tab = Table(hdul[hdu].data)
    missing_cols = [c for c in REQUIRED_COLS if c not in tab.colnames]
    if missing_cols:
        raise ValueError(f"{path} hdu={hdu} missing columns {missing_cols}")
    tab["SURVEY"] = _decode_str(tab["SURVEY"])
    tab["PROGRAM"] = _decode_str(tab["PROGRAM"])
    tab["HEALPIX"] = tab["HEALPIX"].astype(np.int64)
    tab["TARGETID"] = tab["TARGETID"].astype(np.int64)
    return tab


def _read_fastspec_tids(path, hdu):
    with fits.open(path) as hdul:
        tab = Table(hdul[hdu].data)
    if "TARGETID" not in tab.colnames:
        raise ValueError(f"{path} hdu={hdu} has no TARGETID column")
    return np.unique(tab["TARGETID"].astype(np.int64))


def _healpix_key(survey, program, healpix):
    return (str(survey), str(program), int(healpix))


def argument_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--dwarf-catalog",
        default=DEFAULT_DWARF_CATALOG,
        help="Consolidated dwarf catalog FITS (MAIN or named HDU).",
    )
    p.add_argument(
        "--dwarf-hdu",
        default="MAIN",
        help="HDU name or index for dwarf catalog.",
    )
    p.add_argument(
        "--fastspec-merged",
        default=DEFAULT_FASTSPEC_MERGED,
        help="Existing merged fastspec catalog.",
    )
    p.add_argument(
        "--fastspec-hdu",
        type=int,
        default=3,
        help="HDU index with TARGETID for coverage check (FASTSPEC).",
    )
    p.add_argument(
        "--outdir-data",
        default=DEFAULT_OUTDIR_DATA,
        help="Canonical 450k fastspecfit --outdir-data root. Read-only here; used "
        "ONLY to report how many missing objects fall in never-processed healpix.",
    )
    p.add_argument(
        "--incremental-outdir",
        default=DEFAULT_INCREMENTAL_OUTDIR,
        help="Scratch --outdir-data the incremental fit writes to (and combine reads "
        "from). Must match run_incremental_fastspec_job.sh. Never the canonical tree.",
    )
    p.add_argument(
        "--specprod",
        default="iron",
        help="DESI specprod name.",
    )
    p.add_argument(
        "--out-sample",
        default=DEFAULT_OUT_SAMPLE,
        help="Output incremental sample FITS for mpi-fastspecfit.",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="JSON manifest path (default: <out-sample>.manifest.json).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifest and print summary only; do not write sample FITS.",
    )
    return p


def main(argv=None):
    args = argument_parser().parse_args(argv)

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.out_sample + ".manifest.json"

    dwarf = _read_dwarf_main(args.dwarf_catalog, args.dwarf_hdu)
    have_tids = _read_fastspec_tids(args.fastspec_merged, args.fastspec_hdu)

    dwarf_tids = dwarf["TARGETID"].data
    missing_mask = ~np.isin(dwarf_tids, have_tids)
    missing = dwarf[missing_mask]
    n_missing = len(missing)

    base_manifest = {
        "dwarf_catalog": os.path.abspath(args.dwarf_catalog),
        "dwarf_hdu": args.dwarf_hdu,
        "fastspec_merged": os.path.abspath(args.fastspec_merged),
        "fastspec_hdu": args.fastspec_hdu,
        "outdir_data": os.path.abspath(args.outdir_data),
        "incremental_outdir": os.path.abspath(args.incremental_outdir),
        "specprod": args.specprod,
        "out_sample": os.path.abspath(args.out_sample),
        "n_dwarf": len(dwarf),
        "n_have_fastspec": len(have_tids),
    }

    if n_missing == 0:
        manifest = {
            **base_manifest,
            "n_missing_targetids": 0,
            "n_run_sample_rows": 0,
            "n_healpix": 0,
            "n_healpix_new_region": 0,
            "missing_targetids": [],
            "healpix": [],
        }
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)) or ".", exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print("=" * 72)
        print("INCREMENTAL FASTSPEC PREP: nothing to do (0 missing TARGETIDs)")
        print(f"  manifest -> {manifest_path}")
        print("=" * 72)
        sys.exit(0)

    # One run-sample row per missing TARGETID -- nothing else. The fit writes to
    # the empty scratch tree, so neighbors are never at risk and need not be refit.
    run_sample = unique(missing[list(REQUIRED_COLS)], keys=["TARGETID"])
    run_sample = run_sample[list(REQUIRED_COLS)]

    missing_keys = {
        _healpix_key(row["SURVEY"], row["PROGRAM"], row["HEALPIX"])
        for row in missing
    }

    healpix_entries = []
    n_new_region = 0
    for key in sorted(missing_keys):
        survey, program, healpix = key
        # Path the incremental fit WILL write (scratch tree, gzipped like all fastspec).
        scratch_outfile = _healpix_outfile(
            args.incremental_outdir, args.specprod, survey, program, healpix, gzip=True
        )
        # Informational only: does the canonical 450k tree already cover this healpix?
        canonical_outfile = _resolve_outfile(
            args.outdir_data, args.specprod, survey, program, healpix
        )
        in_canonical = os.path.isfile(canonical_outfile)
        n_miss_in_hp = int(np.sum(
            missing_mask
            & (dwarf["SURVEY"] == survey)
            & (dwarf["PROGRAM"] == program)
            & (dwarf["HEALPIX"] == healpix)
        ))
        healpix_entries.append({
            "survey": survey,
            "program": program,
            "healpix": healpix,
            "outfile": scratch_outfile,
            "n_missing_targets": n_miss_in_hp,
            "in_canonical": in_canonical,
        })
        if not in_canonical:
            n_new_region += 1

    manifest = {
        **base_manifest,
        "n_missing_targetids": n_missing,
        "n_run_sample_rows": len(run_sample),
        "n_healpix": len(healpix_entries),
        "n_healpix_new_region": n_new_region,
        "missing_targetids": missing["TARGETID"].tolist(),
        "healpix": healpix_entries,
    }

    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)) or ".", exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    B, E = "\033[1m", "\033[0m"
    bar = "=" * 72
    print(B + bar + E)
    print(B + "INCREMENTAL FASTSPEC PREP" + E)
    print(B + f"  dwarf catalog:     {args.dwarf_catalog} (hdu={args.dwarf_hdu})" + E)
    print(B + f"  fastspec merged:   {args.fastspec_merged} (hdu={args.fastspec_hdu})" + E)
    print(B + f"  incremental outdir:{args.incremental_outdir}" + E)
    print(B + f"  dwarf rows:        {len(dwarf):,d}" + E)
    print(B + f"  have fastspec:     {len(have_tids):,d}" + E)
    print(B + f"  MISSING TARGETIDs: {n_missing:,d}" + E)
    print(B + f"  healpix to fit:    {len(healpix_entries):,d}" + E)
    print(
        B + f"  new-region healpix:{n_new_region:,d}  "
        f"(not in canonical 450k tree; informational)" + E
    )
    print(B + f"  run sample rows:   {len(run_sample):,d}  (missing only -- no neighbor refit)" + E)
    print(B + f"  manifest ->        {manifest_path}" + E)
    if args.dry_run:
        print(B + "  DRY RUN: sample FITS not written" + E)
    else:
        print(B + f"  out sample ->      {args.out_sample}" + E)
    print(B + bar + E)

    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_sample)) or ".", exist_ok=True)
        run_sample.write(args.out_sample, overwrite=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
