#!/usr/bin/env python3
"""Consolidate the per-chunk grz image cutouts into a single, catalog-matched HDF5.

The image cutouts are produced by the SSL/shred pipeline as many
``ssl_shred_data/h5_datasets/data_chunk_*.h5`` shards, each holding

    images    (n, 3, 152, 152)  float32   -- g, r, z cutouts (channels-first)
    targetid  (n,)              int64     -- DESI TARGETID
    mag_g/r/z, mstar, redshift, star_dist -- ancillary columns (not published)

Those shards are the SSL/shred *training superset*: they include objects that are
not in the published dwarf catalog and a handful of TARGETIDs that repeat across
shards. This script produces the clean image data product referenced in the
README by

  1. keeping only rows whose TARGETID is in the catalog (MAIN HDU), and
  2. dropping duplicate TARGETIDs (first occurrence in sorted-filename order wins),

then streaming the surviving cutouts into one gzip-compressed file with just the
``images`` and ``targetid`` datasets (row-matched to each other, keyed by
TARGETID -- not in catalog row order).

Two passes are used so the full stack never has to fit in memory: pass 1 reads
only the tiny ``targetid`` arrays to decide which rows to keep and size the
output; pass 2 reads and writes the kept image rows one shard at a time.

Usage (see job_scripts/make_cat/run_publish_datasets.sh for the batch wrapper):

    python3 save_cutouts_h5.py --out /path/to/desi_dr1_dwarf_catalog_images.h5
"""
import argparse
import glob
import sys

import h5py
import numpy as np
from astropy.table import Table

DEFAULT_CATALOG = (
    "/global/cfs/cdirs/desi/users/virajvm/desi_dwarf_cats/iron/"
    "desi_dr1_dwarf_catalog.fits"
)
DEFAULT_CHUNKS_GLOB = (
    "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/"
    "h5_datasets/data_chunk_*.h5"
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=DEFAULT_CATALOG,
                    help="Dwarf catalog FITS; TARGETIDs in its MAIN HDU define "
                         "the rows to keep. (default: published CFS catalog)")
    ap.add_argument("--chunks-glob", default=DEFAULT_CHUNKS_GLOB,
                    help="Glob for the input data_chunk_*.h5 shards.")
    ap.add_argument("--out", required=True,
                    help="Output HDF5 path for the consolidated cutouts.")
    ap.add_argument("--compression-level", type=int, default=4,
                    help="gzip level for the images dataset (default: 4).")
    return ap.parse_args()


def main():
    args = parse_args()

    files = sorted(glob.glob(args.chunks_glob))
    if not files:
        sys.exit(f"No chunk files matched {args.chunks_glob!r}")

    # Catalog TARGETIDs define which cutouts are published; keep the MAG_Z<20
    # subset around only to report coverage in the log.
    cat = Table.read(args.catalog, hdu="MAIN")
    cat_ids = np.asarray(cat["TARGETID"], dtype=np.int64)
    cat_set = set(cat_ids.tolist())
    magz = np.asarray(cat["MAG_Z"], dtype=float)
    zlt20_set = set(cat_ids[np.isfinite(magz) & (magz < 20)].tolist())
    print(f"[consolidate] catalog rows={len(cat_ids)} "
          f"MAG_Z<20={len(zlt20_set)}", flush=True)

    # ---- Pass 1: decide the kept rows per shard (targetid arrays only) --------
    seen = set()
    keep_idx = {}          # filename -> ndarray of row indices to keep
    pixel = None
    n_keep = n_noncat = n_dup = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            tid = np.asarray(f["targetid"][:], dtype=np.int64)
            if pixel is None:
                pixel = int(f["images"].shape[-1])
        mask = np.zeros(len(tid), dtype=bool)
        for i, t in enumerate(tid):
            t = int(t)
            if t not in cat_set:
                n_noncat += 1
            elif t in seen:
                n_dup += 1
            else:
                seen.add(t)
                mask[i] = True
        idx = np.nonzero(mask)[0]        # ascending -> valid h5py fancy index
        keep_idx[fp] = idx
        n_keep += len(idx)

    n_zlt20_with = len(zlt20_set & seen)
    print(f"[consolidate] shards={len(files)} kept={n_keep} "
          f"dropped_not_in_catalog={n_noncat} dropped_duplicate={n_dup}",
          flush=True)
    print(f"[consolidate] MAG_Z<20 with cutout={n_zlt20_with} "
          f"missing={len(zlt20_set) - n_zlt20_with}", flush=True)

    if n_keep == 0:
        sys.exit("No catalog-matched cutouts found; refusing to write empty file.")

    # ---- Pass 2: stream the kept image rows into the output ------------------
    with h5py.File(args.out, "w") as fout:
        d_img = fout.create_dataset(
            "images", (n_keep, 3, pixel, pixel), dtype="float32",
            chunks=(1, 3, pixel, pixel),
            compression="gzip", compression_opts=args.compression_level,
        )
        d_tid = fout.create_dataset(
            "targetid", (n_keep,), dtype="int64",
            chunks=(min(1024, n_keep),),
        )
        off = 0
        for fp in files:
            idx = keep_idx[fp]
            if len(idx) == 0:
                continue
            with h5py.File(fp, "r") as fin:
                d_img[off:off + len(idx)] = fin["images"][idx, ...]
                d_tid[off:off + len(idx)] = np.asarray(
                    fin["targetid"][idx], dtype=np.int64)
            off += len(idx)
            print(f"[consolidate]   {fp.split('/')[-1]}: +{len(idx)} "
                  f"({off}/{n_keep})", flush=True)
        assert off == n_keep, f"wrote {off}, expected {n_keep}"

    print(f"[consolidate] wrote {n_keep} cutouts -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
