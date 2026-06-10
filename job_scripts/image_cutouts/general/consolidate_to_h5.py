#!/usr/bin/env python3

"""
consolidate_to_h5.py

Read cutouts from the per-brick HDF5 shard store (written by
many_cutouts_general.py, layout in code/cutout_store.py) and pack them into
a single fixed-size HDF5 file suitable for ML training.

Usage:
  python3 consolidate_to_h5.py \
      --catalog-path /path/to/catalog.fits \
      --cutouts-dir /path/to/shard/store/ \
      --output-h5 /path/to/output.h5 \
      --cutout-size 152 --extra-cols "Z,MAG_G,MAG_R,MAG_Z" \
      --include-invvar --include-maskbits

This script does NOT require the Shifter container or MPI. It only needs
numpy, h5py, astropy, and runs on a single node. Set
HDF5_USE_FILE_LOCKING=FALSE when reading shards on Lustre.
"""

import os
import sys
import time
import argparse
import multiprocessing
from collections import defaultdict

import numpy as np

# repo layout: job_scripts/image_cutouts/general/ -> repo root -> code/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..', 'code'))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import cutout_store


def get_binary_mask(maskbits_array, set_bits=(0, 1, 2, 3, 4, 5, 6, 7, 11)):
    """Convert a maskbits array into a binary clean/masked image.

    A pixel is "clean" (1) if none of the specified bits are set;
    "masked" (0) otherwise.  Default bits follow the Legacy Survey convention
    excluding i-band bits 14/15.

    Reference:
        https://github.com/MultimodalUniverse/MultimodalUniverse/blob/
        d20de9b5d50564ca740e170030f807fd870d7f77/scripts/legacysurvey/
        build_parent_sample.py#L176
    """
    mask_val = sum(1 << b for b in set_bits)
    return ((maskbits_array & mask_val) == 0).astype(np.uint8)


def _read_shard_group(args):
    """Read a batch of objects from one shard.

    Parameters
    ----------
    args : tuple
        (shard_path, [(slot, targetid), ...], include_invvar, include_maskbits)
        where slot is the destination row offset within the current block.

    Returns
    -------
    list of (slot, dict) with keys 'image', 'invvar', 'maskbits'
    (values are numpy arrays or None on failure).
    """
    shard, items, include_invvar, include_maskbits = args
    import h5py

    out = []
    try:
        with h5py.File(shard, "r") as f:
            for slot, tgid in items:
                res = {"image": None, "invvar": None, "maskbits": None}
                key = str(tgid)
                if key in f:
                    g = f[key]
                    try:
                        res["image"] = g["image"][:].astype(np.float32)
                        if include_invvar and "invvar" in g:
                            res["invvar"] = g["invvar"][:].astype(np.float32)
                        if include_maskbits and "mask" in g:
                            res["maskbits"] = g["mask"][:].astype(np.int32)
                    except Exception as exc:
                        print(f"WARNING: failed reading {tgid} from {shard}: {exc}",
                              flush=True)
                out.append((slot, res))
    except OSError as exc:
        print(f"WARNING: could not open shard {shard}: {exc}", flush=True)
        out = [(slot, {"image": None, "invvar": None, "maskbits": None})
               for slot, tgid in items]
    return out


def _numpy_dtype_for_h5(col_data):
    """Pick an HDF5-compatible dtype for an astropy/numpy column."""
    import h5py
    dt = np.asarray(col_data).dtype
    if np.issubdtype(dt, np.integer):
        return np.int64
    if np.issubdtype(dt, np.floating):
        return np.float64
    if np.issubdtype(dt, np.bool_):
        return np.bool_
    if np.issubdtype(dt, np.str_) or np.issubdtype(dt, np.bytes_):
        return h5py.string_dtype()
    return np.float64


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Consolidate per-brick cutout shards into a single fixed-size HDF5.",
    )

    parser.add_argument("--catalog-path", type=str, required=True,
                        help="Path to the FITS catalog (same one used for cutout generation).")
    parser.add_argument("--cutouts-dir", type=str, required=True,
                        help="Directory containing the per-brick HDF5 shard store.")
    parser.add_argument("--output-h5", type=str, required=True,
                        help="Output HDF5 file path.")

    parser.add_argument("--ra-col", type=str, default="RA")
    parser.add_argument("--dec-col", type=str, default="DEC")
    parser.add_argument("--id-col", type=str, default="TARGETID")
    parser.add_argument("--brick-col", type=str, default="BRICKNAME",
                        help="Catalog column with the Legacy Surveys brick name (shard key).")

    parser.add_argument("--cutout-size", type=int, default=152,
                        help="Expected cutout dimension in pixels.")
    parser.add_argument("--nbands", type=int, default=3,
                        help="Number of image bands (channels).")

    parser.add_argument("--extra-cols", type=str, default="",
                        help="Comma-separated catalog columns to include in the HDF5 "
                             "(e.g. 'Z,MAG_G,MAG_R,MAG_Z,IS_DWARF').")

    parser.add_argument("--include-invvar", action="store_true",
                        help="Read and store inverse-variance maps from the shards.")
    parser.add_argument("--include-maskbits", action="store_true",
                        help="Read and store maskbits maps from the shards.")
    parser.add_argument("--binary-mask", action="store_true",
                        help="Compute and store a binary clean-pixel mask from maskbits. "
                             "Implies --include-maskbits.")
    parser.add_argument("--mask-bits", type=str, default="0,1,2,3,4,5,6,7,11",
                        help="Comma-separated bit indices to check when computing the "
                             "binary mask (Legacy Survey convention).")

    parser.add_argument("--block-size", type=int, default=64,
                        help="Number of images per HDF5 write / parallel-read batch.")
    parser.add_argument("--compression", type=str, default="gzip",
                        help="HDF5 compression filter. Use 'none' to disable.")
    parser.add_argument("--nworkers", type=int, default=8,
                        help="Number of parallel shard-reading workers.")

    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip objects missing from the store (fill with NaN). "
                             "Without this flag, the script errors if any are missing.")

    args = parser.parse_args()

    import h5py
    from astropy.table import Table

    cutouts_dir = os.path.expandvars(args.cutouts_dir)
    catalog_path = os.path.expandvars(args.catalog_path)
    output_h5 = os.path.expandvars(args.output_h5)

    compression = None if args.compression.lower() == "none" else args.compression
    block = args.block_size
    sz = args.cutout_size
    nb = args.nbands
    include_invvar = args.include_invvar
    compute_binary_mask = args.binary_mask
    if compute_binary_mask:
        args.include_maskbits = True
    include_maskbits = args.include_maskbits
    set_bits = tuple(int(b.strip()) for b in args.mask_bits.split(",") if b.strip())

    extra_col_names = [c.strip() for c in args.extra_cols.split(",") if c.strip()]

    # ------------------------------------------------------------------
    # 1. Read catalog
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"Reading catalog: {catalog_path}")
    cat = Table.read(catalog_path)
    N = len(cat)
    print(f"  {N} objects in catalog")

    if args.brick_col not in cat.colnames:
        print(f"ERROR: brick column '{args.brick_col}' not in catalog", file=sys.stderr)
        sys.exit(1)

    allra = np.asarray(cat[args.ra_col], dtype=np.float64)
    alldec = np.asarray(cat[args.dec_col], dtype=np.float64)
    allobjids = np.asarray(cat[args.id_col], dtype=np.int64)
    allbricks = np.asarray(cat[args.brick_col]).astype(str)

    # ------------------------------------------------------------------
    # 2. Check existence against the shard store
    # ------------------------------------------------------------------
    print(f"Scanning shard store: {cutouts_dir}")
    store = cutout_store.list_existing(cutouts_dir, quarantine_corrupt=False)
    exists = np.array(
        [int(allobjids[k]) in store.get(allbricks[k], ()) for k in range(N)],
        dtype=bool)
    n_found = int(exists.sum())
    n_missing = N - n_found
    print(f"  Found {n_found}/{N} objects in {len(store)} shards ({n_missing} missing)")

    if n_missing > 0 and not args.skip_missing:
        print("ERROR: objects missing from the store and --skip-missing is not set. "
              "Finish cutout generation first or pass --skip-missing.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Create HDF5 and write scalar datasets
    # ------------------------------------------------------------------
    print(f"Creating HDF5: {output_h5}")
    print(f"  include_invvar={include_invvar}, include_maskbits={include_maskbits}, "
          f"binary_mask={compute_binary_mask}")
    os.makedirs(os.path.dirname(output_h5) or ".", exist_ok=True)

    with h5py.File(output_h5, "w") as hf:

        chunk_shape = (min(block, N), nb, sz, sz)
        d_images = hf.create_dataset(
            "images",
            shape=(N, nb, sz, sz),
            dtype=np.float32,
            chunks=chunk_shape,
            compression=compression,
        )

        d_invvar = None
        if include_invvar:
            d_invvar = hf.create_dataset(
                "invvar",
                shape=(N, nb, sz, sz),
                dtype=np.float32,
                chunks=chunk_shape,
                compression=compression,
            )

        d_maskbits = None
        if include_maskbits:
            maskbits_chunk = (min(block, N), sz, sz)
            # int32: DR9/DR10 maskbits use bits up to 15, which overflows int16
            d_maskbits = hf.create_dataset(
                "maskbits",
                shape=(N, sz, sz),
                dtype=np.int32,
                chunks=maskbits_chunk,
                compression=compression,
            )

        d_binary_mask = None
        if compute_binary_mask:
            d_binary_mask = hf.create_dataset(
                "binary_mask",
                shape=(N, sz, sz),
                dtype=np.uint8,
                chunks=(min(block, N), sz, sz),
                compression=compression,
            )
            hf["binary_mask"].attrs["mask_bits_checked"] = list(set_bits)

        hf.create_dataset("TARGETID", data=allobjids, dtype=np.int64)
        hf.create_dataset("RA", data=allra, dtype=np.float64)
        hf.create_dataset("DEC", data=alldec, dtype=np.float64)

        if n_missing > 0:
            hf.create_dataset("has_image", data=exists, dtype=np.bool_)

        for col_name in extra_col_names:
            if col_name not in cat.colnames:
                print(f"  WARNING: column '{col_name}' not in catalog, skipping")
                continue
            col_data = np.asarray(cat[col_name])
            dt = _numpy_dtype_for_h5(col_data)
            hf.create_dataset(col_name, data=col_data, dtype=dt)
            print(f"  Added extra column '{col_name}' (dtype={dt})")

        # ------------------------------------------------------------------
        # 4. Read shards in blocks and write to datasets
        # ------------------------------------------------------------------
        print(f"Writing images (block_size={block}, workers={args.nworkers})...")

        nan_image = np.full((nb, sz, sz), np.nan, dtype=np.float32)
        nan_invvar = np.full((nb, sz, sz), np.nan, dtype=np.float32) if include_invvar else None
        zero_maskbits = np.zeros((sz, sz), dtype=np.int32) if include_maskbits else None

        pool = multiprocessing.Pool(args.nworkers)
        try:
            for blk_start in range(0, N, block):
                blk_end = min(blk_start + block, N)
                blk_size = blk_end - blk_start

                # group the block's objects by shard so each worker opens
                # one shard and reads all its objects in one go
                by_shard = defaultdict(list)
                for j in range(blk_size):
                    k = blk_start + j
                    if exists[k]:
                        shard = cutout_store.shard_path(cutouts_dir, allbricks[k])
                        by_shard[shard].append((j, int(allobjids[k])))

                read_args = [
                    (shard, items, include_invvar, include_maskbits)
                    for shard, items in by_shard.items()
                ]
                read_results = []
                if read_args:
                    for batch in pool.imap_unordered(_read_shard_group, read_args):
                        read_results.extend(batch)

                images_blk = np.empty((blk_size, nb, sz, sz), dtype=np.float32)
                images_blk[:] = nan_image
                invvar_blk = None
                if include_invvar:
                    invvar_blk = np.empty((blk_size, nb, sz, sz), dtype=np.float32)
                    invvar_blk[:] = nan_invvar
                maskbits_blk = np.zeros((blk_size, sz, sz), dtype=np.int32) if include_maskbits else None

                for j, res in read_results:
                    img = res["image"]
                    if img is not None and img.shape == (nb, sz, sz):
                        images_blk[j] = img
                    elif img is not None:
                        print(f"  WARNING: image shape mismatch at index "
                              f"{blk_start+j}: expected ({nb},{sz},{sz}), "
                              f"got {img.shape}", flush=True)

                    if include_invvar:
                        iv = res["invvar"]
                        if iv is not None and iv.shape == (nb, sz, sz):
                            invvar_blk[j] = iv

                    if include_maskbits:
                        mb = res["maskbits"]
                        if mb is not None and mb.shape == (sz, sz):
                            maskbits_blk[j] = mb

                d_images[blk_start:blk_end] = images_blk
                if d_invvar is not None:
                    d_invvar[blk_start:blk_end] = invvar_blk
                if d_maskbits is not None:
                    d_maskbits[blk_start:blk_end] = maskbits_blk
                if d_binary_mask is not None:
                    d_binary_mask[blk_start:blk_end] = get_binary_mask(
                        maskbits_blk, set_bits)

                done = blk_end
                if done % (block * 10) == 0 or done == N:
                    elapsed = time.time() - t0
                    print(f"  {done}/{N} ({done/N:.1%}), {elapsed:.0f}s elapsed",
                          flush=True)
        finally:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    print(f"HDF5 written to: {output_h5}  ({elapsed:.1f}s total)")
    print("Done.")


if __name__ == "__main__":
    main()
