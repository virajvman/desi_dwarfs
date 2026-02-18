#!/usr/bin/env python3

"""
consolidate_to_h5.py

Read individual FITS cutouts produced by many_cutouts_general.py and pack them
into a single HDF5 file suitable for ML training.

Usage:
  python3 consolidate_to_h5.py \
      --catalog-path /path/to/catalog.fits \
      --image-dir /path/to/cutouts/ \
      --output-h5 /path/to/output.h5 \
      --cutout-size 152 --extra-cols "Z,MAG_G,MAG_R,MAG_Z" \
      --include-invvar --include-maskbits

This script does NOT require the Shifter container or MPI.  It only needs
numpy, h5py, fitsio (or astropy), and runs on a single node.
"""

import os
import sys
import time
import argparse
import multiprocessing
import numpy as np


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


def _expected_filename(image_dir, objid, ra, dec):
    """Build the FITS filename that many_cutouts_general.py would have written."""
    return os.path.join(
        image_dir,
        f"image_tgid_{objid:d}_ra_{ra:.3f}_dec_{dec:.3f}.fits",
    )


def _read_one_fits(args):
    """Read a FITS cutout, optionally including invvar and maskbits extensions.

    Parameters
    ----------
    args : tuple
        (path, include_invvar, include_maskbits)

    Returns
    -------
    dict with keys 'image' (always), 'invvar' (optional), 'maskbits' (optional).
    Values are numpy arrays or None on failure.
    """
    path, include_invvar, include_maskbits = args
    result = {"image": None, "invvar": None, "maskbits": None}

    try:
        import fitsio
        result["image"] = fitsio.read(path, ext=0).astype(np.float32)
        if include_invvar:
            try:
                result["invvar"] = fitsio.read(path, ext="INVVAR").astype(np.float32)
            except Exception:
                try:
                    result["invvar"] = fitsio.read(path, ext=1).astype(np.float32)
                except Exception:
                    pass
        if include_maskbits:
            try:
                result["maskbits"] = fitsio.read(path, ext="MASKBITS").astype(np.int16)
            except Exception:
                try:
                    idx = 2 if include_invvar else 1
                    result["maskbits"] = fitsio.read(path, ext=idx).astype(np.int16)
                except Exception:
                    pass
        return result
    except Exception:
        pass

    try:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            result["image"] = hdul[0].data.astype(np.float32)
            if include_invvar:
                for ext_name in ("INVVAR", 1):
                    try:
                        result["invvar"] = hdul[ext_name].data.astype(np.float32)
                        break
                    except Exception:
                        continue
            if include_maskbits:
                for ext_name in ("MASKBITS", 2 if include_invvar else 1):
                    try:
                        result["maskbits"] = hdul[ext_name].data.astype(np.int16)
                        break
                    except Exception:
                        continue
        return result
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}", flush=True)
        return result


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
        description="Consolidate individual FITS cutouts into a single HDF5 file.",
    )

    parser.add_argument("--catalog-path", type=str, required=True,
                        help="Path to the FITS catalog (same one used for cutout generation).")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing individual FITS cutouts.")
    parser.add_argument("--output-h5", type=str, required=True,
                        help="Output HDF5 file path.")

    parser.add_argument("--ra-col", type=str, default="RA")
    parser.add_argument("--dec-col", type=str, default="DEC")
    parser.add_argument("--id-col", type=str, default="TARGETID")

    parser.add_argument("--cutout-size", type=int, default=152,
                        help="Expected cutout dimension in pixels.")
    parser.add_argument("--nbands", type=int, default=3,
                        help="Number of image bands (channels).")

    parser.add_argument("--extra-cols", type=str, default="",
                        help="Comma-separated catalog columns to include in the HDF5 "
                             "(e.g. 'Z,MAG_G,MAG_R,MAG_Z,IS_DWARF').")

    parser.add_argument("--include-invvar", action="store_true",
                        help="Read and store inverse-variance maps from FITS extensions.")
    parser.add_argument("--include-maskbits", action="store_true",
                        help="Read and store maskbits maps from FITS extensions.")
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
                        help="Number of parallel FITS-reading workers.")

    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip missing FITS files (fill with NaN). "
                             "Without this flag, the script errors if any are missing.")
    parser.add_argument("--delete-fits", action="store_true",
                        help="Delete individual FITS files after successful consolidation.")

    args = parser.parse_args()

    import h5py
    from astropy.table import Table

    image_dir = os.path.expandvars(args.image_dir)
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
    # 1. Read catalog and build expected file paths
    # ------------------------------------------------------------------
    t0 = time.time()
    print(f"Reading catalog: {catalog_path}")
    cat = Table.read(catalog_path)
    N = len(cat)
    print(f"  {N} objects in catalog")

    allra = np.asarray(cat[args.ra_col], dtype=np.float64)
    alldec = np.asarray(cat[args.dec_col], dtype=np.float64)
    allobjids = np.asarray(cat[args.id_col], dtype=np.int64)

    filepaths = np.array([
        _expected_filename(image_dir, allobjids[k], allra[k], alldec[k])
        for k in range(N)
    ], dtype=object)

    # ------------------------------------------------------------------
    # 2. Check file existence
    # ------------------------------------------------------------------
    print("Checking which FITS files exist...")
    exists = np.array([os.path.exists(fp) for fp in filepaths], dtype=bool)
    n_found = int(exists.sum())
    n_missing = N - n_found
    print(f"  Found {n_found}/{N} files ({n_missing} missing)")

    if n_missing > 0 and not args.skip_missing:
        print("ERROR: missing FITS files detected and --skip-missing is not set. "
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
            d_maskbits = hf.create_dataset(
                "maskbits",
                shape=(N, sz, sz),
                dtype=np.int16,
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
        # 4. Read FITS in blocks and write to datasets
        # ------------------------------------------------------------------
        print(f"Writing images (block_size={block}, workers={args.nworkers})...")

        nan_image = np.full((nb, sz, sz), np.nan, dtype=np.float32)
        nan_invvar = np.full((nb, sz, sz), np.nan, dtype=np.float32) if include_invvar else None
        zero_maskbits = np.zeros((sz, sz), dtype=np.int16) if include_maskbits else None

        pool = multiprocessing.Pool(args.nworkers)
        try:
            for blk_start in range(0, N, block):
                blk_end = min(blk_start + block, N)
                blk_size = blk_end - blk_start

                blk_paths = filepaths[blk_start:blk_end]
                blk_exists = exists[blk_start:blk_end]

                read_args = [
                    (p, include_invvar, include_maskbits)
                    for p, e in zip(blk_paths, blk_exists) if e
                ]

                if read_args:
                    read_results = list(pool.imap(_read_one_fits, read_args))
                else:
                    read_results = []

                images_blk = np.empty((blk_size, nb, sz, sz), dtype=np.float32)
                invvar_blk = np.empty((blk_size, nb, sz, sz), dtype=np.float32) if include_invvar else None
                maskbits_blk = np.zeros((blk_size, sz, sz), dtype=np.int16) if include_maskbits else None

                read_idx = 0
                for j in range(blk_size):
                    if blk_exists[j]:
                        res = read_results[read_idx]
                        read_idx += 1
                        img = res["image"]
                        if img is not None and img.shape == (nb, sz, sz):
                            images_blk[j] = img
                        else:
                            images_blk[j] = nan_image
                            if img is not None:
                                print(f"  WARNING: image shape mismatch at index "
                                      f"{blk_start+j}: expected ({nb},{sz},{sz}), "
                                      f"got {img.shape}", flush=True)

                        if include_invvar:
                            iv = res["invvar"]
                            if iv is not None and iv.shape == (nb, sz, sz):
                                invvar_blk[j] = iv
                            else:
                                invvar_blk[j] = nan_invvar

                        if include_maskbits:
                            mb = res["maskbits"]
                            if mb is not None and mb.shape == (sz, sz):
                                maskbits_blk[j] = mb
                            else:
                                maskbits_blk[j] = zero_maskbits
                    else:
                        images_blk[j] = nan_image
                        if include_invvar:
                            invvar_blk[j] = nan_invvar
                        if include_maskbits:
                            maskbits_blk[j] = zero_maskbits

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

    # ------------------------------------------------------------------
    # 5. Optionally delete individual FITS files
    # ------------------------------------------------------------------
    if args.delete_fits:
        print("Deleting individual FITS files...")
        n_deleted = 0
        for fp, ex in zip(filepaths, exists):
            if ex:
                try:
                    os.remove(fp)
                    n_deleted += 1
                except OSError as exc:
                    print(f"  WARNING: could not delete {fp}: {exc}")
        print(f"  Deleted {n_deleted} files")

    print("Done.")


if __name__ == "__main__":
    main()
