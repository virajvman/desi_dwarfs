"""
consolidate_reconstructed.py

Post-photometry consolidation pass: pack the per-object grz reconstructed-galaxy
cubes (written by the aperture/COG pipeline into each object folder) into the
per-brick HDF5 reconstructed-cube store (reconstructed_store.py).

Why a separate pass (not done live in dwarf_photo_pipeline.py): the photometry
pipeline parallelizes by OBJECT (mp.Pool over objects), so many workers touch
the same brick at once. The store is brick-sharded with a single-writer-per-
shard rule. Here we group objects BY BRICK and hand each brick to one worker --
distinct bricks are distinct files, so a plain process pool is safe.

For each object we store the ISOLATE reconstruction
(final_reconstruct_galaxy_subfunction.npy). For the lowest-redshift objects
(z < 0.005) the pipeline never runs the isolate path, so only the no_isolate
cube exists; we fall back to it and record which was stored in the
`recon_variant` attr.

Run AFTER -run_cog has produced the w_aper_mags catalog (it now carries
PSFSIZE_G/R/Z, APER_RADEC_CEN_ISOLATE, APER_RADEC_CEN_NO_ISOLATE). WCS headers
and the cutout center come from the input cutout store (same pixel grid).

Example:
    python3 desi_dwarfs/code/consolidate_reconstructed.py \
        -sample BGS_BRIGHT -use_sample shred -ncores 64 -version v1
"""

import os
import sys
import argparse
import multiprocessing as mp

import numpy as np
from astropy.table import Table, vstack

import cutout_store
import reconstructed_store

PHOTO_CAT_DIR = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry"

# Per-object cube filenames written by aperture_cogs.py (aperture_cogs.py:1038).
ISOLATE_CUBE = "final_reconstruct_galaxy_subfunction.npy"
NO_ISOLATE_CUBE = "final_reconstruct_galaxy_subfunction_no_isolate.npy"

# Columns the consolidation needs out of the w_aper_mags catalog.
NEEDED_COLS = (
    "TARGETID", "BRICKNAME", "FILE_PATH",
    "PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_Z",
    "APER_RADEC_CEN_ISOLATE", "APER_RADEC_CEN_NO_ISOLATE",
)


def argument_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-sample", dest="sample", type=str, default="BGS_BRIGHT",
                   help="sample name, or comma-separated list of samples")
    p.add_argument("-use_sample", dest="use_sample", type=str, default="shred",
                   choices=["shred", "clean", "sga"])
    p.add_argument("-ncores", dest="ncores", type=int, default=64)
    p.add_argument("-version", dest="version", type=str, default="v1",
                   help="pipeline/photometry version tag stored with each cube")
    p.add_argument("-end_name", dest="end_name", type=str, default="",
                   help="suffix matching the w_aper_mags catalog name")
    p.add_argument("-overwrite_photometry", dest="overwrite_photometry",
                   action="store_true",
                   help="overwrite cubes already in the shard. Without it, "
                        "objects already present are skipped (incremental "
                        "mode), mirroring dwarf_photo_pipeline.py.")
    return p


def _clean_flag(use_sample):
    return {"shred": "shreds", "clean": "clean", "sga": "sga"}[use_sample]


def catalog_path(sample, use_sample, end_name):
    return os.path.join(
        PHOTO_CAT_DIR,
        "iron_{s}_{f}_catalog_w_aper_mags{e}.fits".format(
            s=sample, f=_clean_flag(use_sample), e=end_name))


def load_catalog(samples, use_sample, end_name):
    """Read+vstack the needed columns from one or more sample catalogs."""
    tables = []
    for s in samples:
        path = catalog_path(s, use_sample, end_name)
        if not os.path.exists(path):
            raise FileNotFoundError("Catalog not found: {}".format(path))
        t = Table.read(path)
        missing = [c for c in NEEDED_COLS if c not in t.colnames]
        if missing:
            raise KeyError("Catalog {} missing columns {}".format(path, missing))
        tables.append(t[list(NEEDED_COLS)])
    return vstack(tables) if len(tables) > 1 else tables[0]


def _read_cutout_attrs(h5file, tgid):
    """Read (header, center_ra, center_dec, box_size) for one TARGETID from an
    already-open cutout shard, without loading the image. Returns None if the
    object is absent."""
    key = str(int(tgid))
    if key not in h5file:
        return None
    g = h5file[key]
    return (g.attrs["header"],
            float(g.attrs["ra"]), float(g.attrs["dec"]),
            int(g.attrs["box_size"]))


def _crop_header_to(header_str, stored_size, target_size):
    """Return the WCS header for a central crop of `stored_size` down to
    `target_size`, mirroring cutout_store.crop_cutout's CRPIX/NAXIS correction.

    The pipeline crops the stored cutout to IMAGE_SIZE_PIX before measuring, so
    the reconstructed cube lives on the cropped grid; its header must carry the
    matching (shifted) CRPIX, not the raw stored-cutout header. A central crop
    leaves the cutout-center RA/DEC unchanged. No-op when sizes match."""
    stored_size = int(stored_size)
    target_size = int(target_size)
    if stored_size == target_size:
        return header_str
    from astropy.io import fits
    start = (stored_size - target_size) // 2
    hdr = fits.Header.fromstring(header_str)
    for key in ("CRPIX1", "CRPIX2"):
        if key in hdr:
            hdr[key] = hdr[key] - start
    for key, val in (("NAXIS1", target_size), ("NAXIS2", target_size)):
        if key in hdr:
            hdr[key] = val
    return hdr.tostring()


def process_brick(task):
    """Worker: build + write all reconstructed-cube records for one brick.
    Returns (brick, manifest_rows, stats) where stats is a dict of counts."""
    brick, rows, cutouts_dir, recon_dir, version, overwrite = task
    import h5py

    stats = {"written": 0, "no_cube": 0, "no_cutout": 0,
             "bad_shape": 0, "error": 0, "no_isolate_fallback": 0,
             "skipped_exists": 0, "size_mismatch": 0}

    # Incremental mode: skip objects already in this brick's shard unless we
    # are overwriting. Read the shard's TARGETIDs once, up front.
    if not overwrite:
        existing = reconstructed_store.list_shard_targetids(recon_dir, brick)
        if existing:
            keep = np.array([int(t) not in existing for t in rows["TARGETID"]])
            stats["skipped_exists"] += int((~keep).sum())
            rows = rows[keep]
            if len(rows) == 0:
                return brick, [], stats

    cutout_shard = cutout_store.shard_path(cutouts_dir, brick)
    if not os.path.exists(cutout_shard):
        stats["no_cutout"] += len(rows)
        return brick, [], stats

    records = []
    with h5py.File(cutout_shard, "r") as cf:
        for r in rows:
            try:
                tgid = int(r["TARGETID"])
                file_path = str(r["FILE_PATH"])

                # Pick the isolate cube; fall back to no_isolate (z<0.005 objects).
                cube_path = os.path.join(file_path, ISOLATE_CUBE)
                variant = "isolate"
                if not os.path.exists(cube_path):
                    cube_path = os.path.join(file_path, NO_ISOLATE_CUBE)
                    variant = "no_isolate"
                    if not os.path.exists(cube_path):
                        stats["no_cube"] += 1
                        continue
                    stats["no_isolate_fallback"] += 1

                image = np.load(cube_path)
                if image.ndim != 3 or image.shape[0] != 3:
                    stats["bad_shape"] += 1
                    continue

                attrs = _read_cutout_attrs(cf, tgid)
                if attrs is None:
                    stats["no_cutout"] += 1
                    continue
                header, cen_ra, cen_dec, stored_box = attrs

                # The cube lives on the cropped grid the pipeline measured on
                # (npix = IMAGE_SIZE_PIX <= stored). It can never be larger than
                # the cutout it was built from; if it is, the cube is stale vs.
                # the current cutout -- skip rather than store a wrong WCS.
                cube_npix = int(image.shape[-1])
                if cube_npix > stored_box:
                    stats["size_mismatch"] += 1
                    continue
                # Crop the stored header to the cube grid (CRPIX shift); no-op
                # when the cutout was stored at exactly IMAGE_SIZE_PIX.
                header = _crop_header_to(header, stored_box, cube_npix)

                # Reconstructed-galaxy center: matched aperture variant.
                cen_col = ("APER_RADEC_CEN_ISOLATE" if variant == "isolate"
                           else "APER_RADEC_CEN_NO_ISOLATE")
                gal_ra, gal_dec = float(r[cen_col][0]), float(r[cen_col][1])

                records.append({
                    "targetid": tgid,
                    "image": image,
                    "header": header,
                    "box_size": cube_npix,
                    "recon_variant": variant,
                    "version": version,
                    "cutout_center_ra": cen_ra,
                    "cutout_center_dec": cen_dec,
                    "recon_gal_ra": gal_ra,
                    "recon_gal_dec": gal_dec,
                    "psfsize_g": float(r["PSFSIZE_G"]),
                    "psfsize_r": float(r["PSFSIZE_R"]),
                    "psfsize_z": float(r["PSFSIZE_Z"]),
                })
            except Exception as e:
                stats["error"] += 1
                print("  [brick {}] error on TARGETID {}: {}".format(
                    brick, r["TARGETID"], e))

    manifest_rows = reconstructed_store.write_recon_batch(recon_dir, brick, records)
    stats["written"] = len(manifest_rows)
    return brick, manifest_rows, stats


def main():
    args = argument_parser().parse_args()
    samples = [s for s in args.sample.split(",") if s]

    cutouts_dir = cutout_store.get_store_dir()
    recon_dir = reconstructed_store.get_store_dir()
    os.makedirs(recon_dir, exist_ok=True)

    print("Reading catalog(s): {}".format(samples))
    cat = load_catalog(samples, args.use_sample, args.end_name)
    print("Total objects: {}".format(len(cat)))
    print("Cutout store:        {}".format(cutouts_dir))
    print("Reconstructed store: {}".format(recon_dir))

    # Group rows by brick so each brick is handled by exactly one worker
    # (distinct bricks are distinct shard files -> safe single-writer).
    bricks = np.asarray(cat["BRICKNAME"], dtype=str)
    tasks = []
    for brick in np.unique(bricks):
        rows = cat[bricks == brick]
        tasks.append((brick, rows, cutouts_dir, recon_dir, args.version,
                      args.overwrite_photometry))
    print("Bricks to process: {}".format(len(tasks)))
    print("Mode: {}".format("OVERWRITE existing cubes" if args.overwrite_photometry
                            else "incremental (skip objects already in store)"))

    all_manifest_rows = []
    totals = {}
    if args.ncores > 1:
        with mp.Pool(processes=args.ncores) as pool:
            for i, (brick, rows, stats) in enumerate(
                    pool.imap_unordered(process_brick, tasks), 1):
                all_manifest_rows.extend(rows)
                for k, v in stats.items():
                    totals[k] = totals.get(k, 0) + v
                if i % 200 == 0:
                    print("  {}/{} bricks done".format(i, len(tasks)))
    else:
        for i, task in enumerate(tasks, 1):
            brick, rows, stats = process_brick(task)
            all_manifest_rows.extend(rows)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v

    # Rebuild the manifest from the shards (ground truth) rather than from just
    # this run's rows -- in incremental mode all_manifest_rows omits the objects
    # already in the store, so writing it directly would truncate the manifest.
    print("Rebuilding manifest from shards...")
    manifest_rows = reconstructed_store.rebuild_manifest(recon_dir)
    print("Manifest entries: {}".format(len(manifest_rows)))

    print("=" * 50)
    print("Consolidation complete.")
    print("  cubes written:          {}".format(totals.get("written", 0)))
    print("  skipped (already in store): {}".format(totals.get("skipped_exists", 0)))
    print("  no_isolate fallbacks:   {}".format(totals.get("no_isolate_fallback", 0)))
    print("  skipped (no cube file): {}".format(totals.get("no_cube", 0)))
    print("  skipped (no cutout):    {}".format(totals.get("no_cutout", 0)))
    print("  skipped (bad shape):    {}".format(totals.get("bad_shape", 0)))
    print("  skipped (size mismatch >cutout): {}".format(totals.get("size_mismatch", 0)))
    print("  errors:                 {}".format(totals.get("error", 0)))


if __name__ == "__main__":
    sys.exit(main())
