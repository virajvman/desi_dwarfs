"""
consolidate.py -- STAGE 2: pack per-object fragments into the per-brick bundle
store.

Mirrors consolidate_reconstructed.py: partition work BY BRICK (one brick = one
shard = one writer), Pool over bricks, skip objects already in the shard via
bundle_store.list_shard_targetids, and REBUILD the manifest from shards at the
end (incremental mode omits already-stored objects, so writing only this-run
rows would truncate the manifest).

CONTAINER-SAFE: numpy/h5py/astropy only (no scarlet, no stage-1 fitter). Run as

    python -m scarlet_photo.consolidate --input-catalog cat.fits --ncores 64

or as a plain script (path is bootstrapped below).
"""

import os
import sys
import argparse
import multiprocessing as mp

import numpy as np

# Make the package importable whether run as `-m scarlet_photo.consolidate` or
# as a bare script, and keep the store module container-safe.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scarlet_photo import bundle_store
else:
    from . import bundle_store

FRAGMENT_NAME = "scarlet_vi_bundle.h5"


def process_brick(task):
    """Pack all available fragments for one brick. task = (brick, rows,
    bundle_dir, overwrite, fragment_name); rows = list of {'TARGETID','FILE_PATH'}."""
    brick, rows, bundle_dir, overwrite, fragment_name = task

    existing = set() if overwrite else bundle_store.list_shard_targetids(bundle_dir, brick)
    stats = {"packed": 0, "no_fragment": 0, "skipped_exists": 0}
    records = []
    for row in rows:
        tgid = int(row["TARGETID"])
        if tgid in existing:
            stats["skipped_exists"] += 1
            continue
        frag = os.path.join(str(row["FILE_PATH"]), fragment_name)
        if not os.path.exists(frag):
            stats["no_fragment"] += 1
            continue
        records.append({"targetid": tgid, "fragment_path": frag})

    manifest_rows = bundle_store.write_bundle_batch(bundle_dir, brick, records) if records else []
    stats["packed"] = len(manifest_rows)
    return brick, manifest_rows, stats


def _group_by_brick(catalog):
    rows_by_brick = {}
    bricks = np.asarray(catalog["BRICKNAME"]).astype(str)
    tgids = np.asarray(catalog["TARGETID"])
    fpaths = np.asarray(catalog["FILE_PATH"]).astype(str)
    for b, t, fp in zip(bricks, tgids, fpaths):
        rows_by_brick.setdefault(b, []).append({"TARGETID": int(t), "FILE_PATH": fp})
    return rows_by_brick


def main(argv=None):
    from astropy.table import Table

    p = argparse.ArgumentParser(description="Consolidate SCARLET VI fragments into the per-brick bundle store.")
    p.add_argument("--input-catalog", required=True, help="FITS catalogue of objects that were fit.")
    p.add_argument("--bundle-dir", default=None, help="Override the bundle store dir.")
    p.add_argument("--ncores", type=int, default=64)
    p.add_argument("--overwrite", action="store_true", help="Re-pack objects already in a shard.")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (testing).")
    p.add_argument("--tgids", type=int, nargs="*", default=None, help="Restrict to these TARGETIDs.")
    args = p.parse_args(argv)

    bundle_dir = args.bundle_dir or bundle_store.get_store_dir()
    cat = Table.read(args.input_catalog)
    if args.tgids:
        cat = cat[np.isin(np.asarray(cat["TARGETID"]), np.asarray(args.tgids))]
    if args.limit:
        cat = cat[: args.limit]

    rows_by_brick = _group_by_brick(cat)
    tasks = [
        (brick, rows, bundle_dir, args.overwrite, FRAGMENT_NAME)
        for brick, rows in rows_by_brick.items()
    ]
    print("Consolidating {} objects across {} bricks -> {}".format(
        len(cat), len(tasks), bundle_dir))

    totals = {"packed": 0, "no_fragment": 0, "skipped_exists": 0}
    all_rows = []
    if args.ncores > 1 and len(tasks) > 1:
        with mp.Pool(args.ncores) as pool:
            for brick, manifest_rows, stats in pool.imap_unordered(process_brick, tasks):
                all_rows.extend(manifest_rows)
                for k in totals:
                    totals[k] += stats[k]
    else:
        for task in tasks:
            brick, manifest_rows, stats = process_brick(task)
            all_rows.extend(manifest_rows)
            for k in totals:
                totals[k] += stats[k]

    # rebuild manifest from shards (NOT all_rows -- incremental skips would truncate it)
    bundle_store.rebuild_manifest(bundle_dir)
    print("Done. packed={packed} no_fragment={no_fragment} skipped_exists={skipped_exists}".format(**totals))


if __name__ == "__main__":
    main()
