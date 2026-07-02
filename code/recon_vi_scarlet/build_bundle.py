"""build_bundle.py -- pack a VI catalog's SCARLET bundles into one HDF5 file.

Reads per-object groups out of the stage-2 per-brick bundle store
(``scarlet_photo.consolidate`` output: ``{bundle_dir}/{brick[:3]}/{brick}.h5``)
and copies them VERBATIM into a single ``bundle.h5`` keyed by TARGETID, plus a
top-level ``/index`` table for fast sidebar loading. Nothing is re-derived --
the shard groups already contain everything the GUI needs (four cubes +
component patches + attrs).

Runs anywhere with numpy/h5py/astropy (NERSC container or laptop)::

    python -m recon_vi_scarlet.build_bundle \
        -vi_catalog PICKED.fits \
        -bundle_dir /global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts/scarlet_bundles \
        -out bundle.h5 [-float16] [-limit N]

Required catalog columns: TARGETID, BRICKNAME. Objects missing from the store
are skipped (logged to ``<out>_skipped.csv``); the run never aborts.

``-float16`` stores the four display cubes as float16 (~halves the bundle);
component patches always stay float32 (they are summed in the compositor).
"""

import os
import sys
import csv
import time
import argparse

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from scarlet_photo import bundle_store  # noqa: E402  (container-safe module)

CUBE_NAMES = ("input_cutout", "science_cube", "nondwarf_cube", "residual_cube")


def _read_table(path):
    from astropy.table import Table
    return Table.read(path)


def _copy_object(shard_h5, out_h5, targetid, float16):
    """Copy group str(targetid) from an open shard file into the output file.
    Returns the copied group. With float16, the four cubes are rewritten as
    float16 (patches and everything else stay verbatim)."""
    key = str(int(targetid))
    if key in out_h5:
        del out_h5[key]
    shard_h5.copy(key, out_h5, name=key)
    g = out_h5[key]
    if float16:
        kw = dict(compression="gzip", compression_opts=4)
        for name in CUBE_NAMES:
            if name not in g:
                continue
            data = g[name][:].astype(np.float16)
            del g[name]
            g.create_dataset(name, data=data, **kw)
    return g


def _write_index(h5, index_rows):
    """Top-level /index compound table for fast list/sidebar loading."""
    import h5py
    dt = np.dtype([
        ("targetid", "i8"),
        ("brickname", h5py.string_dtype()),
        ("box_size", "i4"),
        ("n_components", "i4"),
        ("n_members", "i4"),
    ])
    arr = np.empty(len(index_rows), dtype=dt)
    for i, r in enumerate(index_rows):
        arr[i] = (r["targetid"], r["brickname"], r["box_size"],
                  r["n_components"], r["n_members"])
    if "index" in h5:
        del h5["index"]
    h5.create_dataset("index", data=arr)


def _write_skipped(path, skipped):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TARGETID", "BRICKNAME", "reason"])
        for s in skipped:
            w.writerow([s["targetid"], s["brickname"], s["reason"]])


def argument_parser():
    p = argparse.ArgumentParser(
        description="Pack SCARLET VI bundles for a VI catalog into one HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-vi_catalog", required=True,
                   help="FITS catalog with TARGETID + BRICKNAME (row order = VI order)")
    p.add_argument("-bundle_dir", default=None,
                   help="stage-2 bundle store dir (default: bundle_store.get_store_dir())")
    p.add_argument("-out", required=True, help="output bundle .h5")
    p.add_argument("-float16", action="store_true",
                   help="store the four display cubes as float16 (patches stay float32)")
    p.add_argument("-limit", type=int, default=None, help="debug: first N rows only")
    return p


def main(argv=None):
    import h5py

    args = argument_parser().parse_args(argv)
    bundle_dir = args.bundle_dir or bundle_store.get_store_dir()

    cat = _read_table(args.vi_catalog)
    for col in ("TARGETID", "BRICKNAME"):
        if col not in cat.colnames:
            raise KeyError("VI catalog missing required column {}".format(col))
    rows = [(int(r["TARGETID"]), str(r["BRICKNAME"])) for r in cat]
    if args.limit:
        rows = rows[: args.limit]
    print("Packing {} objects from {} -> {}".format(len(rows), bundle_dir, args.out),
          flush=True)

    out_tmp = args.out + ".tmp"
    if os.path.exists(out_tmp):
        os.remove(out_tmp)

    index_rows = []
    skipped = []
    t0 = time.time()
    shard_cache = {}   # brickname -> open h5py.File (bundles cluster by brick)

    try:
        with h5py.File(out_tmp, "w") as out_h5:
            for tgid, brick in rows:
                spath = bundle_store.shard_path(bundle_dir, brick)
                try:
                    if brick not in shard_cache:
                        shard_cache[brick] = h5py.File(spath, "r")
                    shard = shard_cache[brick]
                    if str(tgid) not in shard:
                        raise KeyError("TARGETID not in shard")
                    g = _copy_object(shard, out_h5, tgid, args.float16)
                except (OSError, KeyError, ValueError) as exc:
                    skipped.append({"targetid": tgid, "brickname": brick,
                                    "reason": repr(exc)})
                    print("SKIP {} ({}): {!r}".format(tgid, brick, exc), flush=True)
                    continue
                index_rows.append({
                    "targetid": tgid,
                    "brickname": brick,
                    "box_size": int(g.attrs.get("box_size", 0)),
                    "n_components": int(g.attrs.get("n_components", 0)),
                    "n_members": int(g.attrs.get("n_members", 0)),
                })
                if (len(index_rows) % 100) == 0:
                    print("  {}/{} packed ({} skipped, {:.0f}s)".format(
                        len(index_rows), len(rows), len(skipped), time.time() - t0),
                        flush=True)
            _write_index(out_h5, index_rows)
            out_h5.attrs["n_objects"] = len(index_rows)
            out_h5.attrs["bundle_dir"] = str(bundle_dir)
            out_h5.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    finally:
        for f in shard_cache.values():
            f.close()

    os.replace(out_tmp, args.out)

    if skipped:
        skip_csv = os.path.splitext(args.out)[0] + "_skipped.csv"
        _write_skipped(skip_csv, skipped)
        print("  skipped objects logged to {}".format(skip_csv))
    print("Done: {} packed, {} skipped ({:.0f}s).".format(
        len(index_rows), len(skipped), time.time() - t0))


if __name__ == "__main__":
    main()
