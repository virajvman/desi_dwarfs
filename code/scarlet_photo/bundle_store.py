"""
bundle_store.py -- per-brick HDF5 store of SCARLET VI bundles (stage-2 output;
the store the VI tool reads).

Deliberately parallel to cutout_store / psf_store / reconstructed_store:

    {dwarf_cutouts}/scarlet_bundles/{brick[:3]}/{brick}.h5
        /{TARGETID}/                      one group per object, mirroring the
            input_cutout    (3,S,S)       per-object fragment written by stage 1
            science_cube    (3,S,S)       (fragment.py)
            nondwarf_cube   (3,S,S)
            residual_cube   (3,S,S)
            components/{id}  (3,h,w)       per-component observed-frame patches
            <group attrs>                  TARGETID, BRICKNAME, box_size, gal_*,
                                           gr_cut, rz_cut, version, created, ...

A consolidation pass (consolidate.py) copies each per-object fragment verbatim
into its brick shard, so this store is a faithful pack of the fragments.

CONTAINER-SAFE: imports only stdlib + numpy at module level; h5py is imported
inside functions. This module must NOT import scarlet or the stage-1 fitter, so
it can run in the bare consolidation container (see scarlet_photo/__init__.py).

Single-writer-per-shard is guaranteed by partitioning consolidation by brick.
"""

import os
import csv
import time
import shutil

import numpy as np

BUNDLE_BASE = "/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts"
BUNDLE_DIR = os.path.join(BUNDLE_BASE, "scarlet_bundles")

MANIFEST_NAME = "scarlet_bundle_manifest.csv"
MANIFEST_FIELDS = (
    "brickname", "targetid", "box_size", "n_components", "n_members",
    "version", "created",
)


def get_store_dir():
    return BUNDLE_DIR


def shard_path(bundle_dir, brickname):
    return os.path.join(bundle_dir, str(brickname)[:3], "{}.h5".format(brickname))


def manifest_path(bundle_dir):
    return os.path.join(bundle_dir, MANIFEST_NAME)


# ----------------------------------------------------------------------
# writing (pack per-object fragments into a brick shard)
# ----------------------------------------------------------------------

def _manifest_row_from_attrs(brickname, attrs):
    return {
        "brickname": str(brickname),
        "targetid": int(attrs.get("TARGETID", -1)),
        "box_size": int(attrs.get("box_size", 0)),
        "n_components": int(attrs.get("n_components", 0)),
        "n_members": int(attrs.get("n_members", 0)),
        "version": str(attrs.get("version", "")),
        "created": str(attrs.get("created", "")),
    }


def write_bundle_batch(bundle_dir, brickname, records):
    """Atomically pack per-object fragments into the shard for `brickname`.

    records: iterable of dicts with keys
        targetid (int), fragment_path (str)  -- path to {FILE_PATH}/scarlet_vi_bundle.h5

    Each fragment's full contents (datasets + components group + attrs) are
    copied verbatim into a group str(TARGETID). Returns manifest row dicts.
    Same atomic copy->.tmp->os.replace pattern as the sibling stores.
    """
    import h5py

    records = list(records)
    if not records:
        return []

    final = shard_path(bundle_dir, brickname)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    if os.path.exists(final):
        shutil.copy2(final, tmp)

    manifest_rows = []
    with h5py.File(tmp, "a") as f:
        for rec in records:
            tgid = int(rec["targetid"])
            frag_path = rec["fragment_path"]
            if not os.path.exists(frag_path):
                continue
            key = str(tgid)
            if key in f:
                del f[key]
            g = f.create_group(key)
            with h5py.File(frag_path, "r") as src:
                for name in src.keys():
                    src.copy(name, g)            # datasets + components/ subgroup
                for ak, av in src.attrs.items():
                    g.attrs[ak] = av
                manifest_rows.append(_manifest_row_from_attrs(brickname, g.attrs))

    os.replace(tmp, final)
    return manifest_rows


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------

def read_bundle(bundle_dir, brickname, targetid):
    """Read one object's bundle. Returns a dict with the four cubes, a
    `components` dict {comp_id: {'patch': (3,h,w), **attrs}}, and the group
    attrs."""
    import h5py

    path = shard_path(bundle_dir, brickname)
    with h5py.File(path, "r") as f:
        key = str(int(targetid))
        if key not in f:
            raise KeyError("TARGETID {} not in bundle shard {}".format(targetid, path))
        g = f[key]
        out = {}
        for name in ("input_cutout", "science_cube", "nondwarf_cube", "residual_cube"):
            if name in g:
                out[name] = g[name][:]
        comps = {}
        if "components" in g:
            for cid, ds in g["components"].items():
                rec = {"patch": ds[:]}
                for ak, av in ds.attrs.items():
                    rec[ak] = av
                comps[cid] = rec
        out["components"] = comps
        out["attrs"] = {k: g.attrs[k] for k in g.attrs.keys()}
    return out


def bundle_exists(bundle_dir, brickname, targetid):
    import h5py
    path = shard_path(bundle_dir, brickname)
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return str(int(targetid)) in f
    except (OSError, ValueError):
        return False


def list_shard_targetids(bundle_dir, brickname):
    import h5py
    path = shard_path(bundle_dir, brickname)
    if not os.path.exists(path):
        return set()
    try:
        with h5py.File(path, "r") as f:
            return {int(k) for k in f.keys()}
    except (OSError, ValueError):
        return set()


# ----------------------------------------------------------------------
# manifest (rebuildable cache; shards are ground truth)
# ----------------------------------------------------------------------

def _iter_shard_paths(bundle_dir):
    if not os.path.isdir(bundle_dir):
        return
    for prefix in sorted(os.listdir(bundle_dir)):
        subdir = os.path.join(bundle_dir, prefix)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if fname.endswith(".h5"):
                yield os.path.join(subdir, fname)


def write_manifest(bundle_dir, rows):
    path = manifest_path(bundle_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    os.replace(tmp, path)


def rebuild_manifest(bundle_dir):
    """Scan every shard and rewrite the manifest from ground truth."""
    import h5py
    rows = []
    for path in _iter_shard_paths(bundle_dir):
        brick = os.path.splitext(os.path.basename(path))[0]
        try:
            with h5py.File(path, "r") as f:
                for k in f.keys():
                    rows.append(_manifest_row_from_attrs(brick, f[k].attrs))
        except (OSError, ValueError):
            continue
    write_manifest(bundle_dir, rows)
    return rows
