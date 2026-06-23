"""
reconstructed_store.py

Single point of access for the per-brick HDF5 store of *reconstructed* dwarf
galaxy grz cubes -- the science-grade output of the aperture/COG photometry
pipeline (the parent-galaxy reconstruction after blend + background subtraction
and masking).

This is a SEPARATE store from the input cutout store (cutout_store.py), kept
deliberately parallel to it:

    cutout_store:        {dwarf_cutouts}/images/{brick[:3]}/{brick}.h5      (input sky cutouts)
    reconstructed_store: {dwarf_cutouts}/reconstructed/{brick[:3]}/{brick}.h5  (this module)

Why separate (decision 2026-06): a sky cutout is an immutable, class-independent
property of the sky, while a reconstruction is a *mutable, photometry-version-
dependent* catalog product. Mixing the derived product into the purge-proof
input store would risk corrupting it and couples two things with different
lifecycles. The reconstruction grid is pixel-identical to the input cutout, so
the WCS header is copied verbatim from the cutout shard and the two stores can
always be cross-read by (BRICKNAME, TARGETID).

Layout (one file per Legacy Surveys brick):

    {recon_dir}/{brick[:3]}/{brick}.h5
        /{TARGETID}/image     (3, b, b) float32   grz reconstructed cube
        /{TARGETID}.attrs:
            header           primary-HDU FITS header string (WCS), copied from
                             the cutout shard -- same pixel grid
            box_size         cube side length in pixels
            recon_variant    'isolate' or 'no_isolate' (which reconstruction
                             was stored; see consolidate_reconstructed.py)
            version          pipeline/photometry version tag
            created          UTC timestamp string
            cutout_center_ra, cutout_center_dec   center the cutout was fetched at
            recon_gal_ra, recon_gal_dec           reconstructed-galaxy aperture
                             centroid (APER_RADEC_CEN_ISOLATE; NaN where the
                             aperture fit failed)
            psfsize_g, psfsize_r, psfsize_z       coadd PSF FWHM (arcsec) of the
                             matched central Tractor source; NaN if missing

A rebuildable CSV manifest (recon_manifest.csv) caches shard contents for fast
existence checks; shards remain ground truth.

Writers must guarantee a single writer per shard. Bricks are independent files,
so partitioning work by brick (one brick -> one writer) is sufficient; this is
exactly what consolidate_reconstructed.py does.

Kept importable under Python 3.8 (numpy/h5py only; astropy stays inside
functions), mirroring cutout_store.py.
"""

import os
import csv
import time
import shutil

import numpy as np

# Canonical store location (CFS, purge-proof), parallel to the cutout images
# store. Both live under the same dwarf_cutouts base.
RECON_BASE = "/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts"
RECON_DIR = os.path.join(RECON_BASE, "reconstructed")

# Rebuildable cache of shard contents (shards remain ground truth).
MANIFEST_NAME = "recon_manifest.csv"
MANIFEST_FIELDS = (
    "brickname", "targetid", "box_size", "recon_variant", "version", "created",
)


def get_store_dir():
    """Return the reconstructed-cube store directory."""
    return RECON_DIR


def shard_path(recon_dir, brickname):
    return os.path.join(recon_dir, str(brickname)[:3], "{}.h5".format(brickname))


def manifest_path(recon_dir):
    return os.path.join(recon_dir, MANIFEST_NAME)


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def write_recon_batch(recon_dir, brickname, records,
                      compression="gzip", compression_opts=4):
    """Atomically add reconstructed-cube `records` to the shard for `brickname`.

    records: iterable of dicts with keys
        targetid (int), image (3, b, b ndarray), header (str), box_size (int),
        recon_variant (str), version (str),
        cutout_center_ra, cutout_center_dec,
        recon_gal_ra, recon_gal_dec,
        psfsize_g, psfsize_r, psfsize_z

    Returns manifest row dicts for the written records (empty list if none).

    Same atomic copy->.tmp->os.replace pattern as cutout_store: a killed job
    never leaves a half-written shard in place of a good one.
    """
    import h5py

    records = list(records)
    if not records:
        return []

    final = shard_path(recon_dir, brickname)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = final + ".tmp"

    if os.path.exists(tmp):
        os.remove(tmp)
    if os.path.exists(final):
        shutil.copy2(final, tmp)

    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_rows = []
    with h5py.File(tmp, "a") as f:
        for rec in records:
            key = str(int(rec["targetid"]))
            if key in f:
                del f[key]
            g = f.create_group(key)
            image = np.asarray(rec["image"], dtype=np.float32)
            g.create_dataset("image", data=image,
                             compression=compression,
                             compression_opts=compression_opts)
            g.attrs["header"] = rec["header"]
            g.attrs["box_size"] = int(rec["box_size"])
            g.attrs["recon_variant"] = str(rec.get("recon_variant", "isolate"))
            g.attrs["version"] = str(rec.get("version", ""))
            g.attrs["created"] = created
            g.attrs["cutout_center_ra"] = float(rec["cutout_center_ra"])
            g.attrs["cutout_center_dec"] = float(rec["cutout_center_dec"])
            g.attrs["recon_gal_ra"] = float(rec["recon_gal_ra"])
            g.attrs["recon_gal_dec"] = float(rec["recon_gal_dec"])
            g.attrs["psfsize_g"] = float(rec["psfsize_g"])
            g.attrs["psfsize_r"] = float(rec["psfsize_r"])
            g.attrs["psfsize_z"] = float(rec["psfsize_z"])
            manifest_rows.append({
                "brickname": str(brickname),
                "targetid": int(rec["targetid"]),
                "box_size": int(rec["box_size"]),
                "recon_variant": str(rec.get("recon_variant", "isolate")),
                "version": str(rec.get("version", "")),
                "created": created,
            })

    os.replace(tmp, final)
    return manifest_rows


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------

def read_recon(recon_dir, brickname, targetid):
    """Read one reconstructed cube. Returns dict with `image` (3, b, b) and the
    group attrs (header, box_size, recon_variant, version, created,
    cutout_center_ra/dec, recon_gal_ra/dec, psfsize_g/r/z)."""
    import h5py

    path = shard_path(recon_dir, brickname)
    with h5py.File(path, "r") as f:
        key = str(int(targetid))
        if key not in f:
            raise KeyError("TARGETID {} not in recon shard {}".format(targetid, path))
        g = f[key]
        out = {"image": g["image"][:]}
        for k in g.attrs.keys():
            out[k] = g.attrs[k]
    return out


def recon_exists(recon_dir, brickname, targetid):
    """True if the object's reconstructed cube is in the store."""
    import h5py
    path = shard_path(recon_dir, brickname)
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return str(int(targetid)) in f
    except (OSError, ValueError):
        return False


def list_shard_targetids(recon_dir, brickname):
    """Set of TARGETIDs present in one brick's reconstructed shard."""
    import h5py
    path = shard_path(recon_dir, brickname)
    if not os.path.exists(path):
        return set()
    try:
        with h5py.File(path, "r") as f:
            return {int(k) for k in f.keys()}
    except (OSError, ValueError):
        return set()


def get_wcs(header_str):
    """Reconstruct the astropy WCS from a stored header string."""
    from astropy.io import fits
    from astropy.wcs import WCS
    return WCS(fits.Header.fromstring(header_str))


# ----------------------------------------------------------------------
# Manifest (rebuildable cache; shards are ground truth)
# ----------------------------------------------------------------------

def _iter_shard_paths(recon_dir):
    for prefix in sorted(os.listdir(recon_dir)) if os.path.isdir(recon_dir) else []:
        subdir = os.path.join(recon_dir, prefix)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if fname.endswith(".h5"):
                yield os.path.join(subdir, fname)


def write_manifest(recon_dir, rows):
    """(Re)write the manifest CSV from row dicts."""
    path = manifest_path(recon_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    os.replace(tmp, path)


def rebuild_manifest(recon_dir):
    """Scan every shard and rewrite the manifest from ground truth. Returns the
    row dicts written."""
    import h5py
    rows = []
    for path in _iter_shard_paths(recon_dir):
        brick = os.path.splitext(os.path.basename(path))[0]
        try:
            with h5py.File(path, "r") as f:
                for k in f.keys():
                    g = f[k]
                    rows.append({
                        "brickname": brick,
                        "targetid": int(k),
                        "box_size": int(g.attrs.get("box_size", 0)),
                        "recon_variant": str(g.attrs.get("recon_variant", "")),
                        "version": str(g.attrs.get("version", "")),
                        "created": str(g.attrs.get("created", "")),
                    })
        except (OSError, ValueError):
            continue
    write_manifest(recon_dir, rows)
    return rows
