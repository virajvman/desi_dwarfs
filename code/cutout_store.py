"""
cutout_store.py

Single point of access for the per-brick HDF5 cutout store.

Layout (one file per Legacy Surveys brick):

    {cutouts_dir}/{BRICKNAME}.h5
        /{TARGETID}/image     (3, b, b) float32   grz image cube
        /{TARGETID}/invvar    (3, b, b) float32   inverse variance (optional)
        /{TARGETID}/mask      (b, b)    int32     maskbits plane (optional)
        /{TARGETID}.attrs:
            header        primary-HDU FITS header as a string (carries the WCS)
            ra, dec       catalog position
            box_size      cutout side length in pixels
            fetch_method  'container' or 'url'
            created       UTC timestamp string

Writers must guarantee a single writer per shard (the bulk MPI pipeline
partitions work by brick, so each shard is owned by exactly one rank).
Concurrent read-only opens are safe; set HDF5_USE_FILE_LOCKING=FALSE on
Lustre.

This module must stay importable under the Python 3.8 inside the
dstndstn/cutouts Shifter container: import only numpy/h5py at module level
and keep astropy imports inside functions.
"""

import os
import io
import csv
import time
import shutil

import numpy as np

# Canonical store locations (CFS, not scratch -- deliberately purge-proof).
CUTOUTS_BASE = "/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts"

# dwarf_photo_pipeline uses use_sample="shred" while the store dir is "shreds"
_STORE_NAMES = {"clean": "clean", "shred": "shreds", "shreds": "shreds", "sga": "sga"}

# Production viewer. Do NOT use viewer-dev: since imagine d2ba303 (2026-01-26)
# it sums invvar across overlapping bricks -> exactly 2x inflated INVVAR in
# brick-overlap strips. Production averages (correct) and serves maskbits now.
PRODUCTION_VIEWER = "https://www.legacysurvey.org/viewer"

# Cumulative manifest of objects that can never be fetched (off-footprint etc.)
FAILED_MANIFEST = "permanently_failed.csv"


def get_store_dir(use_sample):
    """Map a pipeline use_sample flag to its cutout store directory."""
    if use_sample not in _STORE_NAMES:
        raise ValueError("Unknown use_sample '{}' (expected one of {})".format(
            use_sample, sorted(_STORE_NAMES)))
    return os.path.join(CUTOUTS_BASE, _STORE_NAMES[use_sample])


def load_tombstones(cutouts_dir):
    """Return set of TARGETIDs recorded as permanently unfetchable."""
    path = os.path.join(cutouts_dir, FAILED_MANIFEST)
    tombs = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    tombs.add(int(row["targetid"]))
                except (KeyError, ValueError):
                    continue
    return tombs


def shard_path(cutouts_dir, brickname):
    return os.path.join(cutouts_dir, "{}.h5".format(brickname))


def list_shard_targetids(cutouts_dir, brickname):
    """TARGETIDs present in one brick's shard (empty set if shard absent/unreadable)."""
    import h5py
    path = shard_path(cutouts_dir, brickname)
    if not os.path.exists(path):
        return set()
    try:
        with h5py.File(path, "r") as f:
            return set(int(k) for k in f.keys())
    except (OSError, ValueError):
        return set()


def cutout_exists(cutouts_dir, brickname, targetid):
    import h5py
    path = shard_path(cutouts_dir, brickname)
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return str(targetid) in f
    except OSError:
        return False


def list_existing(cutouts_dir, quarantine_corrupt=True):
    """Return {brickname: set(targetid int)} for every shard in the store.

    A shard that cannot be opened is treated as absent (its objects will be
    re-fetched); with quarantine_corrupt it is also renamed to *.h5.corrupt
    so the writer can atomically recreate it.
    """
    import h5py
    from glob import glob

    existing = {}
    if not os.path.isdir(cutouts_dir):
        return existing

    for path in sorted(glob(os.path.join(cutouts_dir, "*.h5"))):
        brick = os.path.splitext(os.path.basename(path))[0]
        try:
            with h5py.File(path, "r") as f:
                existing[brick] = set(int(k) for k in f.keys())
        except (OSError, ValueError) as exc:
            print("WARNING: unreadable shard {} ({}); treating as absent".format(path, exc),
                  flush=True)
            if quarantine_corrupt:
                try:
                    os.replace(path, path + ".corrupt")
                except OSError:
                    pass
    return existing


def write_cutouts_batch(cutouts_dir, brickname, records,
                        compression="gzip", compression_opts=4):
    """Atomically add `records` to the shard for `brickname`.

    records: iterable of dicts with keys
        targetid (int), image (ndarray), header (str), ra, dec, box_size,
        fetch_method, and optionally invvar (ndarray) and mask (ndarray).

    The existing shard (if any) is copied to a .tmp file, groups are added
    (replacing any stale group with the same TARGETID), and the .tmp is
    renamed over the original — a killed job never leaves a half-written
    shard in place of a good one.
    """
    import h5py

    records = list(records)
    if not records:
        return

    os.makedirs(cutouts_dir, exist_ok=True)
    final = shard_path(cutouts_dir, brickname)
    tmp = final + ".tmp"

    if os.path.exists(tmp):
        os.remove(tmp)
    if os.path.exists(final):
        shutil.copy2(final, tmp)

    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with h5py.File(tmp, "a") as f:
        for rec in records:
            key = str(int(rec["targetid"]))
            if key in f:
                del f[key]
            g = f.create_group(key)
            g.create_dataset("image", data=np.asarray(rec["image"], dtype=np.float32),
                             compression=compression, compression_opts=compression_opts)
            if rec.get("invvar") is not None:
                g.create_dataset("invvar", data=np.asarray(rec["invvar"], dtype=np.float32),
                                 compression=compression, compression_opts=compression_opts)
            if rec.get("mask") is not None:
                # int32: DR9/DR10 maskbits use bits up to 15, which overflows int16
                g.create_dataset("mask", data=np.asarray(rec["mask"], dtype=np.int32),
                                 compression=compression, compression_opts=compression_opts)
            g.attrs["header"] = rec["header"]
            g.attrs["ra"] = float(rec["ra"])
            g.attrs["dec"] = float(rec["dec"])
            g.attrs["box_size"] = int(rec["box_size"])
            g.attrs["fetch_method"] = rec.get("fetch_method", "container")
            g.attrs["created"] = created

    os.replace(tmp, final)


def read_cutout(cutouts_dir, brickname, targetid):
    """Read one object. Returns dict with image, invvar (or None),
    mask (or None), header (str), and the group attrs."""
    import h5py

    path = shard_path(cutouts_dir, brickname)
    with h5py.File(path, "r") as f:
        key = str(int(targetid))
        if key not in f:
            raise KeyError("TARGETID {} not in shard {}".format(targetid, path))
        g = f[key]
        out = {
            "image": g["image"][:],
            "invvar": g["invvar"][:] if "invvar" in g else None,
            "mask": g["mask"][:] if "mask" in g else None,
            "header": g.attrs["header"],
        }
        for k in ("ra", "dec", "box_size", "fetch_method", "created"):
            if k in g.attrs:
                out[k] = g.attrs[k]
    return out


def get_wcs(header_str):
    """Reconstruct the astropy WCS from a stored header string."""
    from astropy.io import fits
    from astropy.wcs import WCS
    return WCS(fits.Header.fromstring(header_str))


# ----------------------------------------------------------------------
# Fetching cutouts (shared by the bulk MPI pipeline and the photometry
# pipeline's -make_cats self-healing)
# ----------------------------------------------------------------------

def parse_cutout_fits(hdul, want_invvar, want_maskbits):
    """Extract image/invvar/mask arrays + primary header string from an open
    HDUList (viewer or container output), keying on IMAGETYP with positional
    fallback."""
    image = invvar = mask = None
    for i, hdu in enumerate(hdul):
        itype = str(hdu.header.get("IMAGETYP", "")).strip().upper()
        if i == 0 or itype == "IMAGE":
            image = np.asarray(hdu.data, dtype=np.float32)
        elif itype == "INVVAR":
            invvar = np.asarray(hdu.data, dtype=np.float32)
        elif itype == "MASKBITS":
            mask = np.asarray(hdu.data)
        elif itype == "":
            # no IMAGETYP card: fall back on HDU order (IMAGE, [INVVAR], [MASKBITS])
            if want_invvar and invvar is None:
                invvar = np.asarray(hdu.data, dtype=np.float32)
            elif want_maskbits and mask is None:
                mask = np.asarray(hdu.data)
    if image is None:
        raise RuntimeError("cutout FITS contained no image HDU")
    if want_invvar and invvar is None:
        raise RuntimeError("cutout FITS missing INVVAR HDU")
    if want_maskbits and mask is None:
        raise RuntimeError("cutout FITS missing MASKBITS HDU")
    header_str = hdul[0].header.tostring()
    return image, invvar, mask, header_str


def make_cutout_record(targetid, ra, dec, size, bands,
                       image, invvar, mask, header_str, fetch_method):
    """Build (and shape-validate) a store record for write_cutouts_batch."""
    expected = (len(bands), size, size)
    if image.shape != expected:
        raise RuntimeError("image shape {} != expected {}".format(image.shape, expected))
    return {
        "targetid": targetid,
        "ra": ra,
        "dec": dec,
        "box_size": size,
        "image": image,
        "invvar": invvar,
        "mask": mask,
        "header": header_str,
        "fetch_method": fetch_method,
    }


def fetch_cutout_url(ra, dec, size, targetid,
                     layer="ls-dr9", pixscale=0.262, bands=("g", "r", "z"),
                     invvar=True, maskbits=True,
                     url_base=PRODUCTION_VIEWER, timeout=120):
    """Fetch one cutout from the Legacy Surveys viewer and return a store
    record. Raises on any failure (HTTP error, timeout, missing HDU,
    shape mismatch)."""
    import urllib.request
    from astropy.io import fits

    url = ("{base}/cutout.fits?ra={ra}&dec={dec}&width={s}&height={s}"
           "&layer={layer}&pixscale={pixscale}&bands={bands}").format(
        base=url_base, ra=ra, dec=dec, s=int(size),
        layer=layer, pixscale=pixscale, bands="".join(bands))
    if invvar:
        url += "&invvar"
    if maskbits:
        url += "&maskbits"

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read()
    with fits.open(io.BytesIO(payload), memmap=False) as hdul:
        image, iv, mask, header_str = parse_cutout_fits(hdul, invvar, maskbits)

    return make_cutout_record(int(targetid), float(ra), float(dec), int(size),
                              bands, image, iv, mask, header_str, "url")
