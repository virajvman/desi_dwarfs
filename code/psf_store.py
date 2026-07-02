"""
psf_store.py

Single point of access for the per-brick HDF5 *empirical-PSF* store.

This is the SCARLET-pipeline sibling of cutout_store.py. The image cutout store
(cutout_store.py) carries only a scalar psfsize FWHM in its attrs; SCARLET needs
the full empirical coadd-PSF *image* per object. Those PSFs are built once by a
prepass job (job_scripts/image_cutouts/psf/many_psfs_general.py) and read back
here by the fitter -- the fit itself never touches the network.

Layout (one file per Legacy Surveys brick), deliberately separate from the
image store so the (expensive, frozen) cutout store is never mutated:

    {psfs_dir}/{BRICKNAME[:3]}/{BRICKNAME}.h5
        /{TARGETID}/psf       (3, B, B) float32   grz empirical coadd PSF,
                                                  standardized: centered, zero-
                                                  padded to a common odd box B
                                                  (PSF_BOX), unit-normalized per
                                                  band; an all-NaN plane marks a
                                                  band with no CCD coverage.
        /{TARGETID}.attrs:
            ra, dec        catalog position the PSF was sampled at (= cutout center)
            layer          Legacy Surveys layer (default ls-dr9)
            bands          string of bands actually covered, e.g. "grz" or "gr"
            fetch_method   'container' or 'url'
            created        UTC timestamp string
            native_size_{g,r,z}  raw coadd PSF stamp size before padding (int, 0 if absent)
            n_ccds_{g,r,z}       number of CCDs coadded in that band (int)

A rebuildable CSV manifest (psf_manifest.csv) caches shard contents for fast
existence checks; shards remain ground truth.

Writers must guarantee a single writer per shard (the bulk MPI pipeline
partitions work by brick, so each shard is owned by exactly one rank).
Concurrent read-only opens are safe; set HDF5_USE_FILE_LOCKING=FALSE on Lustre.

This module must stay importable under the Python 3.8 inside the
dstndstn/cutouts Shifter container: import only numpy at module level and keep
astropy/h5py imports inside functions.
"""

import os
import io
import csv
import time
import shutil

import numpy as np

# Canonical store location: a sibling of the image store, NOT inside it. The
# cutout store is the foundational, expensive-to-rebuild asset; the PSF store is
# a network-derived, occasionally-incomplete product, so it lives on its own.
PSFS_BASE = "/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts"
PSFS_DIR = os.path.join(PSFS_BASE, "psfs")

# Production viewer for the URL fallback / validation (NOT viewer-dev).
PRODUCTION_VIEWER = "https://www.legacysurvey.org/viewer"

# Cumulative manifest of objects whose PSF can never be built (off-footprint).
FAILED_MANIFEST = "permanently_failed.csv"

# Rebuildable cache of shard contents (shards remain ground truth).
MANIFEST_NAME = "psf_manifest.csv"
MANIFEST_FIELDS = (
    "brickname", "targetid", "bands", "fetch_method", "created",
)
DEFAULT_LAYER = "ls-dr9"
DEFAULT_BANDS = ("g", "r", "z")

# Common standardized PSF box (odd). Empirical coadd PSF stamps are ~25-63 px at
# the native 0.262"/px scale; 63 holds the widest DECam PsfEx basis without
# clipping the wings. The pixel scale is invariant across stamps, so padding a
# smaller stamp into this box is exact (see DESIGN.md / scarlet PSF discussion).
PSF_BOX = 63


def get_store_dir():
    """Return the PSF store directory."""
    return PSFS_DIR


def load_tombstones(psfs_dir):
    """Return set of TARGETIDs recorded as permanently un-buildable."""
    path = os.path.join(psfs_dir, FAILED_MANIFEST)
    tombs = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    tombs.add(int(row["targetid"]))
                except (KeyError, ValueError):
                    continue
    return tombs


def shard_path(psfs_dir, brickname):
    return os.path.join(psfs_dir, str(brickname)[:3], "{}.h5".format(brickname))


def _truthy(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "t")


def _manifest_row_from_group(brickname, targetid, group):
    return {
        "brickname": str(brickname),
        "targetid": int(targetid),
        "bands": str(group.attrs.get("bands", "")),
        "fetch_method": str(group.attrs.get("fetch_method", "")),
        "created": str(group.attrs.get("created", "")),
    }


def _normalize_manifest_row(row):
    out = {k: row.get(k, "") for k in MANIFEST_FIELDS}
    out["targetid"] = int(out["targetid"])
    return out


def _manifest_rows_to_index(rows):
    index = {}
    for row in rows:
        row = _normalize_manifest_row(row)
        index.setdefault(row["brickname"], {})[row["targetid"]] = row
    return index


def _manifest_row_from_record(brickname, rec, created):
    return {
        "brickname": str(brickname),
        "targetid": int(rec["targetid"]),
        "bands": str(rec.get("bands", "")),
        "fetch_method": str(rec.get("fetch_method", "container")),
        "created": created,
    }


def psf_satisfies_request(row, require_bands=None):
    """True if a manifest row already covers a PSF request.

    Presence is enough by default (one PSF per object). If require_bands is given
    (a string/iterable of band letters), every requested band must be present in
    the stored `bands` string -- this lets a rerun re-build objects that were
    only partially covered (e.g. fell off one band's footprint).
    """
    if row is None:
        return False
    if require_bands:
        have = set(str(row.get("bands", "")))
        return set(require_bands).issubset(have)
    return True


def manifest_path(psfs_dir):
    return os.path.join(psfs_dir, MANIFEST_NAME)


def manifest_delta_path(psfs_dir, tag):
    return os.path.join(psfs_dir, "manifest_{}.csv".format(tag))


def list_shard_manifest(psfs_dir, brickname):
    """{TARGETID: manifest row dict} for one brick's shard."""
    import h5py
    path = shard_path(psfs_dir, brickname)
    if not os.path.exists(path):
        return {}
    try:
        with h5py.File(path, "r") as f:
            return {int(k): _manifest_row_from_group(brickname, k, f[k])
                    for k in f.keys()}
    except (OSError, ValueError):
        return {}


def psf_exists(psfs_dir, brickname, targetid):
    """True if the object's PSF is in the store."""
    import h5py
    path = shard_path(psfs_dir, brickname)
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return str(int(targetid)) in f
    except OSError:
        return False


def _scan_shard(path, quarantine_corrupt=True):
    """Scan one shard and return manifest rows. Unreadable shards are quarantined."""
    import h5py

    brick = os.path.splitext(os.path.basename(path))[0]
    try:
        with h5py.File(path, "r") as f:
            return [_manifest_row_from_group(brick, k, f[k]) for k in f.keys()]
    except (OSError, ValueError) as exc:
        print("WARNING: unreadable shard {} ({}); treating as absent".format(path, exc),
              flush=True)
        if quarantine_corrupt:
            try:
                os.replace(path, path + ".corrupt")
            except OSError:
                pass
        return []


def _iter_shard_paths(psfs_dir):
    from glob import glob
    if not os.path.isdir(psfs_dir):
        return []
    return sorted(glob(os.path.join(psfs_dir, "*", "*.h5")))


def _scan_shards_parallel(paths, nproc, quarantine_corrupt=True):
    if not paths:
        return []
    if nproc is None or int(nproc) <= 1:
        rows = []
        for path in paths:
            rows.extend(_scan_shard(path, quarantine_corrupt=quarantine_corrupt))
        return rows

    import multiprocessing
    from functools import partial

    worker = partial(_scan_shard, quarantine_corrupt=quarantine_corrupt)
    if "NERSC_HOST" in os.environ:
        pool = multiprocessing.get_context("spawn").Pool(processes=int(nproc))
    else:
        pool = multiprocessing.Pool(processes=int(nproc))
    with pool:
        parts = pool.map(worker, paths, chunksize=max(1, len(paths) // (int(nproc) * 4)))
    rows = []
    for part in parts:
        rows.extend(part)
    return rows


def _write_manifest_atomic(psfs_dir, rows):
    os.makedirs(psfs_dir, exist_ok=True)
    final = manifest_path(psfs_dir)
    tmp = final + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({
                "brickname": row["brickname"],
                "targetid": int(row["targetid"]),
                "bands": row.get("bands", ""),
                "fetch_method": row.get("fetch_method", ""),
                "created": row.get("created", ""),
            })
    os.replace(tmp, final)


def build_manifest(psfs_dir, nproc=None):
    """Rebuild psf_manifest.csv from shards (parallel scan when nproc > 1)."""
    paths = _iter_shard_paths(psfs_dir)
    print("Building {} from {} shards ...".format(MANIFEST_NAME, len(paths)), flush=True)
    t0 = time.time()
    rows = _scan_shards_parallel(paths, nproc, quarantine_corrupt=True)
    _write_manifest_atomic(psfs_dir, rows)
    print("  {} manifest rows written ({:.1f}s)".format(len(rows), time.time() - t0),
          flush=True)
    return _manifest_rows_to_index(rows)


def load_manifest(psfs_dir, nproc=None, bootstrap=True):
    """Return {brickname: {targetid: row}}; bootstrap via build_manifest if absent."""
    path = manifest_path(psfs_dir)
    if not os.path.exists(path):
        if bootstrap:
            return build_manifest(psfs_dir, nproc=nproc)
        return {}
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return _manifest_rows_to_index(rows)


def append_manifest_delta(psfs_dir, tag, rows):
    """Append manifest rows to manifest_{tag}.csv (per-rank/per-job delta file)."""
    if not rows:
        return
    path = manifest_delta_path(psfs_dir, tag)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({
                "brickname": row["brickname"],
                "targetid": int(row["targetid"]),
                "bands": row.get("bands", ""),
                "fetch_method": row.get("fetch_method", ""),
                "created": row.get("created", ""),
            })


def _manifest_row_better(new_row, old_row):
    """True if new_row should replace old_row for the same (brick, targetid).

    Prefer more bands; tie-break on the more recent timestamp.
    """
    new_n = len(str(new_row.get("bands", "")))
    old_n = len(str(old_row.get("bands", "")))
    if new_n != old_n:
        return new_n > old_n
    return str(new_row.get("created", "")) >= str(old_row.get("created", ""))


def merge_manifest_deltas(psfs_dir):
    """Fold manifest_*.csv deltas into psf_manifest.csv; delete delta files."""
    from glob import glob

    delta_files = sorted(glob(os.path.join(psfs_dir, "manifest_*.csv")))
    if not delta_files:
        return

    merged = {}
    manifest = manifest_path(psfs_dir)
    if os.path.exists(manifest):
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                row = _normalize_manifest_row(row)
                merged[(row["brickname"], row["targetid"])] = row

    n_new = 0
    for dpath in delta_files:
        with open(dpath, newline="") as f:
            for row in csv.DictReader(f):
                row = _normalize_manifest_row(row)
                key = (row["brickname"], row["targetid"])
                if key not in merged or _manifest_row_better(row, merged[key]):
                    if key not in merged:
                        n_new += 1
                    merged[key] = row

    rows = [merged[k] for k in sorted(merged)]
    _write_manifest_atomic(psfs_dir, rows)

    for dpath in delta_files:
        os.remove(dpath)

    print("Manifest merge: {} total rows ({} new from deltas)".format(
        len(rows), n_new), flush=True)


# ----------------------------------------------------------------------
# PSF standardization (shared by the builder and the URL fallback so both
# land in the SAME representation -- essential for a fair URL cross-check)
# ----------------------------------------------------------------------

def standardize_psf(psf2d, box=PSF_BOX):
    """Center a single-band PSF stamp into a box x box array and unit-normalize.

    The empirical coadd PSF stamps are odd-sized and centered on the central
    pixel at the native 0.262"/px scale; the pixel scale is invariant across
    stamps, so zero-padding into a common box is exact (it only adds sky where
    the PSF is ~0). A stamp larger than `box` is centrally cropped (logged by
    the caller); that only ever discards far-wing flux.
    """
    a = np.asarray(psf2d, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("PSF stamp must be 2D, got shape {}".format(a.shape))
    h, w = a.shape
    if h > box or w > box:
        sy = max(0, (h - box) // 2)
        sx = max(0, (w - box) // 2)
        a = a[sy:sy + min(h, box), sx:sx + min(w, box)]
        h, w = a.shape
    out = np.zeros((box, box), dtype=np.float64)
    y0 = (box - h) // 2
    x0 = (box - w) // 2
    out[y0:y0 + h, x0:x0 + w] = a
    s = out.sum()
    if s != 0 and np.isfinite(s):
        out /= s
    return out.astype(np.float32)


def standardize_psf_cube(coadd_by_band, bands=DEFAULT_BANDS, box=PSF_BOX):
    """Stack per-band raw coadd PSFs into a standardized (nbands, box, box) cube.

    coadd_by_band: {band: 2D ndarray or None}. A band mapped to None (no CCD
    coverage) becomes an all-NaN plane so downstream code can detect it.
    """
    cube = np.full((len(bands), box, box), np.nan, dtype=np.float32)
    for i, b in enumerate(bands):
        arr = coadd_by_band.get(b)
        if arr is not None:
            cube[i] = standardize_psf(arr, box=box)
    return cube


# ----------------------------------------------------------------------
# Writing / reading the store
# ----------------------------------------------------------------------

def write_psfs_batch(psfs_dir, brickname, records,
                     compression="gzip", compression_opts=4):
    """Atomically add `records` to the shard for `brickname`.

    records: iterable of dicts with keys
        targetid (int), ra, dec, psf (ndarray (nbands, B, B)), bands (str),
        fetch_method, and optionally layer, native_size (dict band->int),
        n_ccds (dict band->int).

    Returns manifest row dicts for the written records (empty list if none).

    The existing shard (if any) is copied to a .tmp file, groups are added
    (replacing any stale group with the same TARGETID), and the .tmp is renamed
    over the original -- a killed job never leaves a half-written shard.
    """
    import h5py

    records = list(records)
    if not records:
        return []

    final = shard_path(psfs_dir, brickname)
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
            g.create_dataset("psf", data=np.asarray(rec["psf"], dtype=np.float32),
                             compression=compression, compression_opts=compression_opts)
            g.attrs["ra"] = float(rec["ra"])
            g.attrs["dec"] = float(rec["dec"])
            g.attrs["layer"] = str(rec.get("layer", DEFAULT_LAYER))
            g.attrs["bands"] = str(rec.get("bands", ""))
            g.attrs["fetch_method"] = rec.get("fetch_method", "container")
            g.attrs["created"] = created
            native = rec.get("native_size", {}) or {}
            nccd = rec.get("n_ccds", {}) or {}
            for band in DEFAULT_BANDS:
                g.attrs["native_size_{}".format(band)] = int(native.get(band, 0))
                g.attrs["n_ccds_{}".format(band)] = int(nccd.get(band, 0))
            manifest_rows.append(_manifest_row_from_record(brickname, rec, created))

    os.replace(tmp, final)
    return manifest_rows


def read_psf(psfs_dir, brickname, targetid):
    """Read one object's PSF. Returns dict with `psf` (nbands, B, B) and attrs."""
    import h5py

    path = shard_path(psfs_dir, brickname)
    with h5py.File(path, "r") as f:
        key = str(int(targetid))
        if key not in f:
            raise KeyError("TARGETID {} not in PSF shard {}".format(targetid, path))
        g = f[key]
        out = {"psf": g["psf"][:]}
        for k in g.attrs.keys():
            out[k] = g.attrs[k]
    return out


# ----------------------------------------------------------------------
# URL fallback / validation: fetch the empirical coadd PSF from the viewer.
# ----------------------------------------------------------------------

def fetch_coadd_psf_url(ra, dec, bands=DEFAULT_BANDS, layer=DEFAULT_LAYER,
                        url_base=PRODUCTION_VIEWER, timeout=120):
    """Fetch the empirical coadd PSF from the Legacy Surveys viewer.

    Returns {band: 2D ndarray} with the RAW (un-standardized) per-band stamp.
    Bands the viewer did not return (no coverage) are absent from the dict.
    Raises on HTTP error / timeout / unparseable response.

    The viewer writes one HDU per covered band, tagging each with a 'BAND' card
    (and BANDS/BAND%i in the primary header); we key off those rather than HDU
    order, since a partially-covered position drops bands and would otherwise
    misalign with a positional [0]=g,[1]=r,[2]=z assumption.

    Uses `requests` (not `urllib`) deliberately: requests verifies TLS against
    its own bundled `certifi` CA store rather than the OS/container's system
    trust store, which sidesteps containers (e.g. dstndstn/cutouts on NERSC
    shifter) whose system CA bundle is stale/missing --
    urllib.request.urlopen hit CERTIFICATE_VERIFY_FAILED there while this
    exact requests.Session().get(url) pattern (from the scarlet_photo.py
    prototype's fetch_psf) is confirmed working in the same environment.
    """
    import requests
    from astropy.io import fits

    url = ("{base}/coadd-psf/?ra={ra}&dec={dec}&layer={layer}&bands={bands}").format(
        base=url_base, ra=ra, dec=dec, layer=layer, bands="".join(bands))

    with requests.Session() as session:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.content

    out = {}
    with fits.open(io.BytesIO(payload), memmap=False) as hdul:
        # Primary header carries BANDS / BAND%i; each HDU carries its own BAND.
        primary_bands = str(hdul[0].header.get("BANDS", "")) if len(hdul) else ""
        for i, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            band = hdu.header.get("BAND", None)
            if band is None:
                # Fall back to BANDS-string order for this HDU index.
                band = primary_bands[i] if i < len(primary_bands) else None
            if band is None:
                continue
            out[str(band).strip()] = np.asarray(hdu.data, dtype=np.float64)
    return out
