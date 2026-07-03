#!/usr/bin/python3

"""
many_psfs_general.py

MPI + multiprocessing prepass that builds the EMPIRICAL Legacy Surveys coadd PSF
per object and writes per-brick HDF5 shards (see code/psf_store.py for the store
layout). It is the SCARLET-pipeline sibling of many_cutouts_general.py: same
by-brick partitioning, atomic shard writes, manifest skip, and Shifter/srun
launch -- but instead of an image cutout it stores a standardized (3, B, B)
empirical PSF cube.

Why a separate prepass (not in the fitter):
* The empirical coadd PSF is network/CFS-derived and occasionally fails
  (off-footprint, missing calibs). Isolating it into a re-runnable batch keeps
  the scarlet fit a pure, offline read of a cached PSF.

How the PSF is built (inside the dstndstn/cutouts container):
* build_coadd_psf() reproduces the Legacy Surveys viewer 'coadd-psf' endpoint
  (imagine/map/views.py::exposures_common, copsf branch) using the INSTALLED
  imagine/legacypipe/tractor libraries -- it does NOT modify them. For each CCD
  touching the position it renders the (normalized) PsfEx model at the sub-image
  center and inverse-variance-weights it, exactly as the viewer does. The result
  matches the URL output; `--validate-url` proves it on a sample.
* The container has CFS access, so the build needs no network. A per-object URL
  fallback (the same viewer over HTTP) covers the rare container failure.

The model frame in scarlet uses a narrow Gaussian PSF; THIS file produces the
*observation* PSF (the empirical one we deconvolve from).

Usage:
  # Validation (login or compute node, no MPI): compare container vs URL.
  shifter --image dstndstn/cutouts:dvsro3 python3 many_psfs_general.py \
      --catalog-path /path/cat.fits --outdir-data /path/psfs \
      --validate-url 30 --validate-outdir /path/psf_validation --nompi

  # Production: see get_psfs_general.sbatch / psfs_cnn_general.sh

Modeled on many_cutouts_general.py (Viraj Manwadkar) and the imagine viewer
coadd-psf endpoint (Dustin Lang).
"""

import os
import csv
import sys
import time
import signal
import multiprocessing
from collections import defaultdict

import numpy as np

# repo layout: job_scripts/image_cutouts/psf/ -> repo root -> code/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..', 'code'))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import psf_store
from psf_store import (FAILED_MANIFEST, PRODUCTION_VIEWER, DEFAULT_BANDS, PSF_BOX,
                       standardize_psf, standardize_psf_cube, fetch_coadd_psf_url,
                       load_tombstones, load_manifest, append_manifest_delta,
                       merge_manifest_deltas, psf_satisfies_request)


def weighted_partition(weights, n):
    '''Partition `weights` into `n` groups with approximately equal sums.

    Returns list of lists of indices for each group; non-contiguous items may be
    grouped together for better balancing. (Copied from many_cutouts_general.)
    '''
    sumweights = np.zeros(n, dtype=float)
    groups = [list() for _ in range(n)]
    weights = np.asarray(weights)
    for i in np.argsort(-weights):
        j = np.argmin(sumweights)
        groups[j].append(i)
        sumweights[j] += weights[i]
    assert len(groups) == n
    return groups


# ----------------------------------------------------------------------
# Empirical coadd-PSF construction (reproduces imagine coadd_psf)
# ----------------------------------------------------------------------

# Half-size in coadd pixels used by the viewer to define the CCD-selection box
# (imagine/map/views.py: size=32 -> 64x64 box at 0.262"/px). The output PSF
# stamp size is set by the PsfEx basis, NOT by this box.
COADD_PSF_HALFSIZE = 32


def _combine_band_psfs(items):
    """Inverse-variance-weighted coadd of per-CCD PSF stamps for one band.

    items: list of (psf2d ndarray, iv float). Stamps are centered; we pad each to
    the common (odd) max size before summing -- identical to the viewer's direct
    `sumpsf += psfimg*iv` when all stamps share a size (the normal case, since
    one band maps to one camera in a region), and robust when they differ.

    Returns (raw_coadd 2d or None, native_size int, n_used int).
    """
    if not items:
        return None, 0, 0
    hmax = max(p.shape[0] for p, _ in items)
    wmax = max(p.shape[1] for p, _ in items)
    if hmax % 2 == 0:
        hmax += 1
    if wmax % 2 == 0:
        wmax += 1
    acc = np.zeros((hmax, wmax), dtype=np.float64)
    sumiv = 0.0
    for p, iv in items:
        h, w = p.shape
        y0 = (hmax - h) // 2
        x0 = (wmax - w) // 2
        acc[y0:y0 + h, x0:x0 + w] += np.asarray(p, dtype=np.float64) * iv
        sumiv += iv
    if sumiv <= 0:
        return None, hmax, len(items)
    return acc / sumiv, hmax, len(items)


def build_coadd_psf(ra, dec, bands=DEFAULT_BANDS, layer_name='ls-dr9',
                    halfsize=COADD_PSF_HALFSIZE):
    """Reproduce the Legacy Surveys viewer coadd-psf at (ra, dec).

    Returns (coadd_by_band, native_size, n_ccds) where coadd_by_band maps each
    band to its RAW (un-standardized, un-renormalized) inverse-variance-weighted
    coadd PSF (or None if no CCD covered it). Mirrors imagine exposures_common's
    copsf branch step-for-step; only runs inside the container (imagine/legacypipe
    on the path, CFS mounted).
    """
    from astrometry.util.util import Tan
    from map.views import get_layer, touchup_ccds

    layer = get_layer(layer_name)
    try:
        survey = layer.survey
    except Exception:
        # Split layer (ls-dr9): pick the north/south sub-layer for this position.
        layer = layer.get_layer_for_radec(ra, dec)
        survey = layer.survey

    avail = list(layer.get_bands())
    bands = [b for b in bands if b in avail]
    empty = ({b: None for b in bands}, {b: 0 for b in bands}, {b: 0 for b in bands})
    if not bands:
        return empty

    # CCD-selection box (same geometry as the viewer).
    pixscale = 0.262 / 3600.
    W = H = halfsize * 2
    wcs = Tan(*[float(x) for x in [
        ra, dec, halfsize + 0.5, halfsize + 0.5, -pixscale, 0., 0., pixscale, W, H]])
    nil, north = wcs.pixelxy2radec(halfsize + 0.5, H)
    nil, south = wcs.pixelxy2radec(halfsize + 0.5, 1)
    west, nil = wcs.pixelxy2radec(1, halfsize + 0.5)
    east, nil = wcs.pixelxy2radec(W, halfsize + 0.5)

    CCDs = layer.ccds_touching_box(north, south, east, west)
    if CCDs is None or len(CCDs) == 0:
        return empty
    CCDs = touchup_ccds(CCDs, survey)
    if 'ccd_cuts' in CCDs.get_columns():
        CCDs.cut(CCDs.ccd_cuts == 0)
    if len(CCDs) == 0:
        return empty
    CCDs.cut(np.isin(CCDs.filter, list(layer.get_bands())))
    if len(CCDs) == 0:
        return empty
    CCDs = CCDs[np.array([f in bands for f in CCDs.filter])]
    if len(CCDs) == 0:
        return empty

    items = defaultdict(list)  # band -> list of (psfimg, iv)
    for ccd in CCDs:
        try:
            im = survey.get_image_object(ccd)
            imwcs = im.get_wcs()
            ok, cx, cy = imwcs.radec2pixelxy([east, west, west, east],
                                             [north, north, south, south])
            Hc, Wc = im.shape
            x0 = int(np.clip(np.floor(min(cx)), 0, Wc - 1))
            x1 = int(np.clip(np.ceil(max(cx)), 0, Wc - 1))
            y0 = int(np.clip(np.floor(min(cy)), 0, Hc - 1))
            y1 = int(np.clip(np.ceil(max(cy)), 0, Hc - 1))
            if x0 == x1 or y0 == y1:
                continue
            slc = (slice(y0, y1 + 1), slice(x0, x1 + 1))
            tim = im.get_tractor_image(slc, pixPsf=True, nanomaggies=False,
                                       readsky=False, subsky=False, pixels=False,
                                       dq=False, normalizePsf=True,
                                       old_calibs_ok=True)
            if tim is None:
                continue
            psf = tim.getPsf()
            th, tw = tim.shape
            psfimg = psf.getImage(tw / 2., th / 2.)
            ivdata = tim.getInvvar()
            band = tim.band
            if band not in bands:
                continue
            if ivdata is None or np.all(ivdata == 0):
                continue
            iv = float(np.median(ivdata[ivdata > 0]))
            items[band].append((np.asarray(psfimg, dtype=np.float64), iv))
        except Exception as exc:
            print('  WARNING: skipping a CCD for ({:.5f},{:.5f}): {!r}'.format(
                ra, dec, exc), flush=True)
            continue

    coadd, native, nccd = {}, {}, {}
    for b in bands:
        c, nsize, nused = _combine_band_psfs(items.get(b, []))
        coadd[b] = c
        native[b] = int(nsize)
        nccd[b] = int(nused)
    return coadd, native, nccd


# ----------------------------------------------------------------------
# Worker-side building
# ----------------------------------------------------------------------

class _PsfTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _PsfTimeout("coadd-psf build timed out")


def _record_from_coadd(task, coadd, native, nccd, fetch_method):
    """Standardize a per-band raw coadd dict into a store record (or None if no
    band was covered).

    The cube is ALWAYS a fixed DEFAULT_BANDS (g,r,z) 3-plane array regardless of
    --bands, so the stored plane order is invariant; a band not requested (or not
    covered) is an all-NaN plane. --bands only narrows which CCDs are coadded.
    """
    covered = "".join(b for b in DEFAULT_BANDS if coadd.get(b) is not None)
    if not covered:
        return None
    # Warn if a native stamp exceeds the box (would be center-cropped).
    for b in DEFAULT_BANDS:
        if native.get(b, 0) > task['box']:
            print('  WARNING: native PSF {}px > box {}px for ({:.5f},{:.5f}) band {}'
                  ' -- wings will be cropped'.format(native[b], task['box'],
                  task['ra'], task['dec'], b), flush=True)
    cube = standardize_psf_cube(coadd, bands=DEFAULT_BANDS, box=task['box'])
    return {
        'targetid': task['targetid'],
        'ra': task['ra'],
        'dec': task['dec'],
        'psf': cube,
        'bands': covered,
        'fetch_method': fetch_method,
        'layer': task['layer'],
        'native_size': native,
        'n_ccds': nccd,
    }


def _build_container(task):
    """Build the coadd PSF via the container (CFS reads); returns a store record."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(task['timeout'])
    try:
        coadd, native, nccd = build_coadd_psf(
            task['ra'], task['dec'], bands=task['bands'],
            layer_name=task['layer'], halfsize=task['halfsize'])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    rec = _record_from_coadd(task, coadd, native, nccd, 'container')
    if rec is None:
        raise RuntimeError('no CCDs overlapping (container)')
    return rec


def _build_url(task):
    """One-shot fallback: fetch the coadd PSF from the production viewer."""
    raw = fetch_coadd_psf_url(task['ra'], task['dec'], bands=task['bands'],
                              layer=task['layer'], url_base=task['url_base'],
                              timeout=task['url_timeout'])
    coadd = {b: raw.get(b) for b in task['bands']}
    native = {b: (coadd[b].shape[0] if coadd[b] is not None else 0) for b in task['bands']}
    nccd = {b: 0 for b in task['bands']}  # unknown from the URL
    rec = _record_from_coadd(task, coadd, native, nccd, 'url')
    if rec is None:
        raise RuntimeError('no CCDs overlapping (url)')
    return rec


def _build_one_safe(task):
    """Worker entry. Returns ('ok', brick, record) or ('fail', brick, failrec).

    Container timeouts are retried up to max_retries; any other container error
    is deterministic and skips straight to the URL fallback.
    """
    brick = task['brick']

    if task['dry_run']:
        print('Rank {}, object {}: ra={} dec={} brick={} layer={}'.format(
            task['rank'], task['targetid'], task['ra'], task['dec'],
            brick, task['layer']), flush=True)
        return ('ok', brick, None)

    errors = []
    for attempt in range(task['max_retries'] + 1):
        try:
            return ('ok', brick, _build_container(task))
        except _PsfTimeout:
            errors.append('container timeout (attempt {})'.format(attempt + 1))
            print('Rank {}: TIMEOUT on {} (attempt {}/{})'.format(
                task['rank'], task['targetid'], attempt + 1,
                task['max_retries'] + 1), flush=True)
        except Exception as exc:
            errors.append('container: {!r}'.format(exc))
            print('Rank {}: ERROR on {}: {!r}'.format(
                task['rank'], task['targetid'], exc), flush=True)
            break  # deterministic -- retrying the container is pointless

    if task['url_fallback']:
        try:
            return ('ok', brick, _build_url(task))
        except Exception as exc:
            errors.append('url: {!r}'.format(exc))
            print('Rank {}: URL fallback failed on {}: {!r}'.format(
                task['rank'], task['targetid'], exc), flush=True)

    return ('fail', brick, {
        'targetid': task['targetid'], 'ra': task['ra'], 'dec': task['dec'],
        'brickname': brick, 'reason': ' | '.join(errors),
    })


def _make_task(args, bands, tgid, ra, dec, brick, rank):
    return {
        'targetid': int(tgid), 'ra': float(ra), 'dec': float(dec),
        'brick': brick, 'bands': bands, 'layer': args.layer, 'box': args.box,
        'halfsize': args.halfsize, 'timeout': args.timeout,
        'max_retries': args.max_retries, 'url_fallback': args.url_fallback,
        'url_base': args.url_base, 'url_timeout': args.url_timeout,
        'dry_run': args.dry_run, 'rank': rank,
    }


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------

def _read_catalog(args):
    from astropy.table import Table
    cat = Table.read(args.catalog_path)
    for col, name in ((args.ra_col, '--ra-col'), (args.dec_col, '--dec-col'),
                      (args.id_col, '--id-col'), (args.brick_col, '--brick-col')):
        if col not in cat.colnames:
            sys.exit("ERROR: column '{}' ({}) not found in {}. Available: {}".format(
                col, name, args.catalog_path, ', '.join(cat.colnames[:40])))
    return cat


def plan(args, outdir_data):
    """Rank-0 planning: decide what needs building and partition by brick."""
    cat = _read_catalog(args)
    n = len(cat)
    print('Total objects in catalog: {}'.format(n), flush=True)

    allra = np.asarray(cat[args.ra_col], dtype=np.float64)
    alldec = np.asarray(cat[args.dec_col], dtype=np.float64)
    alltgid = np.asarray(cat[args.id_col], dtype=np.int64)
    allbrick = np.asarray(cat[args.brick_col]).astype(str)

    tombs = set()
    if not args.retry_failed:
        tombs = load_tombstones(outdir_data)
        if tombs:
            print('Excluding {} tombstoned objects from {} '
                  '(--retry-failed to re-attempt)'.format(len(tombs), FAILED_MANIFEST),
                  flush=True)

    print('Loading PSF manifest from {} ...'.format(outdir_data), flush=True)
    t0 = time.time()
    existing = load_manifest(outdir_data, nproc=args.manifest_nproc,
                             bootstrap=not args.rebuild_manifest)
    n_existing = sum(len(v) for v in existing.values())
    print('  {} objects in manifest covering {} bricks ({:.1f}s)'.format(
        n_existing, len(existing), time.time() - t0), flush=True)

    require_bands = args.require_bands if args.require_bands else None
    seen = set()
    need = []
    for k in range(n):
        tgid = int(alltgid[k])
        if tgid in seen:
            continue
        seen.add(tgid)
        if tgid in tombs:
            continue
        row = existing.get(allbrick[k], {}).get(tgid)
        if psf_satisfies_request(row, require_bands=require_bands):
            continue
        need.append(k)
    need = np.asarray(need, dtype=np.int64)
    print('Need to build {}/{} PSFs'.format(len(need), n), flush=True)

    ra, dec = allra[need], alldec[need]
    tgid, brick = alltgid[need], allbrick[need]

    brick_names, brick_inverse = np.unique(brick, return_inverse=True)
    brick_rows = [np.flatnonzero(brick_inverse == i) for i in range(len(brick_names))]
    # weight by object count (PSF cost ~ #CCDs, unknown ahead -- count is a proxy)
    brick_weights = np.array([float(len(rows)) for rows in brick_rows])
    return brick_names, brick_rows, ra, dec, tgid, brick_weights


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def do_psfs(args, comm=None, outdir_data='.'):
    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    bands = tuple(b.strip() for b in args.bands.split(','))

    t0 = time.time()
    if rank == 0:
        os.makedirs(outdir_data, exist_ok=True)
        if args.rebuild_manifest:
            print('Rebuilding PSF manifest from shards ...', flush=True)
            psf_store.build_manifest(outdir_data, nproc=args.manifest_nproc)
        else:
            # Fold in any manifest deltas a previously-killed run left behind
            # (e.g. one that hit its wall-clock limit before the end-of-job
            # merge_manifest_deltas ran). Without this, the stale/empty
            # psf_manifest.csv masks shards already on disk and everything
            # rebuilds. Shards remain ground truth -- --rebuild-manifest rescans
            # them directly if the deltas were also lost.
            merge_manifest_deltas(outdir_data)
        brick_names, brick_rows, ra, dec, tgid, brick_weights = plan(args, outdir_data)
        groups = weighted_partition(brick_weights, size)
        print('Planning took {:.2f} sec'.format(time.time() - t0), flush=True)
    else:
        brick_names = brick_rows = ra = dec = tgid = groups = None

    if comm:
        brick_names = comm.bcast(brick_names, root=0)
        brick_rows = comm.bcast(brick_rows, root=0)
        ra = comm.bcast(ra, root=0)
        dec = comm.bcast(dec, root=0)
        tgid = comm.bcast(tgid, root=0)
        groups = comm.bcast(groups, root=0)

    if len(brick_names) == 0:
        if rank == 0:
            print('Nothing to do.', flush=True)
        if comm is not None:
            comm.barrier()
        if rank == 0 and not args.dry_run:
            merge_manifest_deltas(outdir_data)
            _merge_failures(outdir_data)
        return

    my_bricks = groups[rank]
    tasks = []
    expected = {}
    for bi in my_bricks:
        bname = brick_names[bi]
        rows = brick_rows[bi]
        expected[bname] = len(rows)
        for r in rows:
            tasks.append(_make_task(args, bands, tgid[r], ra[r], dec[r], bname, rank))

    total = len(tasks)
    print('Rank {}: assigned {} objects in {} bricks'.format(
        rank, total, len(my_bricks)), flush=True)

    buffers = defaultdict(list)
    settled = defaultdict(int)
    failed = []
    n_done = 0
    n_written = 0

    def handle_result(result):
        nonlocal n_done, n_written
        status, bname, payload = result
        n_done += 1
        settled[bname] += 1
        if status == 'ok':
            if payload is not None:
                buffers[bname].append(payload)
        else:
            failed.append(payload)
        if settled[bname] == expected[bname]:
            recs = buffers.pop(bname, [])
            if recs:
                manifest_rows = psf_store.write_psfs_batch(outdir_data, bname, recs)
                append_manifest_delta(outdir_data, 'rank{}'.format(rank), manifest_rows)
                n_written += 1
        if n_done % 100 == 0 or n_done == total:
            print('Rank {}: {}/{} done, {} failed, {} shards written, {:.0f}s'.format(
                rank, n_done, total, len(failed), n_written, time.time() - t0),
                flush=True)

    if total > 0 and args.mp > 1 and not args.dry_run:
        pool = multiprocessing.Pool(args.mp, maxtasksperchild=50)
        try:
            for result in pool.imap_unordered(_build_one_safe, tasks, chunksize=1):
                handle_result(result)
        finally:
            pool.close()
            pool.join()
    else:
        for task in tasks:
            handle_result(_build_one_safe(task))

    # flush anything left (only possible after a pool error)
    for bname in list(buffers.keys()):
        recs = buffers.pop(bname)
        if recs:
            print('Rank {}: WARNING flushing incomplete brick {}'.format(rank, bname),
                  flush=True)
            manifest_rows = psf_store.write_psfs_batch(outdir_data, bname, recs)
            append_manifest_delta(outdir_data, 'rank{}'.format(rank), manifest_rows)

    if failed:
        rank_manifest = os.path.join(outdir_data, 'failed_rank{}.csv'.format(rank))
        with open(rank_manifest, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['targetid', 'ra', 'dec', 'brickname', 'reason'])
            w.writeheader()
            w.writerows(failed)
        print('Rank {}: {} failures -> {}'.format(rank, len(failed), rank_manifest),
              flush=True)

    print('Rank {}: finished at {} ({}/{} succeeded)'.format(
        rank, time.asctime(), total - len(failed), total), flush=True)

    if comm is not None:
        comm.barrier()

    if rank == 0 and not args.dry_run:
        merge_manifest_deltas(outdir_data)
        _merge_failures(outdir_data)
        print('All ranks done at {}'.format(time.asctime()), flush=True)


def _merge_failures(outdir_data):
    """Merge per-rank failure files into the cumulative manifest (dedup by
    TARGETID, dropping rows for objects that made it into the store)."""
    from glob import glob

    rank_files = sorted(glob(os.path.join(outdir_data, 'failed_rank*.csv')))
    failed_path = os.path.join(outdir_data, FAILED_MANIFEST)

    rows = {}
    if os.path.exists(failed_path):
        with open(failed_path, newline='') as f:
            for row in csv.DictReader(f):
                rows[int(row['targetid'])] = row

    n_new = 0
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for rf in rank_files:
        with open(rf, newline='') as f:
            for row in csv.DictReader(f):
                tg = int(row['targetid'])
                if tg not in rows:
                    row['timestamp'] = stamp
                    rows[tg] = row
                    n_new += 1

    if rows:
        store_index = load_manifest(outdir_data, bootstrap=False)
        all_present = set()
        for brick_rows in store_index.values():
            all_present.update(brick_rows)
        rows = {tg: r for tg, r in rows.items() if tg not in all_present}

        with open(failed_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['targetid', 'ra', 'dec', 'brickname',
                                              'reason', 'timestamp'])
            w.writeheader()
            for tg in sorted(rows):
                w.writerow({k: rows[tg].get(k, '') for k in w.fieldnames})

    for rf in rank_files:
        os.remove(rf)

    print('Failure manifest: {} total tombstones ({} new this run)'.format(
        len(rows), n_new), flush=True)


# ----------------------------------------------------------------------
# Validation: container vs URL on a sample (manual gate before production)
# ----------------------------------------------------------------------

def _centroid(img):
    a = np.nan_to_num(np.asarray(img, dtype=np.float64), nan=0.0)
    s = a.sum()
    if s <= 0:
        return (np.nan, np.nan)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return (float((a * yy).sum() / s), float((a * xx).sum() / s))


def validate_url(args):
    """Compare the container build against the URL on a sample of objects.

    For each sampled object and band: standardize both PSFs identically and
    report max|delta|/peak, raw-flux ratio, and centroid shift; save a
    container/url/residual panel. Manual gate -- eyeball results before the
    production batch.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cat = _read_catalog(args)
    n = len(cat)
    bands = tuple(b.strip() for b in args.bands.split(','))

    # Evenly sample across the catalog (deterministic; spans the sky / regimes).
    nval = min(args.validate_url, n)
    idx = np.linspace(0, n - 1, nval).astype(int)
    idx = np.unique(idx)

    outdir = args.validate_outdir or os.path.join(
        os.path.dirname(os.path.abspath(args.catalog_path)), 'psf_validation')
    os.makedirs(outdir, exist_ok=True)
    print('Validating {} objects; writing panels to {}'.format(len(idx), outdir),
          flush=True)

    rows = []
    for k in idx:
        tgid = int(cat[args.id_col][k])
        ra = float(cat[args.ra_col][k])
        dec = float(cat[args.dec_col][k])
        try:
            coadd_c, native_c, nccd_c = build_coadd_psf(
                ra, dec, bands=bands, layer_name=args.layer, halfsize=args.halfsize)
        except Exception as exc:
            print('  {}: container build FAILED: {!r}'.format(tgid, exc), flush=True)
            continue
        try:
            raw_u = fetch_coadd_psf_url(ra, dec, bands=bands, layer=args.layer,
                                        url_base=args.url_base, timeout=args.url_timeout)
        except Exception as exc:
            print('  {}: URL fetch FAILED: {!r}'.format(tgid, exc), flush=True)
            continue

        fig, axes = plt.subplots(len(bands), 3, figsize=(9, 3 * len(bands)),
                                 squeeze=False)
        for bi, b in enumerate(bands):
            c_raw = coadd_c.get(b)
            u_raw = raw_u.get(b)
            if c_raw is None or u_raw is None:
                rows.append((tgid, b, np.nan, np.nan, np.nan,
                             c_raw is not None, u_raw is not None))
                for j in range(3):
                    axes[bi][j].axis('off')
                continue
            c_std = standardize_psf(c_raw, box=args.box)
            u_std = standardize_psf(u_raw, box=args.box)
            peak = float(np.nanmax(u_std)) or 1.0
            max_frac = float(np.nanmax(np.abs(c_std - u_std)) / peak)
            flux_ratio = float(np.sum(c_raw) / np.sum(u_raw)) if np.sum(u_raw) else np.nan
            cy_c, cx_c = _centroid(c_std)
            cy_u, cx_u = _centroid(u_std)
            cshift = float(np.hypot(cy_c - cy_u, cx_c - cx_u))
            rows.append((tgid, b, max_frac, flux_ratio, cshift, True, True))

            eps = 1e-12
            axes[bi][0].imshow(np.log10(np.abs(c_std) + eps), origin='lower')
            axes[bi][0].set_title('{} container'.format(b))
            axes[bi][1].imshow(np.log10(np.abs(u_std) + eps), origin='lower')
            axes[bi][1].set_title('{} url'.format(b))
            resid = c_std - u_std
            vlim = max(abs(resid.min()), abs(resid.max()), 1e-9)
            axes[bi][2].imshow(resid, origin='lower', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
            axes[bi][2].set_title('resid max|d|/peak={:.1e}'.format(max_frac))
            for j in range(3):
                axes[bi][j].set_xticks([])
                axes[bi][j].set_yticks([])
        fig.suptitle('TGID {} ra={:.4f} dec={:.4f}'.format(tgid, ra, dec))
        fig.savefig(os.path.join(outdir, 'psf_val_{}.png'.format(tgid)),
                    bbox_inches='tight')
        plt.close(fig)

    # Summary table + CSV.
    csv_path = os.path.join(outdir, 'psf_validation_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['targetid', 'band', 'max_abs_frac', 'flux_ratio', 'centroid_shift_px',
                    'container_has_band', 'url_has_band'])
        w.writerows(rows)

    valid = [r for r in rows if r[5] and r[6] and np.isfinite(r[2])]
    print('\n==== PSF validation summary ({} band-comparisons) ===='.format(len(valid)),
          flush=True)
    if valid:
        mf = np.array([r[2] for r in valid])
        fr = np.array([r[3] for r in valid])
        cs = np.array([r[4] for r in valid])
        print('  max|delta|/peak : median={:.2e}  max={:.2e}  (tol < 1e-3)'.format(
            np.nanmedian(mf), np.nanmax(mf)), flush=True)
        print('  flux ratio      : median={:.4f}  range=[{:.4f},{:.4f}]  (tol within 0.5%)'.format(
            np.nanmedian(fr), np.nanmin(fr), np.nanmax(fr)), flush=True)
        print('  centroid shift  : median={:.3f}  max={:.3f} px  (tol < 0.05)'.format(
            np.nanmedian(cs), np.nanmax(cs)), flush=True)
        n_bad = int(np.sum((mf >= 1e-3) | (np.abs(fr - 1) >= 0.005) | (cs >= 0.05)))
        print('  {} / {} band-comparisons exceed tolerance'.format(n_bad, len(valid)),
              flush=True)
    print('  metrics CSV : {}'.format(csv_path), flush=True)
    print('  panels      : {}'.format(outdir), flush=True)


# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Bulk empirical Legacy Survey coadd PSFs -> per-brick HDF5 '
                    'shards (MPI + multiprocessing), reproducing the viewer '
                    'coadd-psf inside the dstndstn/cutouts container.',
    )

    parser.add_argument('--catalog-path', type=str, required=True,
                        help='Path to the input FITS catalog.')
    parser.add_argument('--outdir-data', type=str, required=True,
                        help='PSF store directory (per-brick HDF5 shards).')

    parser.add_argument('--ra-col', type=str, default='RA')
    parser.add_argument('--dec-col', type=str, default='DEC')
    parser.add_argument('--id-col', type=str, default='TARGETID')
    parser.add_argument('--brick-col', type=str, default='BRICKNAME',
                        help='Catalog column with the brick name (shard key).')

    parser.add_argument('--layer', type=str, default='ls-dr9')
    parser.add_argument('--bands', type=str, default='g,r,z')
    parser.add_argument('--box', type=int, default=PSF_BOX,
                        help='Standardized PSF box size (odd). PSFs are centered '
                             'and zero-padded to this; larger native stamps are '
                             'center-cropped (logged).')
    parser.add_argument('--halfsize', type=int, default=COADD_PSF_HALFSIZE,
                        help='Half-size (coadd px) of the CCD-selection box; '
                             'matches the viewer (32). Does not set the PSF size.')
    parser.add_argument('--require-bands', type=str, default=None,
                        help='If set (e.g. "grz"), rebuild objects whose stored '
                             'PSF does not already cover all these bands.')

    parser.add_argument('--timeout', type=int, default=300,
                        help='Per-object container build timeout in seconds.')
    parser.add_argument('--max-retries', type=int, default=2,
                        help='Container retries per object (timeouts only; '
                             'deterministic errors go straight to the URL fallback).')
    parser.add_argument('--url-fallback', dest='url_fallback', action='store_true',
                        default=True,
                        help='Fall back to the production viewer for objects the '
                             'container cannot build (default).')
    parser.add_argument('--no-url-fallback', dest='url_fallback', action='store_false')
    parser.add_argument('--url-base', type=str, default=PRODUCTION_VIEWER,
                        help='Viewer base URL for the fallback/validation.')
    parser.add_argument('--url-timeout', type=int, default=120)
    parser.add_argument('--retry-failed', action='store_true',
                        help='Re-attempt objects listed in {}.'.format(FAILED_MANIFEST))
    parser.add_argument('--rebuild-manifest', action='store_true',
                        help='Rebuild psf_manifest.csv from shards before planning.')
    parser.add_argument('--manifest-nproc', type=int, default=None,
                        help='Worker processes for manifest rebuild/bootstrap scan.')

    parser.add_argument('--mp', type=int, default=1,
                        help='Multiprocessing workers per MPI rank.')
    parser.add_argument('--nompi', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    parser.add_argument('--validate-url', type=int, default=0,
                        help='Validation mode (no MPI write): build N sampled '
                             'objects via container AND URL, compare, and write '
                             'diagnostic panels. 0 = off (normal production).')
    parser.add_argument('--validate-outdir', type=str, default=None,
                        help='Where to write validation panels/metrics '
                             '(default: <catalog dir>/psf_validation).')

    args = parser.parse_args()

    outdir_data = os.path.expandvars(args.outdir_data)
    args.catalog_path = os.path.expandvars(args.catalog_path)

    if args.box % 2 == 0:
        sys.exit("ERROR: --box must be odd (got {})".format(args.box))

    if args.validate_url and args.validate_url > 0:
        validate_url(args)
        return

    if args.manifest_nproc is None:
        args.manifest_nproc = int(os.environ.get(
            'SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

    if args.nompi:
        comm = None
    else:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            comm = None

    if args.mp > 1 and 'NERSC_HOST' in os.environ:
        multiprocessing.set_start_method('spawn')

    do_psfs(args, comm=comm, outdir_data=outdir_data)


if __name__ == '__main__':
    main()
