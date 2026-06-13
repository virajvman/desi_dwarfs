#!/usr/bin/python3

"""
many_cutouts_general.py

General-purpose MPI + multiprocessing pipeline to generate Legacy Survey
image cutouts on NERSC via the dstndstn/cutouts Shifter container, writing
per-brick HDF5 shards (see code/cutout_store.py for the store layout).

Design notes (decisions from the 2026-06 redesign):

* Work is partitioned across MPI ranks BY BRICK (weighted by total pixel
  area), so each shard has exactly one writing rank. Pool workers fetch
  arrays in memory; the rank main process buffers per brick and writes each
  shard atomically once all its objects have arrived.

* The container image (all tags <= 2025-05) predates imagine fix 1a034cd
  ("when rendering maskbits, only pass first band", 2025-07-16). Without it,
  any ls-dr9 cutout straddling dec=32.375 in the NGC crashes with IndexError
  in LegacySurveySplitLayer.render_rgb. _apply_split_layer_patch() reproduces
  the upstream fix inside each worker; it is a no-op on a fixed container.

* Deterministic per-object exceptions are NOT retried against the container;
  they go straight to a one-shot fallback against the production viewer
  (https://www.legacysurvey.org/viewer/ -- NOT viewer-dev, which serves 2x
  invvar in brick overlaps since imagine d2ba303). Objects that fail both
  paths are appended to {outdir}/permanently_failed.csv, which plan() reads
  and excludes on subsequent runs (--retry-failed to override).

Usage examples:
  # Dry run (no MPI)
  shifter --image dstndstn/cutouts:dvsro3 python3 many_cutouts_general.py \
      --catalog-path /path/to/catalog.fits --outdir-data /path/to/store \
      --cutout-size 152 --nompi --dry-run

  # Production: see get_imgs_{clean,shreds,sga}.sbatch / cutouts_cnn_general.sh

---------------
Modified again by: Viraj Manwadkar (virajvm) by code from John Moustakas
Modified by: Yao-Yuan Mao (yymao)
Modified from: https://github.com/legacysurvey/imagine/blob/master/many-cutouts.py
Original author: Dustin Lang (dstndstn)
"""

import os
import csv
import sys
import time
import signal
import tempfile
import multiprocessing
from collections import defaultdict

import numpy as np

# repo layout: job_scripts/image_cutouts/general/ -> repo root -> code/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..', 'code'))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import cutout_store
from cutout_store import (FAILED_MANIFEST, PRODUCTION_VIEWER,
                          parse_cutout_fits, make_cutout_record,
                          fetch_cutout_url, load_tombstones,
                          load_manifest, build_manifest, append_manifest_delta,
                          merge_manifest_deltas, cutout_satisfies_request)


def weighted_partition(weights, n):
    '''Partition `weights` into `n` groups with approximately equal sums.

    Returns list of lists of indices of weights for each group. Allows
    non-contiguous items to be grouped together for better balancing.
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
# PSF-size sampling (optional, --save-psf; intended for random cutouts)
# ----------------------------------------------------------------------

# ls-dr9 north/south boundary: bricks above this Dec are rendered from the
# north (MzLS/BASS) coadd, below from south (DECaLS). Sampling the psfsize map
# from the matching region keeps the PSF consistent with the saved image.
PSFSIZE_DEC_SPLIT = 32.375


def _psfsize_map_path(survey_dir, region, brick, band):
    """Path to a per-brick psfsize coadd map (whether or not it exists)."""
    return os.path.join(survey_dir, region, 'coadd', brick[:3], brick,
                        'legacysurvey-{}-psfsize-{}.fits.fz'.format(brick, band))


def _sample_psfsize_map(path, recs):
    """Sample one band's psfsize map at each record's center RA/Dec.

    Returns a list (parallel to `recs`) of FWHM values in arcsec, with NaN
    where the map is missing, the position is off-map, or there is no coverage
    (pixel == 0). Reads the WCS from the lazily-loaded header (no decompress)
    and reads each needed pixel with the smallest possible tile decompression
    via fitsio, falling back to a single full astropy read if fitsio is absent.
    Never raises -- any failure yields NaN so the image is still written.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    n = len(recs)
    nan = float('nan')
    if not os.path.exists(path):
        print('WARNING: psfsize map missing, storing NaN for all objects: {}'.format(path),
              flush=True)
        return [nan] * n

    # WCS + image dims from the header alone (does not decompress the data).
    try:
        with fits.open(path, memmap=False) as hdul:
            hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            hdr = hdu.header
            wcs = WCS(hdr)
            nx = int(hdr['NAXIS1'])
            ny = int(hdr['NAXIS2'])
    except Exception as exc:
        print('WARNING: failed reading psfsize header {} ({}); storing NaN'.format(path, exc),
              flush=True)
        return [nan] * n

    # Nearest integer pixel for each object (None if off-map or unprojectable).
    pix_xy = []
    for rec in recs:
        try:
            x, y = wcs.all_world2pix(float(rec['ra']), float(rec['dec']), 0)
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            pix_xy.append((xi, yi) if (0 <= xi < nx and 0 <= yi < ny) else None)
        except Exception:
            pix_xy.append(None)

    out = [nan] * n
    try:
        import fitsio
        with fitsio.FITS(path) as F:
            ext = F[1] if len(F) > 1 else F[0]
            for i, xy in enumerate(pix_xy):
                if xy is None:
                    continue
                xi, yi = xy
                pix = float(ext[yi:yi + 1, xi:xi + 1][0, 0])
                out[i] = pix if pix > 0 else nan
    except ImportError:
        # No fitsio in this environment: one full decompress via astropy.
        with fits.open(path, memmap=False) as hdul:
            data = (hdul[1] if len(hdul) > 1 else hdul[0]).data
            for i, xy in enumerate(pix_xy):
                if xy is None:
                    continue
                xi, yi = xy
                pix = float(data[yi, xi])
                out[i] = pix if pix > 0 else nan
    except Exception as exc:
        print('WARNING: failed reading psfsize data {} ({}); storing NaN'.format(path, exc),
              flush=True)
        return [nan] * n
    return out


def attach_psfsize(recs, brickname, survey_dir, bands):
    """Attach psfsize_{band} (arcsec FWHM, NaN where unavailable) to each record
    in `recs` by sampling the per-brick coadd maps at the cutout centers.

    Called at brick-flush time (single writer process), so each band's map is
    opened exactly once per brick. Region (north/south) is resolved once from
    the brick's representative Dec; all objects in a brick share it.
    """
    if not recs:
        return
    dec0 = float(np.median([float(rec['dec']) for rec in recs]))
    region = 'north' if dec0 > PSFSIZE_DEC_SPLIT else 'south'

    for rec in recs:
        rec.setdefault('psfsize', {})
    for band in bands:
        path = _psfsize_map_path(survey_dir, region, brickname, band)
        vals = _sample_psfsize_map(path, recs)
        for rec, val in zip(recs, vals):
            rec['psfsize'][band] = val


# ----------------------------------------------------------------------
# Worker-side fetching
# ----------------------------------------------------------------------

class _CutoutTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _CutoutTimeout("Cutout call timed out")


_PATCH_DONE = False


def _apply_split_layer_patch():
    """Reproduce imagine fix 1a034cd inside the (older) container code.

    When rendering the (band-independent) maskbits plane, the sub-layer
    renderers return a 1-element list, but the SplitLayer merge loop for
    dec-split-straddling cutouts indexes it with the full band list ->
    IndexError. Truncating the requested bands to the first band when
    maskbits=True restores the intended behavior for all objects.
    """
    global _PATCH_DONE
    if _PATCH_DONE:
        return
    _PATCH_DONE = True
    try:
        from map import views as _views
        cls = getattr(_views, 'LegacySurveySplitLayer', None)
        if cls is None or getattr(cls.render_rgb, '_maskbits_patched', False):
            return
        _orig = cls.render_rgb

        def render_rgb_fixed(self, wcs, zoom, x, y, bands=None, **kwargs):
            if kwargs.get('maskbits') and bands is not None and len(bands) > 1:
                bands = bands[:1]
            return _orig(self, wcs, zoom, x, y, bands=bands, **kwargs)

        render_rgb_fixed._maskbits_patched = True
        cls.render_rgb = render_rgb_fixed
    except Exception as exc:
        print('WARNING: maskbits monkeypatch not applied: {}'.format(exc), flush=True)


def _fetch_container(task):
    """Fetch one cutout via the container (CFS reads); returns a store record."""
    from cutout import cutout
    from astropy.io import fits

    _apply_split_layer_patch()

    tmpfn = os.path.join(
        tempfile.gettempdir(),
        'cutout_{}_{}.fits'.format(task['targetid'], os.getpid()))
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(task['timeout'])
        try:
            cutout(task['ra'], task['dec'], tmpfn,
                   width=task['size'], height=task['size'],
                   layer=task['layer'], pixscale=task['pixscale'],
                   force=True, bands=list(task['bands']),
                   invvar=task['invvar'], maskbits=task['maskbits'])
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if not os.path.exists(tmpfn):
            raise RuntimeError('container cutout produced no output file')
        with fits.open(tmpfn, memmap=False) as hdul:
            image, invvar, mask, header_str = parse_cutout_fits(
                hdul, task['invvar'], task['maskbits'])
    finally:
        if os.path.exists(tmpfn):
            os.remove(tmpfn)

    return make_cutout_record(task['targetid'], task['ra'], task['dec'],
                              task['size'], task['bands'],
                              image, invvar, mask, header_str, 'container',
                              layer=task['layer'])


def _fetch_url(task):
    """One-shot fallback fetch from the production Legacy Surveys viewer."""
    return fetch_cutout_url(
        task['ra'], task['dec'], task['size'], task['targetid'],
        layer=task['layer'], pixscale=task['pixscale'], bands=task['bands'],
        invvar=task['invvar'], maskbits=task['maskbits'],
        url_base=task['url_base'], timeout=task['url_timeout'])


def _fetch_one_safe(task):
    """Worker entry point. Returns ('ok', brick, record) or ('fail', failrec).

    Container timeouts are retried up to max_retries; any other container
    exception is deterministic and skips straight to the URL fallback.
    """
    brick = task['brick']

    if task['dry_run']:
        print('Rank {}, object {}: ra={} dec={} brick={} size={} layer={}'.format(
            task['rank'], task['targetid'], task['ra'], task['dec'],
            brick, task['size'], task['layer']), flush=True)
        return ('ok', brick, None)

    errors = []
    for attempt in range(task['max_retries'] + 1):
        try:
            return ('ok', brick, _fetch_container(task))
        except _CutoutTimeout:
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
            return ('ok', brick, _fetch_url(task))
        except Exception as exc:
            errors.append('url: {!r}'.format(exc))
            print('Rank {}: URL fallback failed on {}: {!r}'.format(
                task['rank'], task['targetid'], exc), flush=True)

    failrec = {
        'targetid': task['targetid'],
        'ra': task['ra'],
        'dec': task['dec'],
        'brickname': brick,
        'reason': ' | '.join(errors),
    }
    return ('fail', brick, failrec)


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------

def plan(args, outdir_data, size):
    """Rank-0 planning: decide what needs fetching and partition by brick.

    Returns (brick_names, brick_rows, ra, dec, tgid, sizes, groups) where
    brick_rows[i] indexes into the needed-object arrays for brick_names[i]
    and groups[r] lists brick indices assigned to MPI rank r.
    """
    from astropy.table import Table

    cat = Table.read(args.catalog_path)
    n = len(cat)
    print('Total objects in catalog: {}'.format(n), flush=True)

    for col, name in ((args.ra_col, '--ra-col'), (args.dec_col, '--dec-col'),
                      (args.id_col, '--id-col'), (args.brick_col, '--brick-col')):
        if col not in cat.colnames:
            sys.exit("ERROR: column '{}' ({}) not found in {}. Available: {}".format(
                col, name, args.catalog_path, ', '.join(cat.colnames[:40])))

    allra = np.asarray(cat[args.ra_col], dtype=np.float64)
    alldec = np.asarray(cat[args.dec_col], dtype=np.float64)
    alltgid = np.asarray(cat[args.id_col], dtype=np.int64)
    allbrick = np.asarray(cat[args.brick_col]).astype(str)

    if args.size_col is not None:
        if args.size_col not in cat.colnames:
            sys.exit("ERROR: --size-col '{}' not found in catalog".format(args.size_col))
        allsizes = np.asarray(cat[args.size_col], dtype=np.int64)
    else:
        allsizes = np.full(n, args.cutout_size, dtype=np.int64)

    tombs = set()
    if not args.retry_failed:
        tombs = load_tombstones(outdir_data)
        if tombs:
            print('Excluding {} tombstoned objects from {} '
                  '(--retry-failed to re-attempt)'.format(len(tombs), FAILED_MANIFEST),
                  flush=True)

    print('Loading cutout manifest from {} ...'.format(outdir_data), flush=True)
    t0 = time.time()
    existing = cutout_store.load_manifest(outdir_data, nproc=args.manifest_nproc,
                                          bootstrap=not args.rebuild_manifest)
    n_existing = sum(len(v) for v in existing.values())
    print('  {} objects in manifest covering {} bricks ({:.1f}s)'.format(
        n_existing, len(existing), time.time() - t0), flush=True)

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
        if cutout_satisfies_request(row, allsizes[k],
                                    require_invvar=args.invvar,
                                    require_mask=args.maskbits):
            continue
        need.append(k)
    need = np.asarray(need, dtype=np.int64)
    print('Need to fetch {}/{} cutouts'.format(len(need), n), flush=True)

    ra, dec = allra[need], alldec[need]
    tgid, sizes, brick = alltgid[need], allsizes[need], allbrick[need]

    brick_names, brick_inverse = np.unique(brick, return_inverse=True)
    brick_rows = [np.flatnonzero(brick_inverse == i) for i in range(len(brick_names))]
    # weight by total pixel area so ranks get comparable work
    brick_weights = np.array([np.sum(sizes[rows].astype(float) ** 2) for rows in brick_rows])

    return brick_names, brick_rows, ra, dec, tgid, sizes, brick_weights


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def do_cutouts(args, comm=None, outdir_data='.'):
    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    bands = tuple(b.strip() for b in args.bands.split(','))

    t0 = time.time()
    if rank == 0:
        os.makedirs(outdir_data, exist_ok=True)
        if args.rebuild_manifest:
            print('Rebuilding cutout manifest from shards ...', flush=True)
            cutout_store.build_manifest(outdir_data, nproc=args.manifest_nproc)
        (brick_names, brick_rows, ra, dec, tgid,
         sizes, brick_weights) = plan(args, outdir_data, size)
        groups = weighted_partition(brick_weights, size)
        print('Planning took {:.2f} sec'.format(time.time() - t0), flush=True)
    else:
        brick_names = brick_rows = ra = dec = tgid = sizes = groups = None

    if comm:
        brick_names = comm.bcast(brick_names, root=0)
        brick_rows = comm.bcast(brick_rows, root=0)
        ra = comm.bcast(ra, root=0)
        dec = comm.bcast(dec, root=0)
        tgid = comm.bcast(tgid, root=0)
        sizes = comm.bcast(sizes, root=0)
        groups = comm.bcast(groups, root=0)

    if len(brick_names) == 0:
        if rank == 0:
            print('Nothing to do.', flush=True)
        if comm is not None:
            comm.barrier()
        # Still fold in (and clear) any deltas orphaned by a previously crashed
        # run -- otherwise they linger unmerged until that catalog is re-run.
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
            tasks.append({
                'targetid': int(tgid[r]), 'ra': float(ra[r]), 'dec': float(dec[r]),
                'brick': bname, 'size': int(sizes[r]),
                'layer': args.layer, 'pixscale': args.pixscale, 'bands': bands,
                'invvar': args.invvar, 'maskbits': args.maskbits,
                'timeout': args.timeout, 'max_retries': args.max_retries,
                'url_fallback': args.url_fallback, 'url_base': args.url_base,
                'url_timeout': args.url_timeout,
                'dry_run': args.dry_run, 'rank': rank,
            })

    total = len(tasks)
    print('Rank {}: assigned {} objects in {} bricks'.format(
        rank, total, len(my_bricks)), flush=True)

    buffers = defaultdict(list)
    settled = defaultdict(int)   # successes + failures per brick
    failed = []
    n_done = 0
    n_written = 0

    def handle_result(result):
        nonlocal n_done, n_written
        status, bname, payload = result
        n_done += 1
        settled[bname] += 1
        if status == 'ok':
            if payload is not None:           # None on dry runs
                buffers[bname].append(payload)
        else:
            failed.append(payload)
        if settled[bname] == expected[bname]:
            recs = buffers.pop(bname, [])
            if recs:
                if args.save_psf:
                    attach_psfsize(recs, bname, args.psfsize_survey_dir, bands)
                manifest_rows = cutout_store.write_cutouts_batch(outdir_data, bname, recs)
                append_manifest_delta(outdir_data, 'rank{}'.format(rank), manifest_rows)
                n_written += 1
        if n_done % 100 == 0 or n_done == total:
            print('Rank {}: {}/{} done, {} failed, {} shards written, {:.0f}s'.format(
                rank, n_done, total, len(failed), n_written, time.time() - t0),
                flush=True)

    if total > 0 and args.mp > 1 and not args.dry_run:
        pool = multiprocessing.Pool(args.mp, maxtasksperchild=50)
        try:
            for result in pool.imap_unordered(_fetch_one_safe, tasks, chunksize=1):
                handle_result(result)
        finally:
            pool.close()
            pool.join()
    else:
        for task in tasks:
            handle_result(_fetch_one_safe(task))

    # flush anything left (only possible after a pool error)
    for bname in list(buffers.keys()):
        recs = buffers.pop(bname)
        if recs:
            print('Rank {}: WARNING flushing incomplete brick {}'.format(rank, bname),
                  flush=True)
            if args.save_psf:
                attach_psfsize(recs, bname, args.psfsize_survey_dir, bands)
            manifest_rows = cutout_store.write_cutouts_batch(outdir_data, bname, recs)
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
        store_index = cutout_store.load_manifest(outdir_data, bootstrap=False)
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Bulk Legacy Survey cutouts -> per-brick HDF5 shards '
                    '(MPI + multiprocessing).',
    )

    parser.add_argument('--catalog-path', type=str, required=True,
                        help='Path to the input FITS catalog.')
    parser.add_argument('--outdir-data', type=str, required=True,
                        help='Cutout store directory (per-brick HDF5 shards).')

    parser.add_argument('--ra-col', type=str, default='RA')
    parser.add_argument('--dec-col', type=str, default='DEC')
    parser.add_argument('--id-col', type=str, default='TARGETID')
    parser.add_argument('--brick-col', type=str, default='BRICKNAME',
                        help='Catalog column with the Legacy Surveys brick name '
                             '(shard key).')
    parser.add_argument('--cutout-size', type=int, default=152,
                        help='Cutout size in pixels (used if --size-col is not set).')
    parser.add_argument('--size-col', type=str, default=None,
                        help='Catalog column for per-object cutout sizes.')

    parser.add_argument('--pixscale', type=float, default=0.262)
    parser.add_argument('--layer', type=str, default='ls-dr9')
    parser.add_argument('--bands', type=str, default='g,r,z')
    # NOTE: paired flags instead of argparse.BooleanOptionalAction --
    # the container python is 3.8, BooleanOptionalAction needs 3.9+
    parser.add_argument('--invvar', dest='invvar', action='store_true', default=True,
                        help='Include inverse-variance maps (default).')
    parser.add_argument('--no-invvar', dest='invvar', action='store_false')
    parser.add_argument('--maskbits', dest='maskbits', action='store_true', default=True,
                        help='Include maskbits maps (default).')
    parser.add_argument('--no-maskbits', dest='maskbits', action='store_false')

    parser.add_argument('--save-psf', dest='save_psf', action='store_true', default=False,
                        help='Sample the per-brick PSF-size coadd map at each cutout '
                             'center and store psfsize_{g,r,z} attrs (arcsec FWHM, NaN '
                             'where unavailable). Off by default; intended for random '
                             'cutouts (source-centered runs already carry PSFSIZE_* from '
                             'the Tractor catalog). Does not gate existence checks.')
    parser.add_argument('--psfsize-survey-dir', dest='psfsize_survey_dir', type=str,
                        default='/global/cfs/cdirs/cosmo/data/legacysurvey/dr9',
                        help='Base Legacy Surveys directory; "{north,south}/coadd/..." '
                             'is appended per brick (region by the +32.375 deg split). '
                             'Only used with --save-psf.')

    parser.add_argument('--timeout', type=int, default=120,
                        help='Per-cutout container timeout in seconds.')
    parser.add_argument('--max-retries', type=int, default=2,
                        help='Container retries per object (timeouts only; '
                             'deterministic errors go straight to the URL fallback).')
    parser.add_argument('--url-fallback', dest='url_fallback', action='store_true',
                        default=True,
                        help='Fall back to the production viewer for objects the '
                             'container cannot fetch (default).')
    parser.add_argument('--no-url-fallback', dest='url_fallback', action='store_false')
    parser.add_argument('--url-base', type=str, default=PRODUCTION_VIEWER,
                        help='Viewer base URL for the fallback (use production, not '
                             'viewer-dev: dev serves 2x invvar in brick overlaps).')
    parser.add_argument('--url-timeout', type=int, default=120)
    parser.add_argument('--retry-failed', action='store_true',
                        help='Re-attempt objects listed in {} (e.g. after a container '
                             'fix).'.format(FAILED_MANIFEST))
    parser.add_argument('--rebuild-manifest', action='store_true',
                        help='Rebuild cutout_manifest.csv from shards before planning '
                             '(rank 0 only; use if manifest drift is suspected).')
    parser.add_argument('--manifest-nproc', type=int, default=None,
                        help='Worker processes for manifest rebuild/bootstrap scan '
                             '(default: SLURM_CPUS_PER_TASK or cpu_count).')

    parser.add_argument('--mp', type=int, default=1,
                        help='Multiprocessing workers per MPI rank.')
    parser.add_argument('--nompi', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    outdir_data = os.path.expandvars(args.outdir_data)
    args.catalog_path = os.path.expandvars(args.catalog_path)

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

    # https://docs.nersc.gov/development/languages/python/parallel-python/#use-the-spawn-start-method
    if args.mp > 1 and 'NERSC_HOST' in os.environ:
        multiprocessing.set_start_method('spawn')

    do_cutouts(args, comm=comm, outdir_data=outdir_data)


if __name__ == '__main__':
    main()
