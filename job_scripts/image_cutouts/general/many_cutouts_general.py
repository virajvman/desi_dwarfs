#!/usr/bin/python3

"""
many_cutouts_general.py

General-purpose MPI + multiprocessing wrapper to generate a large number of
Legacy Survey image cutouts on NERSC via the dstndstn/cutouts Shifter container.

Usage examples:
  # Dry run (no MPI)
  shifter --image dstndstn/cutouts:dvsro3 python3 many_cutouts_general.py \
      --catalog-path /path/to/catalog.fits --outdir-data /path/to/output \
      --ra-col RA --dec-col DEC --id-col TARGETID --cutout-size 152 \
      --nompi --dry-run

  # Production (launched via cutouts_cnn_general.sh / get_imgs_general.sbatch)
  # See those wrapper scripts for SLURM submission.

---------------
Modified again by: Viraj Manwadkar (virajvm) by code from John Moustakas
Modified by: Yao-Yuan Mao (yymao)
Modified from: https://github.com/legacysurvey/imagine/blob/master/many-cutouts.py
Original author: Dustin Lang (dstndstn)
"""

import os
import sys
import time
import signal
import threading
import multiprocessing
import numpy as np
import fitsio


def weighted_partition(weights, n, groups_per_node=None):
    '''
    Partition `weights` into `n` groups with approximately same sum(weights)

    Args:
        weights: array-like weights
        n: number of groups

    Returns list of lists of indices of weights for each group

    Notes:
        compared to `dist_discrete_all`, this function allows non-contiguous
        items to be grouped together which allows better balancing.
    '''
    sumweights = np.zeros(n, dtype=float)

    groups = list()
    for i in range(n):
        groups.append(list())

    weights = np.asarray(weights)
    for i in np.argsort(-weights):
        j = np.argmin(sumweights)
        groups[j].append(i)
        sumweights[j] += weights[i]

    assert len(groups) == n

    if groups_per_node is None:
        return groups
    else:
        distributed_groups = [None,] * len(groups)
        num_nodes = (n + groups_per_node - 1) // groups_per_node
        i = 0
        for noderank in range(groups_per_node):
            for inode in range(num_nodes):
                j = inode*groups_per_node + noderank
                if i < n and j < n:
                    distributed_groups[j] = groups[i]
                    i += 1

        for i in range(len(distributed_groups)):
            assert distributed_groups[i] is not None, 'group {} not set'.format(i)

        return distributed_groups


class _CutoutTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _CutoutTimeout("Cutout call timed out")


def cutout_one(jpegfile, ra, dec, dry_run, rank, iobj, cut_size,
               layer, pixscale, bands, invvar, maskbits):
    from cutout import cutout

    width = cut_size
    height = cut_size

    if dry_run:
        print(f'Rank {rank}, object {iobj}: ra={ra} dec={dec} '
              f'output={jpegfile} size={cut_size} layer={layer} '
              f'pixscale={pixscale} bands={bands}')
    else:
        cutout(ra, dec, jpegfile, width=width, height=height,
               layer=layer, pixscale=pixscale, force=False,
               bands=bands, invvar=invvar, maskbits=maskbits)


def _cutout_one_safe(args):
    """Worker wrapper with per-task timeout and error handling."""
    (jpegfile, ra, dec, dry_run, rank, iobj, cut_size,
     layer, pixscale, bands, invvar, maskbits,
     timeout, max_retries) = args

    for attempt in range(max_retries + 1):
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            cutout_one(jpegfile, ra, dec, dry_run, rank, iobj, cut_size,
                       layer, pixscale, bands, invvar, maskbits)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return None
        except _CutoutTimeout:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            print(f'Rank {rank}: TIMEOUT on object {iobj} (attempt {attempt+1}/{max_retries+1}): {jpegfile}',
                  flush=True)
        except Exception as e:
            signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass
            print(f'Rank {rank}: ERROR on object {iobj} (attempt {attempt+1}/{max_retries+1}): {e}',
                  flush=True)

    return (iobj, ra, dec, jpegfile)


def plan(comm=None, outdir_data='.', catalog_path=None,
         ra_col='RA', dec_col='DEC', id_col='TARGETID',
         cutout_size=152, size_col=None):

    from astropy.table import Table

    t0 = time.time()
    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    out = Table.read(catalog_path)
    print(f"Total objects in catalog: {len(out)}")

    allra = np.array(out[ra_col], dtype=object)
    alldec = np.array(out[dec_col], dtype=object)
    allobjids = np.array(out[id_col], dtype=object)

    if size_col is not None and size_col in out.colnames:
        allsizes = np.array(out[size_col], dtype=object)
    else:
        allsizes = np.full(len(out), cutout_size, dtype=int)

    file_names = []
    need_inds = []
    n = len(out)

    print("Checking which cutouts already exist...")
    for k in range(n):
        file_i = os.path.join(
            outdir_data,
            f"image_tgid_{allobjids[k]:d}_ra_{allra[k]:.3f}_dec_{alldec[k]:.3f}.fits"
        )
        if not os.path.exists(file_i):
            file_names.append(file_i)
            need_inds.append(k)

        if (k + 1) % 10000 == 0 or (k + 1) == n:
            print(f"  Checked {k+1}/{n} ({(k+1)/n:.1%})")

    need_inds = np.array(need_inds)
    print(f"Need to generate {len(need_inds)}/{n} cutouts")

    if len(need_inds) == 0:
        return (np.array([], dtype=object), np.array([]), np.array([]),
                weighted_partition(np.array([]), size),
                np.array([]), np.array([], dtype=int))

    allra = allra[need_inds]
    alldec = alldec[need_inds]
    allobjids = allobjids[need_inds]
    allsizes = allsizes[need_inds]

    jpegfiles = np.array(file_names, dtype=object)
    groups = weighted_partition(np.ones_like(alldec), size)

    return jpegfiles, allra, alldec, groups, allobjids, allsizes


def _join_pool_with_timeout(pool, timeout=300):
    """Call pool.join() in a thread; terminate the pool if it takes too long."""
    join_thread = threading.Thread(target=pool.join)
    join_thread.start()
    join_thread.join(timeout=timeout)
    if join_thread.is_alive():
        print("WARNING: Pool.join() timed out, forcing terminate()", flush=True)
        pool.terminate()
        join_thread.join(timeout=30)


def do_cutouts(args, comm=None, outdir_data='.'):

    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    bands = [b.strip() for b in args.bands.split(',')]

    t0 = time.time()
    if rank == 0:
        jpegfiles, allra, alldec, groups, allobjids, allsizes = plan(
            comm=comm, outdir_data=outdir_data,
            catalog_path=args.catalog_path,
            ra_col=args.ra_col, dec_col=args.dec_col, id_col=args.id_col,
            cutout_size=args.cutout_size, size_col=args.size_col,
        )
        print(f'Planning took {(time.time() - t0):.2f} sec')
    else:
        jpegfiles, allra, alldec, groups, allobjids, allsizes = [], [], [], [], [], []

    if comm:
        jpegfiles = comm.bcast(jpegfiles, root=0)
        allra = comm.bcast(allra, root=0)
        alldec = comm.bcast(alldec, root=0)
        groups = comm.bcast(groups, root=0)
        allobjids = comm.bcast(allobjids, root=0)
        allsizes = comm.bcast(allsizes, root=0)

    sys.stdout.flush()

    if len(jpegfiles) == 0:
        print(f'Rank {rank}: nothing to do')
        if comm is not None:
            comm.barrier()
        return

    assert len(groups) == size

    my_indices = groups[rank]
    total_for_rank = len(my_indices)
    print(f'Rank {rank}: assigned {total_for_rank} cutouts', flush=True)

    all_mpargs = [
        (jpegfiles[ii], allra[ii], alldec[ii], args.dry_run, rank,
         allobjids[ii], allsizes[ii],
         args.layer, args.pixscale, bands, args.invvar, args.maskbits,
         args.timeout, args.max_retries)
        for ii in my_indices
    ]

    failed = []

    if args.mp > 1:
        pool = multiprocessing.Pool(args.mp, maxtasksperchild=50)
        try:
            for i, result in enumerate(
                pool.imap_unordered(_cutout_one_safe, all_mpargs, chunksize=4)
            ):
                if result is not None:
                    failed.append(result)
                done = i + 1
                if done % 100 == 0 or done == total_for_rank:
                    elapsed = time.time() - t0
                    print(f'Rank {rank}: {done}/{total_for_rank} done, '
                          f'{len(failed)} failed, {elapsed:.0f}s elapsed',
                          flush=True)
        except Exception as e:
            print(f'Rank {rank}: Pool iteration error: {e}', flush=True)
        finally:
            pool.close()
            _join_pool_with_timeout(pool, timeout=300)
    else:
        for i, task_args in enumerate(all_mpargs):
            result = _cutout_one_safe(task_args)
            if result is not None:
                failed.append(result)
            done = i + 1
            if done % 100 == 0 or done == total_for_rank:
                elapsed = time.time() - t0
                print(f'Rank {rank}: {done}/{total_for_rank} done, '
                      f'{len(failed)} failed, {elapsed:.0f}s elapsed',
                      flush=True)

    if failed:
        manifest_path = os.path.join(outdir_data, f'failed_cutouts_rank{rank}.csv')
        with open(manifest_path, 'w') as f:
            f.write("targetid,ra,dec,filepath\n")
            for (objid, ra, dec, fpath) in failed:
                f.write(f"{objid},{ra},{dec},{fpath}\n")
        print(f'Rank {rank}: {len(failed)} failures written to {manifest_path}',
              flush=True)

    print(f'Rank {rank}: finished at {time.asctime()} '
          f'({total_for_rank - len(failed)}/{total_for_rank} succeeded)',
          flush=True)

    if comm is not None:
        comm.barrier()

    if rank == 0 and not args.dry_run:
        print(f'All ranks done at {time.asctime()}')


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Generate Legacy Survey image cutouts in bulk using MPI + multiprocessing.',
    )

    parser.add_argument('--catalog-path', type=str, required=True,
                        help='Path to the input FITS catalog.')
    parser.add_argument('--outdir-data', type=str, required=True,
                        help='Output directory for cutout FITS files.')

    parser.add_argument('--ra-col', type=str, default='RA',
                        help='Catalog column name for RA.')
    parser.add_argument('--dec-col', type=str, default='DEC',
                        help='Catalog column name for DEC.')
    parser.add_argument('--id-col', type=str, default='TARGETID',
                        help='Catalog column name for target IDs.')
    parser.add_argument('--cutout-size', type=int, default=152,
                        help='Cutout size in pixels (used if --size-col is not set).')
    parser.add_argument('--size-col', type=str, default=None,
                        help='Catalog column for per-object cutout sizes (overrides --cutout-size).')

    parser.add_argument('--pixscale', type=float, default=0.262,
                        help='Pixel scale in arcsec/pixel.')
    parser.add_argument('--layer', type=str, default='ls-dr9',
                        help='Legacy Survey layer name.')
    parser.add_argument('--bands', type=str, default='g,r,z',
                        help='Comma-separated band names.')
    parser.add_argument('--invvar', action=argparse.BooleanOptionalAction, default=True,
                        help='Include inverse-variance maps.')
    parser.add_argument('--maskbits', action=argparse.BooleanOptionalAction, default=True,
                        help='Include maskbits maps.')

    parser.add_argument('--timeout', type=int, default=120,
                        help='Per-cutout timeout in seconds (safety net for hung I/O).')
    parser.add_argument('--max-retries', type=int, default=2,
                        help='Number of retries per failed cutout.')

    parser.add_argument('--mp', type=int, default=1,
                        help='Number of multiprocessing workers per MPI rank.')
    parser.add_argument('--plan', action='store_true',
                        help='Plan how many nodes to use and exit.')
    parser.add_argument('--nompi', action='store_true',
                        help='Do not use MPI parallelism.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print cutout commands without executing.')

    args = parser.parse_args()

    outdir_data = os.path.expandvars(args.outdir_data)
    args.catalog_path = os.path.expandvars(args.catalog_path)

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
