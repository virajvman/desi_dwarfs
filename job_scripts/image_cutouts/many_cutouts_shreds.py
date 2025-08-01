#!/usr/bin/python3

"""
many-cutouts.py

MPI wrapper to get a large number of image cutouts. Only run ahead of a VAC
release.
dstndstn/cutouts:dvsro3 <- this is the updated container to use
shifterimg pull dstndstn/cutouts:dvsro3
shifter --image dstndstn/cutouts:dvsro3 cutout --output cutout.jpg --ra 234.2915 --dec 16.7684 --size 128 --layer ls-dr9 --pixscale 0.262 --force --invvar -masks
shifter --image dstndstn/cutouts:dvsro python3
$HOME/DESI2_LOWZ/many_cutouts.py --mp 1 --dry-run

---------------
Modified again by: Viraj Manwadkar (virajvm) by code from John Moustakas
Modified by: Yao-Yuan Mao (yymao)
Modified from: https://github.com/legacysurvey/imagine/blob/master/many-cutouts.py
Original author: Dustin Lang (dstndstn)

"""

import os, sys, time
import numpy as np
import fitsio
import multiprocessing
from glob import glob

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
    #- sumweights will track the sum of the weights that have been assigned
    #- to each group so far
    sumweights = np.zeros(n, dtype=float)

    #- Initialize list of lists of indices for each group
    groups = list()
    for i in range(n):
        groups.append(list())

    #- Assign items from highest weight to lowest weight, always assigning
    #- to whichever group currently has the fewest weights
    weights = np.asarray(weights)
    for i in np.argsort(-weights):
        j = np.argmin(sumweights)
        groups[j].append(i)
        sumweights[j] += weights[i]

    assert len(groups) == n

    #- Reorder groups to spread out large items across different nodes
    #- NOTE: this isn't perfect, e.g. study
    #-   weighted_partition(np.arange(12), 6, groups_per_node=2)
    #- even better would be to zigzag back and forth across the nodes instead
    #- of loop across the nodes.
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

        #- do a final check that all groups were assigned
        for i in range(len(distributed_groups)):
            assert distributed_groups[i] is not None, 'group {} not set'.format(i)

        return distributed_groups

def _cutout_one(args):
    return cutout_one(*args)

def cutout_one(jpegfile, ra, dec, dry_run, rank, iobj,cut_size):
    """
    pixscale = 0.262
    """
    from cutout import cutout

    width = cut_size
    height = cut_size
    
    cmdargs = f'--ra={ra} --dec={dec} --output={jpegfile} --width={cut_size} --height={cut_size} --layer=ls-dr9 --pixscale=0.262 --force --invvar'
    
    if dry_run:
        #print(f'rank {rank} workin on object {iobj}')
        print(f'Rank {rank}, object {iobj}: cutout {cmdargs}')
    else:
        cutout(ra, dec, jpegfile, width=width, height=height, layer='ls-dr9', pixscale=0.262, force=False, bands = ["g","r","z"],invvar=True,maskbits=True)

def plan(comm=None,outdir_data='.', mp=1):
    
    from astropy.table import Table
                                
    t0 = time.time()
    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size
        
    ## let us read the file to get ra, dec info     
    out = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v4.fits") 
    
    print("TOTAL NUMBER = ", len(out))
    
    # out = out #just trying with the first one hundred or so
    #use this to make the file names using objid
    allra = np.array(out["RA"],dtype = object)
    alldec = np.array(out["DEC"],dtype = object)
    allobjids = np.array(out["TARGETID"],dtype = object)
    allsizes = np.array(out["IMAGE_SIZE_PIX"],dtype = object)
    
    file_names = []
    for k in range(len(allobjids)):
        file_i = outdir_data + f"/image_tgid_{allobjids[k]:d}_ra_{allra[k]:.3f}_dec_{alldec[k]:3f}.fits"         
        file_names.append(file_i)

    ## we need to get the information now 
    jpegfiles = np.array(file_names, dtype = object)
    
    ##the first argument here is the weights. we just want equal weights for all targets
    groups = weighted_partition(np.ones_like(alldec), size)
            
    return jpegfiles, allra, alldec, groups, allobjids, allsizes
                
                
def do_cutouts(args, comm=None, outdir_data='.'):

    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    t0 = time.time()
    if rank == 0:
        jpegfiles, allra, alldec, groups, allobjids,allsizes = plan(
                comm=comm,outdir_data=outdir_data, mp=args.mp)
        print(f'Planning took {(time.time() - t0):.2f} sec')
    else:
        jpegfiles, allra, alldec, groups, allobjids, allsizes = [], [], [], [], [], []

    if comm:
        jpegfiles = comm.bcast(jpegfiles, root=0)
        allra = comm.bcast(allra, root=0)
        alldec = comm.bcast(alldec, root=0)
        groups = comm.bcast(groups, root=0)
        allobjids = comm.bcast(allobjids, root=0)
        allsizes = comm.bcast(allsizes,root=0)
        
    sys.stdout.flush()
    
    # all done
    if len(jpegfiles) == 0 or len(np.hstack(jpegfiles)) == 0:
        return
        
    assert(len(groups) == size)
    

    for ii in groups[rank]:
        print(f'Rank {rank} started at {time.asctime()}')
        sys.stdout.flush()
        
        ## the list of args below are for the _cutout_one function
        # mpargs = [(jpegfile, ra, dec, args.dry_run, rank, iobj) for iobj, (jpegfile, ra, dec) in
        #           enumerate(zip(jpegfiles[ii], allra[ii], alldec[ii]))]
        
        mpargs = [(jpegfiles[ii], allra[ii], alldec[ii], args.dry_run, rank, allobjids[ii],allsizes[ii])]
        
        if args.mp > 1:
            with multiprocessing.Pool(args.mp) as P:
                P.map(_cutout_one, mpargs)
        else:
            [cutout_one(*mparg) for mparg in mpargs]

    print(f'  rank {rank} is done')
    sys.stdout.flush()

    if comm is not None:
        comm.barrier()

    if rank == 0 and not args.dry_run:
        print(f'All done at {time.asctime()}')
            
            
            
            
def main():
    """Main wrapper.

    """
    import argparse    
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mp', type=int, default=1, help='Number of multiprocessing processes per MPI rank or node.')

    parser.add_argument('--plan', action='store_true', help='Plan how many nodes to use and how to distribute the targets.')
    parser.add_argument('--nompi', action='store_true', help='Do not use MPI parallelism.')
    parser.add_argument('--dry-run', action='store_true', help='Generate but do not run commands.')
    parser.add_argument('--outdir-data', default='$PSCRATCH/redo_photometry_plots/all_good_cutouts', type=str, help='Base output data directory.')
    
    args = parser.parse_args()

    outdir_data = os.path.expandvars(args.outdir_data)

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
        import multiprocessing
        multiprocessing.set_start_method('spawn')
        
    do_cutouts(args, comm=comm, outdir_data=outdir_data)

if __name__ == '__main__':
    main()
            
            
            
            
            
            
            
    
            
            
            
            
