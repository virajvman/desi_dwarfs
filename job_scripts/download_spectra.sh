#!/bin/bash -l
#SBATCH --account=desi
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=spec_download
#SBATCH --output=download_spectra.log
#
# Shared QOS: this job reserves only part of a Perlmutter CPU node (64 logical
# CPUs = 32 physical cores, 128 GB) and is charged for ~1/4 of a node-hour
# rather than a whole node. This is appropriate because the work is I/O-bound
# (reading ~28k coadd files off CFS), so more cores give diminishing returns.
# Note: do NOT set --nodes in shared QOS. If you find you have filesystem
# headroom, raise --cpus-per-task and -ncores together (and --mem if needed).

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Pin BLAS/OpenMP to 1 thread per worker so the ~64 multiprocessing workers
# don't each spawn extra threads and oversubscribe the allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# First run (resumable). If this times out or crashes, re-run the SAME command
# but DROP --overwrite: sync mode then downloads only the TARGETIDs missing
# relative to the last checkpoint.
python3 desi_dwarfs/code/download_spectra.py \
    -ncores 64 \
    -nchunks 370 \
    -checkpoint_every 25 \
    -save_name desi_dr1_dwarf_catalog_spectra \
    --overwrite