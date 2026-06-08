#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256GB
#SBATCH --time=00:30:00
#SBATCH --job-name=deredshift_stacking
#SBATCH --output=deredshift_stacking.log

# De-redshift and resample all dwarf spectra onto a 0.2 A rest-frame grid
# (flux-conserving rebin, use_invvar=False). Writes:
#   .../desi_y1_dwarf_combine_deredshift_hires_noinvvar.h5

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# multiprocessing.Pool(128) in deredshift_resample_desi_spectra; keep BLAS
# single-threaded per worker to avoid oversubscribing the 128 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 -u -c "
import sys
sys.path.insert(0, 'desi_dwarfs/code')
from stacking_analysis.stack_explore import deredshift_for_stacking
deredshift_for_stacking(use_invvar=False, delta_wave=0.4)
"
