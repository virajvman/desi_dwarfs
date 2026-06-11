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
#SBATCH --job-name=stack_direct_metallicity
#SBATCH --output=stack_direct_metallicity.log

# Run the direct-method (T_e) abundance fits on stacked dwarf spectra:
# EW-binned stacks (STACK_PATH/) and mass-only stacks (STACK_PATH/mstar_only/).
# UltraNest fits many bootstrap rows per EW bin; mass-only bins have a single row.
# Submit with sbatch, or run the python line directly on an exclusive CPU node.

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# joblib (loky) spawns N_JOBS worker processes; keep BLAS single-threaded per
# worker to avoid oversubscribing the 128 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 desi_dwarfs/code/nebular_stuff/stack_direct_metallicity.py \
    --line-flux-type FLUX --fix-ne100 --products both
