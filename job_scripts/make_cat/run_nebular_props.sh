#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --job-name=nebular_props
#SBATCH --output=add_nebular_props.log

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# joblib (loky) spawns N_JOBS worker processes; keep BLAS single-threaded per
# worker to avoid oversubscribing the 128 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Single-stage joint 5-parameter fit (Plan A). Do NOT pass
# --use-informative-priors: with the 7-line SNR>5 te_mask the data constrain
# the full 5D posterior, and Plan A keeps the ne-Te-Av covariance that Plan B's
# independent-marginal feedback discards.
python3 desi_dwarfs/code/add_nebular_props.py --line-flux-type FLUX --overwrite-te-cache \
    /pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits
