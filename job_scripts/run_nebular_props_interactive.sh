#!/bin/bash -l
# Interactive version of run_nebular_props.sh.
# Run this directly in a terminal on an exclusive Perlmutter CPU node
# (e.g. a node spawned through JupyterHub), NOT via sbatch.

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
    /pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits 2>&1 | tee add_nebular_props_interactive.log
