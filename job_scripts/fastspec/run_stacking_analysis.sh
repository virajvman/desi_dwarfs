#!/bin/bash
# Full M* x H-alpha-EW stack pipeline (includes new [9.0, 9.25] mass bin).
# Run on a Perlmutter login or compute node with DESI env + pscratch access.

set -euo pipefail

source /global/cfs/cdirs/desi/software/desi_environment.sh main

cd /global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs

echo "=== [1/3] Bootstrap stacking (stack_mstar_haew_5pct.py) ==="
python3 code/nebular_stuff/stack_mstar_haew_5pct.py

echo "=== [2/3] FastSpecFit on stacks ==="
bash job_scripts/fastspec/run_stack_fastspec_haew_5pct.sh

echo "=== [3/3] Direct-method metallicity (SII n_e default) ==="
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 code/nebular_stuff/stack_direct_metallicity.py \
    --line-flux-type BOXFLUX --density-diagnostic SII

echo "=== Pipeline complete ==="