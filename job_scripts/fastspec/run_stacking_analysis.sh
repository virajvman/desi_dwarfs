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
#SBATCH --time=06:00:00
#SBATCH --job-name=run_stacking_analysis
#SBATCH --output=run_stacking_analysis.log

# Full M* x H-alpha-EW stack pipeline (3 stages), as a single batch job so it
# runs on a scheduled compute node instead of an interactive Jupyter node.
# Mass: 0.5 dex (6-8), 0.25 dex (8-9.25). EW: <30, 30-100, >100 A.
#
# Submit:   sbatch job_scripts/fastspec/run_stacking_analysis.sh
# Monitor:  squeue --me   |   tail -f run_stacking_analysis.log
# (Or run the body directly on an exclusive interactive Perlmutter CPU node.)
#
# Stage 2 (run_stack_fastspec_haew_5pct.sh) sets up its OWN FastSpecFit
# environment in a child shell (`bash ...`), so its `module load fastspecfit`
# does not leak into stages 1 and 3, which use the main DESI environment.

set -e

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
