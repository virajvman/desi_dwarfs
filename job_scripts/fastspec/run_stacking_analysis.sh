#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=run_stacking_analysis
#SBATCH --output=run_stacking_analysis.log

# Full stack pipeline (3 stages), as a single batch job so it runs on a
# scheduled compute node instead of an interactive Jupyter node.
# Products: EW-binned (M* x H-alpha EW) + mass-only (M* only, mstar_only/)
#           + mass viz (integer-centered 0.5-dex bins, mstar_viz/, FITS only).
# Mass: 0.5 dex (6-8), 0.25 dex (8-9.25). EW: <30, 30-100, >100 A (EW product only).
# Stage 1 also writes the mstar_viz/ stacks (full lambda grid to 9800 A); stages
# 2-3 deliberately ignore mstar_viz/ (visualization only, no fastspec/metallicity).
#
# Submit:   sbatch job_scripts/fastspec/run_stacking_analysis.sh
# Monitor:  squeue --me   |   tail -f run_stacking_analysis.log
# (Or run the body directly on an exclusive interactive Perlmutter CPU node.)
#
# Stage 2 (run_stack_fastspec_haew_5pct.sh) sets up its OWN FastSpecFit
# environment in a child shell (`bash ...`), so its `module load fastspecfit`
# does not leak into stages 1 and 3, which use the main DESI environment.
# Stage 2 fits both STACK_PATH/ (EW-binned) and STACK_PATH/mstar_only/.
# Stage 3 runs direct-method metallicity on both products (default --products both).

set -e

source /global/cfs/cdirs/desi/software/desi_environment.sh main

cd /global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs

# Unbuffered Python so the .log shows live per-bin progress -- SLURM block-buffers
# stdout, so a killed job otherwise loses all its progress prints. One BLAS thread
# per process: stage 1's bootstrap coadds run on a process pool (BOOT_NJOBS),
# stage 3's UltraNest parallelizes per row, and stage 2's stackfit --mp forks
# workers; in every case we want a single thread per process to avoid CPU
# oversubscription and the fork-with-threaded-BLAS hang.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== [1/3] Bootstrap stacking (stack_mstar_haew_5pct.py) ==="
python3 code/nebular_stuff/stack_mstar_haew_5pct.py

echo "=== [2/3] FastSpecFit on stacks ==="
bash job_scripts/fastspec/run_stack_fastspec_haew_5pct.sh

echo "=== [3/3] Direct-method metallicity (n_e fixed at 100 cm^-3) ==="
python3 code/nebular_stuff/stack_direct_metallicity.py \
    --line-flux-type FLUX --fix-ne100

echo "=== Pipeline complete ==="
