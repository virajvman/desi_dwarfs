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
#SBATCH --job-name=run_stack_elg_noelg
#SBATCH --output=run_stack_elg_noelg.log

# ELG vs non-ELG stacking pipeline (3 stages), mirroring run_stacking_analysis.sh
# but for the separate ELG/NO-ELG product in stack_files/mstar/. Kept independent
# from the haew_5pct pipeline so either product can be (re)run on its own.
#
# Mass bins: 0.25 dex from log M*=6.0 to 9.5. Each (sample, mass) cell stacks
# when the catalog count >= 50; output FITS carry 1 mean row + 200 bootstrap rows
# (Scholte propagated-ivar error model, identical to the haew_5pct product).
#
# Submit:   sbatch job_scripts/fastspec/run_stack_elg_noelg.sh
# Monitor:  squeue --me   |   tail -f run_stack_elg_noelg.log
# (Or run the body directly on an exclusive interactive Perlmutter CPU node.)
#
# Stage 2 (run_stack_fastspec_elg_noelg.sh) sets up its OWN FastSpecFit
# environment in a child shell (`bash ...`), so its `module load fastspecfit`
# does not leak into stages 1 and 3, which use the main DESI environment.
# Stage 3 runs the direct method on the ELG/non-ELG product only (--products
# elg_noelg), with FLUX line fluxes and n_e fixed at 100 cm^-3, matching the
# haew_5pct run's direct-metallicity settings.

set -e

source /global/cfs/cdirs/desi/software/desi_environment.sh main

cd /global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs

echo "=== [1/3] Bootstrap stacking (stack_mstar_elg_vs_noelg.py) ==="
python3 code/stacking_analysis/stack_mstar_elg_vs_noelg.py

echo "=== [2/3] FastSpecFit on ELG / NO-ELG stacks ==="
bash job_scripts/fastspec/run_stack_fastspec_elg_noelg.sh

echo "=== [3/3] Direct-method metallicity (n_e fixed at 100 cm^-3) ==="
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python3 code/nebular_stuff/stack_direct_metallicity.py \
    --products elg_noelg --line-flux-type FLUX --fix-ne100

echo "=== Pipeline complete ==="
