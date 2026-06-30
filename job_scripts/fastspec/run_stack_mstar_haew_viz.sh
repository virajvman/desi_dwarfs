#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=stack_mstar_haew_viz
#SBATCH --output=run_stack_mstar_haew_viz.log

# Mean coadds for paper visualization: custom log M* windows x fixed H-alpha EW
# bins. Full rest wavelength grid (0.2 A, 3600-9800 A). No FastSpecFit stage.
#
# Mass bins: (7.75, 8.25] and (7.50, 8.50] log Msun
# EW bins: ew_lt30, ew_30_100, ew_gt100 (Angstroms)
# Output: .../stack_files/mstar_viz_haew/stack_ALL_mstar_*.fits
#
# Submit:   sbatch job_scripts/fastspec/run_stack_mstar_haew_viz.sh
# Monitor:  squeue --me   |   tail -f run_stack_mstar_haew_viz.log

set -eo pipefail

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"

source /global/cfs/cdirs/desi/software/desi_environment.sh main

cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== Mean M* x H-alpha EW viz stacks (stack_mstar_haew_viz.py) ==="
python3 code/nebular_stuff/stack_mstar_haew_viz.py

echo "=== Done ==="
