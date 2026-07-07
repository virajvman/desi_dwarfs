#!/bin/bash
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=specz_observe
#SBATCH --output=specz_observe_%A_%a.log
#SBATCH --array=0-9
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=FAIL

# Generates + mock-observes mock spectra in chunks. With N_MOCKS_TOTAL=1e6
# and CHUNK_SIZE=1000 there are 1000 chunks; the array of 10 tasks handles
# 100 chunks each. Adjust --array and CHUNKS_PER_TASK together.

set -eo pipefail

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code/specz_incompleteness"

source /global/cfs/cdirs/desi/software/desi_environment.sh main

set -u

CHUNKS_PER_TASK=100
START=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_TASK ))
END=$(( START + CHUNKS_PER_TASK ))

cd "${CODE_DIR}"
python mock_observe.py --start "${START}" --end "${END}"
