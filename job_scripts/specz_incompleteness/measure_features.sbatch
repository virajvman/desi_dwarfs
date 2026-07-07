#!/bin/bash
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --job-name=specz_features
#SBATCH --output=specz_features_%A_%a.log
#SBATCH --array=0-9
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=FAIL

# Measures NN features on the noisy mocks (per chunk). Same chunk layout as
# generate_and_observe.sbatch. After all array tasks finish, consolidate:
#   python measure_features.py --consolidate

set -eo pipefail

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code/specz_incompleteness"

source /global/cfs/cdirs/desi/software/desi_environment.sh main

set -u

CHUNKS_PER_TASK=100
START=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_TASK ))
END=$(( START + CHUNKS_PER_TASK ))

cd "${CODE_DIR}"
python measure_features.py --start "${START}" --end "${END}"
