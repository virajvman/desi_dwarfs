#!/bin/bash
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --job-name=specz_refsample
#SBATCH --output=specz_refsample_%j.log
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=FAIL

set -eo pipefail

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code/specz_incompleteness"

source /global/cfs/cdirs/desi/software/desi_environment.sh main

set -u

cd "${CODE_DIR}"
python build_reference_sample.py
