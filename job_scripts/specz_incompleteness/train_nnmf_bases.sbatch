#!/bin/bash
#SBATCH --account=desi_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --job-name=specz_nnmf_bases
#SBATCH --output=specz_nnmf_bases_%j.log
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=FAIL

set -eo pipefail

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code/specz_incompleteness"

# Same environment used for the production NNMF runs (needs cupy + nearly_nmf
# on top of the DESI stack).
source /global/cfs/cdirs/desi/software/desi_environment.sh main

set -u

cd "${CODE_DIR}"
python train_nnmf_bases.py
