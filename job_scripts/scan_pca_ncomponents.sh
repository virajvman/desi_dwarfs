#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=150GB
#SBATCH --time=00:30:00
#SBATCH --output=pca_ncomponent_scan.log

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code"

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
python3 -u ${CODE_DIR}/nnmf_pca_analysis/scan_pca_ncomponents.py
