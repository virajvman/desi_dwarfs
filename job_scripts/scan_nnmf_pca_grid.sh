#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=150GB
#SBATCH --time=02:00:00
#SBATCH --output=nnmf_pca_grid_scan.log

set -eo pipefail

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code"

# Run under NERSC's standalone, GPU-optimized PyTorch module -- NOT the DESI
# software stack (same reasoning as scan_pca_ncomponents.sh). Sourcing
# desi_environment.sh would leak a py3.13 conda env / PYTHONPATH into the
# module's py3.12 interpreter and break numpy's C-extensions, so we start clean.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
module load pytorch/2.11.0

# scan_nnmf_pca_grid.py is self-contained (numpy / matplotlib / h5py / torch
# only) -- no DESI imports -- so the bare pytorch module satisfies it. It
# consumes templates_ntemp{n}.npy + hcoeffs_*_ntemp{n}.npy written by
# scan_nnmf_ntemplates.py, so it must run AFTER that job succeeds (afterok).
python3 -u ${CODE_DIR}/nnmf_pca_analysis/scan_nnmf_pca_grid.py
