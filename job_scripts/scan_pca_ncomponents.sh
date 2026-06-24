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
#SBATCH --time=00:30:00
#SBATCH --output=pca_ncomponent_scan.log

set -eo pipefail

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code"

# Run under NERSC's standalone, GPU-optimized PyTorch module -- NOT the DESI
# software stack. `desi_environment.sh` activates a py3.13 conda env and exports
# PYTHONPATH at its site-packages; if that leaks into the module's py3.12
# interpreter, numpy's C-extensions fail to import. So we start clean: never
# source DESI here, and scrub any Python env vars inherited from the submit
# shell (sbatch exports the submitting environment by default).
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
module load pytorch/2.11.0

# scan_pca_ncomponents.py is self-contained (numpy / matplotlib / h5py / torch
# only) -- no DESI imports -- so the bare pytorch module satisfies it.
python3 -u ${CODE_DIR}/nnmf_pca_analysis/scan_pca_ncomponents.py
