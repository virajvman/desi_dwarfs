#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=consolidate_recon
#SBATCH --output=consolidate_reconstructed.log

set -eo pipefail

# Canonical repo paths are hardcoded (SLURM copies the script to spool, so
# BASH_SOURCE/$0 would not resolve to the repo). See CLAUDE.md.
REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code"

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# HDF5 shard stores on CFS/Lustre: disable file locking (we partition by brick,
# so each shard has a single writer regardless).
export HDF5_USE_FILE_LOCKING=FALSE

# ------------------------------
# Configurable flags
# ------------------------------
# Shred samples and the SGA sample are packed into the SAME unified store. They
# are run as two SEQUENTIAL invocations below (not concurrent): the store is
# brick-sharded with one-writer-per-shard, and a brick can hold both shred and
# SGA objects, so they must not write the same shard at the same time.
SHRED_SAMPLES="BGS_BRIGHT,BGS_FAINT,ELG,LOWZ"
SGA_SAMPLE="SGA"
VERSION="v1"
END_NAME=""                  # must match the w_aper_mags catalog suffix used by the run
OVERWRITE_PHOTOMETRY=true    # true: refresh cubes already in the store;
                             # false: incremental (skip objects already present)

COMMON_ARGS=(-ncores 128 -version "$VERSION" -end_name "$END_NAME")
if [ "$OVERWRITE_PHOTOMETRY" = true ]; then
    COMMON_ARGS+=(-overwrite_photometry)
fi

# echo "=== Consolidating shred samples ==="
# srun --cpu-bind=cores python3 "${CODE_DIR}/consolidate_reconstructed.py" \
    # -sample "$SHRED_SAMPLES" -use_sample shred "${COMMON_ARGS[@]}"

echo "=== Consolidating SGA sample ==="
srun --cpu-bind=cores python3 "${CODE_DIR}/consolidate_reconstructed.py" \
    -sample "$SGA_SAMPLE" -use_sample sga "${COMMON_ARGS[@]}"
