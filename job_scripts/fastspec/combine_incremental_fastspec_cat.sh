#!/bin/bash -l
#SBATCH --job-name=combine_fastspec_incr
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/combine_fastspec_incr_%j.out
#SBATCH --error=logs/combine_fastspec_incr_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=virajvm@stanford.edu
#
# Combine-only step for the incremental fastspec workflow.
# Use when the fit in run_incremental_fastspec_job.sh finished but combine failed.
#
#   sbatch job_scripts/fastspec/combine_incremental_fastspec_cat.sh
#
# Set REPLACE_ORIGINAL=1 to overwrite the canonical merged catalog in-place.

REPLACE_ORIGINAL=${REPLACE_ORIGINAL:-0}

manifest=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs_fastspec_incremental.fits.manifest.json
fastspec_merged=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/iron/catalogs/fastspec-iron-dr1-dwarfs.fits
out_merged=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/iron/catalogs/fastspec-iron-dr1-dwarfs-v2.fits

# NOTE: sbatch copies this script to a spool dir; BASH_SOURCE does NOT point at the repo.
REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code"

export FASTSPECFIT_VERSION=3.4.3
source /dvs_ro/common/software/desi/desi_environment.sh main
module load fastspecfit/${FASTSPECFIT_VERSION}

mkdir -p logs

COMBINE_ARGS=(
    --manifest="${manifest}"
    --fastspec-merged="${fastspec_merged}"
    --out-merged="${out_merged}"
)
if (( REPLACE_ORIGINAL )); then
    COMBINE_ARGS+=(--replace-original)
fi

python3 "${CODE_DIR}/combine_incremental_fastspec.py" "${COMBINE_ARGS[@]}"
