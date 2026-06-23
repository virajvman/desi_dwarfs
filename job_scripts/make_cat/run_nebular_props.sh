#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256GB
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --job-name=nebular_props
#SBATCH --output=add_nebular_props.log

# set -e  : abort the job (and skip the publish step below) if the nebular fit
#           fails, and fail loudly if the publish itself errors.
# set -u  : deliberately NOT used -- desi_environment.sh references unset vars
#           (e.g. DESI_ROOT) and would abort under nounset. See CLAUDE.md.
set -eo pipefail

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# joblib (loky) spawns N_JOBS worker processes; keep BLAS single-threaded per
# worker to avoid oversubscribing the 128 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Catalog this stage edits in place (appends the SPEC_DERIVED HDU).
CATALOG=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits

# Group-readable publish location on CFS (permanent; pscratch is purged).
PUBLISH_DIR=/global/cfs/cdirs/desi/users/virajvm/desi_dwarf_cats/iron
PUBLISH_DEST="${PUBLISH_DIR}/desi_dr1_dwarf_catalog.fits"

# Single-stage joint 5-parameter fit (Plan A). Do NOT pass
# --use-informative-priors: with the 7-line SNR>5 te_mask the data constrain
# the full 5D posterior, and Plan A keeps the ne-Te-Av covariance that Plan B's
# independent-marginal feedback discards.
python3 desi_dwarfs/code/add_nebular_props.py --line-flux-type FLUX --overwrite-te-cache \
    "${CATALOG}"

# --- Publish the finished catalog to the group-readable CFS location ---------
# Reached only if the fit above succeeded (set -e). Atomic publish: copy to a
# temp dotfile in the destination dir, fix perms, then rename into place so
# desi-group readers never observe a partially written FITS.
#   - group desi is inherited automatically from the dest dir's setgid bit
#     (2750), so no chgrp is needed.
#   - chmod 640  : a fresh cp lands as 0644 under the default umask (022), i.e.
#                  world-readable. Force group-read-only, no world, no execute.
PUBLISH_TMP="${PUBLISH_DIR}/.desi_dr1_dwarf_catalog.fits.tmp.${SLURM_JOB_ID}"
echo "Publishing catalog -> ${PUBLISH_DEST}"
cp "${CATALOG}" "${PUBLISH_TMP}"
chmod 640 "${PUBLISH_TMP}"
mv -f "${PUBLISH_TMP}" "${PUBLISH_DEST}"
echo "Published: ${PUBLISH_DEST}"
