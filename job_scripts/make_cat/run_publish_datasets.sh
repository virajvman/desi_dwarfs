#!/bin/bash -l
#
# run_publish_datasets.sh -- publish the companion data products (spectra +
# image cutouts) to the group-readable CFS release directory, next to the
# catalog FITS that run_nebular_props.sh already publishes.
#
# This is a ONE-TIME / on-demand publish (not chained into submit_make_cat.sh):
# the spectra .h5 is already built, and the imaging is consolidated from
# already-downloaded shards. Re-run by hand whenever those inputs change:
#
#     sbatch run_publish_datasets.sh
#
# It writes:
#   iron/spectra/desi_dr1_dwarf_catalog_spectra.h5   (copied as-is)
#   iron/images/desi_dr1_dwarf_catalog_images.h5      (consolidated: catalog-
#                                                      matched, de-duplicated)
# and repairs the group/permissions of the existing catalog FITS so DESI
# collaborators can actually read the whole release (see step 0/1).

#SBATCH --account=desi
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --job-name=publish_datasets
#SBATCH --output=publish_datasets.log

# set -e : abort (and skip later publishes) on any failure. -u is deliberately
#          NOT used -- desi_environment.sh references unset vars.
set -eo pipefail

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs
DESI_GROUP=desi

# --- Release layout (parallels run_nebular_props.sh's PUBLISH_DIR) ------------
PUBLISH_ROOT=/global/cfs/cdirs/desi/users/virajvm/desi_dwarf_cats/iron
SPEC_DIR=${PUBLISH_ROOT}/spectra
IMG_DIR=${PUBLISH_ROOT}/images
CATALOG_CFS=${PUBLISH_ROOT}/desi_dr1_dwarf_catalog.fits

SPEC_DEST=${SPEC_DIR}/desi_dr1_dwarf_catalog_spectra.h5
IMG_DEST=${IMG_DIR}/desi_dr1_dwarf_catalog_images.h5

# --- Inputs (already downloaded on pscratch) ---------------------------------
SPEC_SRC=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/data/desi_dr1_dwarf_catalog_spectra.h5
CHUNKS_GLOB='/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/ssl_shred_data/h5_datasets/data_chunk_*.h5'

# Clean up temp dotfiles if the job dies mid-write.
SPEC_TMP="${SPEC_DIR}/.$(basename "${SPEC_DEST}").tmp.${SLURM_JOB_ID}"
IMG_TMP="${IMG_DIR}/.$(basename "${IMG_DEST}").tmp.${SLURM_JOB_ID}"
trap 'rm -f "${SPEC_TMP}" "${IMG_TMP}"' EXIT

# --- 0. Release dirs: group desi + setgid ------------------------------------
# chmod 2750 sets the setgid bit so every file created here (now and in future
# publishes) inherits group desi automatically. The parent iron/ dir currently
# lacks setgid, which is why the catalog FITS ended up group-virajvm/0640 and
# unreadable by collaborators -- fixed in step 1.
for d in "${PUBLISH_ROOT}" "${SPEC_DIR}" "${IMG_DIR}"; do
    mkdir -p "${d}"
    chgrp "${DESI_GROUP}" "${d}"
    chmod 2750 "${d}"
done

# --- 1. Make the already-published catalog group-readable --------------------
if [[ -f "${CATALOG_CFS}" ]]; then
    chgrp "${DESI_GROUP}" "${CATALOG_CFS}"
    chmod 640 "${CATALOG_CFS}"
    echo "Fixed group/perms: ${CATALOG_CFS}"
fi

# --- 2. Spectra: atomic publish (single file already built) ------------------
echo "Publishing spectra -> ${SPEC_DEST}"
cp "${SPEC_SRC}" "${SPEC_TMP}"
chgrp "${DESI_GROUP}" "${SPEC_TMP}"
chmod 640 "${SPEC_TMP}"
mv -f "${SPEC_TMP}" "${SPEC_DEST}"
echo "Published: ${SPEC_DEST}"

# --- 3. Imaging: consolidate (catalog-match + dedup) then atomic publish -----
echo "Consolidating imaging -> ${IMG_DEST}"
python3 "${REPO}/code/save_cutouts_h5.py" \
    --catalog "${CATALOG_CFS}" \
    --chunks-glob "${CHUNKS_GLOB}" \
    --out "${IMG_TMP}"
chgrp "${DESI_GROUP}" "${IMG_TMP}"
chmod 640 "${IMG_TMP}"
mv -f "${IMG_TMP}" "${IMG_DEST}"
echo "Published: ${IMG_DEST}"

echo "All datasets published under ${PUBLISH_ROOT}"
