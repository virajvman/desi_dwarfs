#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=get_psfs_general.log

# ============================================================
# Build EMPIRICAL Legacy Surveys coadd PSFs per object -> per-brick HDF5
# shards in the PSF store (code/psf_store.py). Sibling of the image-cutout
# job (get_imgs_general.sbatch); reproduces the viewer coadd-psf inside the
# dstndstn/cutouts container (CFS reads, no network). Never run two PSF-writing
# jobs against the store at the same time.
#
# BEFORE the production run, validate container-vs-URL agreement (manual gate):
#   shifter --image dstndstn/cutouts:dvsro3 python3 \
#     <repo>/job_scripts/image_cutouts/psf/many_psfs_general.py \
#     --catalog-path $CATALOG_PATH --outdir-data $OUTDIR \
#     --validate-url 30 --nompi
# Inspect the panels + metrics it prints, THEN submit this job.
#
# Incremental: objects already in the PSF manifest, or tombstoned in
# permanently_failed.csv, are skipped.
# ============================================================
CATALOG_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits"
# PSF store (must match psf_store.PSFS_DIR) -- sibling of the image store
OUTDIR="/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts/psfs"
RA_COL="RA"
DEC_COL="DEC"
ID_COL="TARGETID"

N=1       # number of nodes (must match #SBATCH --nodes)
mp=4      # multiprocessing workers per MPI rank

# Extra flags forwarded to many_psfs_general.py
# e.g. "--require-bands grz --retry-failed --bands g,r,z --box 63"
EXTRA_ARGS=""
# ============================================================

# NOTE: sbatch copies this script to a spool dir, so BASH_SOURCE does NOT
# point at the repo -- the script dir must be hardcoded.
SCRIPT_DIR="/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/job_scripts/image_cutouts/psf"

echo "Starting at $(date)"

cmd="sh $SCRIPT_DIR/psfs_cnn_general.sh $N $mp $CATALOG_PATH $OUTDIR $RA_COL $DEC_COL $ID_COL $EXTRA_ARGS"
echo $cmd
$cmd

if [ $? -eq 0 ]; then
    echo "SUCCESS: PSFs done at $(date)"
else
    echo "FAILED at $(date)"
    exit 1
fi

echo "All done at $(date)"
