#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=get_psfs_general.log

# ============================================================
# Build EMPIRICAL Legacy Surveys coadd PSFs per object -> per-brick HDF5
# shards in the PSF store (code/psf_store.py). Sibling of the image-cutout
# job (get_imgs_general.sbatch); reproduces the viewer coadd-psf inside the
# dstndstn/cutouts container (CFS reads, no network). Never run two PSF-writing
# jobs against the store at the same time.
#
# BEFORE the production run, validate container-vs-URL agreement (manual gate):
# see README.md in this directory, or just submit validate_psf_debug.sbatch
# (qos=debug, fast turnaround). Inspect the panels + metrics it prints, THEN
# submit this job.
#
# Incremental: objects already in the PSF manifest, or tombstoned in
# permanently_failed.csv, are skipped -- this also means running MULTIPLE
# catalogs through the SAME store (see CATALOG_PATHS below) costs nothing
# extra for any TARGETIDs the catalogs happen to share (the store is unified
# across sample classes, see psf_store.get_store_dir()'s own docstring).
# ============================================================

# One or more input catalogs to process, in order, against the SAME PSF
# store -- e.g. BGS_BRIGHT + SGA. Add more paths here for additional samples;
# each just needs RA_COL/DEC_COL/ID_COL/brick columns (same for every sample
# so far -- override below if a future catalog uses different names).
CATALOG_PATHS=(
    "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits"
    "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_SGA_sga_catalog_w_aper_mags.fits"
)
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

for CATALOG_PATH in "${CATALOG_PATHS[@]}"; do
    if [ ! -f "$CATALOG_PATH" ]; then
        echo "FAILED: catalog not found: $CATALOG_PATH"
        exit 1
    fi
    echo "---- Processing $CATALOG_PATH at $(date) ----"
    cmd="sh $SCRIPT_DIR/psfs_cnn_general.sh $N $mp $CATALOG_PATH $OUTDIR $RA_COL $DEC_COL $ID_COL $EXTRA_ARGS"
    echo $cmd
    $cmd

    if [ $? -eq 0 ]; then
        echo "SUCCESS: $CATALOG_PATH done at $(date)"
    else
        echo "FAILED on $CATALOG_PATH at $(date)"
        exit 1
    fi
done

echo "All catalogs done at $(date)"
