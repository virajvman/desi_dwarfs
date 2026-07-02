#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=validate_psf_debug.log

# ============================================================
# Manual PSF validation gate: compares the container-built coadd PSF
# (build_coadd_psf inside dstndstn/cutouts) against the legacysurvey.org
# viewer's own coadd-psf URL endpoint on a small sample of objects. Run this
# and eyeball the results BEFORE ever submitting get_psfs_general.sbatch (the
# production batch) -- see README.md in this directory for the full
# rationale, tolerances, and how to read the output.
#
# qos=debug: fast turnaround, short walltime, single small job -- this does
# NOT write to the real PSF store (validate_url() only ever reads/compares,
# see many_psfs_general.py); it's safe to run repeatedly without touching
# production data. Needs network (URL fetch), unlike the production job.
# ============================================================

# NOTE: sbatch copies this script to a spool dir, so BASH_SOURCE does NOT
# point at the repo -- the script dir must be hardcoded (see CLAUDE.md).
SCRIPT_DIR="/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/job_scripts/image_cutouts/psf"

CATALOG_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits"
# Not actually written to in validate mode (validate_url() never calls
# psf_store's writers) -- pointing at the real store is harmless and
# convenient, but any writable path works.
OUTDIR="/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts/psfs"
VALIDATE_OUTDIR="/pscratch/sd/v/virajvm/dwarf_cutouts_debug/psf_validation"
N_VALIDATE=30

mkdir -p "$VALIDATE_OUTDIR"

echo "Starting PSF validation at $(date)"

shifter --image=dstndstn/cutouts:dvsro3 python3 \
    "${SCRIPT_DIR}/many_psfs_general.py" \
    --catalog-path "$CATALOG_PATH" \
    --outdir-data "$OUTDIR" \
    --validate-url "$N_VALIDATE" \
    --validate-outdir "$VALIDATE_OUTDIR" \
    --nompi

echo "Done at $(date)."
echo "Inspect metrics: ${VALIDATE_OUTDIR}/psf_validation_metrics.csv"
echo "Inspect panels : ${VALIDATE_OUTDIR}/psf_val_<TARGETID>.png"
echo "Only submit get_psfs_general.sbatch (production) after this looks good."
