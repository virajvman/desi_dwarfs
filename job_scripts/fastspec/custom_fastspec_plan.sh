#!/bin/bash -l

# Match the production run (run_custom_fastspec_job.sh): fastspecfit/3.4.3 module.
# (The old desiutil/desispec/desitarget/speclite swaps were for the 3.4.1-era
# stack and now fail to load against `main` -- let `main` provide them.)
export FASTSPECFIT_VERSION=3.4.3
source /dvs_ro/common/software/desi/desi_environment.sh main
module load fastspecfit/${FASTSPECFIT_VERSION}

export DESI_SPECTRO_REDUX=/dvs_ro/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/dvs_ro/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/dvs_ro/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/dvs_ro/cfs/cdirs/desi/public/external/templates/fastspecfit

mpi-fastspecfit \
    --samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits \
    --outdir-data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/ \
    --emlinesfile=/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines.ecsv \
    --templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits \
    --specprod iron --mp=16 --nompi --plan