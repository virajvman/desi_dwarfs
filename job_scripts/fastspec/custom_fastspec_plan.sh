#!/bin/bash -l

# Match the env in etc/fastspecfit-env.sh
source /dvs_ro/common/software/desi/desi_environment.sh main
module swap desiutil/3.6.1
module swap desispec/0.71.2
module swap desitarget/4.7.2
module swap speclite/v1.0.0
module load fastspecfit/3.4.1

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