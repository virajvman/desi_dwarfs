#! /bin/bash
# Step 0: Make sure the correct version and branch of fastspec is checked out in the repo!!!

source /global/common/software/desi/desi_environment.sh main
module load fastspecfit/main
module swap desiutil/3.4.3
module swap desispec/0.68.1
module swap desitarget/2.8.0
module swap desimodel/0.19.2
module swap speclite/v0.20

# pip install /global/homes/d/dscholte/packages/fastspecfit/

# export PYTHONPATH=/global/u2/d/dscholte/packages/fastspecfit/py:$PYTHONPATH
# export PATH=/global/u2/d/dscholte/packages/fastspecfit/bin:$PATH

export DESI_SPECTRO_REDUX=/global/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/global/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/global/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/global/cfs/cdirs/desi/public/external/templates/fastspecfit

# srun -n 1 -c 128 mpi-fastspecfit \
#     --samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/temp/trial_iron.fits \
#     --outdir-data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/ \
#     --emlinesfile=/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines.ecsv \
#     --templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits \
#     --nmonte=100 --vdisp-nominal 100 --vdisp-bounds 50 500 --mp=120 --specprod iron

srun -n 1 mpi-fastspecfit \
    --samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/temp/trial_iron.fits \
    --outdir-data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/ \
    --specprod iron \
    --merge