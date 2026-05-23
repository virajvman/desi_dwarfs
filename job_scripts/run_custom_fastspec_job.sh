#!/bin/bash
#SBATCH --job-name=fastspec_dwarfs_400k
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=128
#SBATCH --time=23:00:00
#SBATCH --output=logs/fastspec_400k_%j.out
#SBATCH --error=logs/fastspec_400k_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=virajvm@stanford.edu

source /global/common/software/desi/desi_environment.sh main
module load fastspecfit/main
module swap desiutil/3.4.3
module swap desispec/0.68.1
module swap desitarget/2.8.0
module swap desimodel/0.19.2
module swap speclite/v0.20

export DESI_SPECTRO_REDUX=/global/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/global/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/global/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/global/cfs/cdirs/desi/public/external/templates/fastspecfit

mkdir -p logs

srun -n 10 -c 128 mpi-fastspecfit \
    --samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits \
    --outdir-data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/ \
    --emlinesfile=/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines.ecsv \
    --templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits \
    --nmonte=100 --vdisp-nominal 100 --vdisp-bounds 50 200 --mp=120 --specprod iron

srun -n 1 mpi-fastspecfit \
    --samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits \
    --outdir-data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/ \
    --specprod iron \
    --merge