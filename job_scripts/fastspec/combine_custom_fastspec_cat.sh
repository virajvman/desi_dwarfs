#!/bin/bash
#SBATCH --job-name=combine_dwarfs
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=logs/combine_fastspec.out
#SBATCH --error=logs/combine_fastspec.err
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

srun -n 1 mpi-fastspecfit \
    --samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits \
    --outdir-data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/ \
    --specprod iron \
    --merge --mp 64