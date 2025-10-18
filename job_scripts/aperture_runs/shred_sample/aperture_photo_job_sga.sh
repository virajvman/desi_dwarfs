#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192GB
#SBATCH --time=05:00:00
#SBATCH --job-name=sga_run
#SBATCH --output=aperture_shred_sga.log

set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------
# Configurable flags
# ------------------------------
SAMPLE="SGA"
MAKE_CATS=false      # set true/false
RUN_APER=true
RUN_COG=true

# Command-line args
BASE_ARGS="-sample $SAMPLE -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 100 -no_cnn_cut -use_sample sga -get_cnn_inputs"

# ------------------------------
# Run steps
# ------------------------------

if [ "$MAKE_CATS" = true ]; then
    python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -make_cats
fi

if [ "$RUN_APER" = true ]; then
    srun --cpu-bind=cores python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_aper
fi

shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4

# srun --cpu-bind=cores shifter --image docker:legacysurvey/legacypipe:DR10.3.4 \
#     python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -img_source -use_sample sga

srun --cpu-bind=cores shifter --image docker:legacysurvey/legacypipe:DR10.3.4 \
    python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -parent_galaxy -bkg_source -blend_remove_source -use_sample sga

if [ "$RUN_COG" = true ]; then
    srun --cpu-bind=cores python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_cog
fi
