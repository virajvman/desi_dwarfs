#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=07:00:00
#SBATCH --job-name=elg_clean
#SBATCH --output=aperture_clean_elg.log

set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# HDF5 cutout shard store: readers need file locking disabled on CFS/Lustre
export HDF5_USE_FILE_LOCKING=FALSE

# ------------------------------
# Configurable flags
# ------------------------------
SAMPLE="ELG"
MAKE_CATS=true      # set true/false
RUN_APER=true
RUN_COG=true
RUN_SHIFTER=true

# Match dwarf_photo_pipeline consolidated catalog for tractor incremental mode
END_NAME=""
OVERWRITE_PHOTOMETRY=false
TRACTOR_PHOTO_ARGS=(-end_name "$END_NAME")
if [ "$OVERWRITE_PHOTOMETRY" = true ]; then
    TRACTOR_PHOTO_ARGS+=(-overwrite_photometry)
fi

# Command-line args
BASE_ARGS="-sample $SAMPLE -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 50 -no_cnn_cut -use_sample clean"

# ------------------------------
# Run steps
# ------------------------------

if [ "$MAKE_CATS" = true ]; then
    srun --cpu-bind=cores python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -make_cats
fi

if [ "$RUN_APER" = true ]; then
    srun --cpu-bind=cores python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_aper
fi

if [ "$RUN_SHIFTER" = true ]; then
    shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
    
    srun --cpu-bind=cores shifter --env=PYTHONUSERBASE=$HOME/.local-legacypipe --image docker:legacysurvey/legacypipe:DR10.3.4 \
        python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -img_source -use_sample clean "${TRACTOR_PHOTO_ARGS[@]}"
    
    srun --cpu-bind=cores shifter --env=PYTHONUSERBASE=$HOME/.local-legacypipe --image docker:legacysurvey/legacypipe:DR10.3.4 \
        python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -parent_galaxy -bkg_source -blend_remove_source -use_sample clean "${TRACTOR_PHOTO_ARGS[@]}"
fi

if [ "$RUN_COG" = true ]; then
    srun --cpu-bind=cores python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_cog
fi
