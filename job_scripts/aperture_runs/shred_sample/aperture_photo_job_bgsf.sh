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
#SBATCH --time=02:00:00
#SBATCH --job-name=bgsf_shred
#SBATCH --output=aperture_shred_bgsf.log

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
SAMPLE="BGS_FAINT"
MAKE_CATS=false      # set true/false
RUN_APER=true
RUN_COG=true
RUN_SHIFTER=true

# Command-line args
BASE_ARGS="-sample $SAMPLE -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 10 -no_cnn_cut -use_sample shred -get_cnn_inputs"

# ------------------------------
# Run steps
# ------------------------------

if [ "$MAKE_CATS" = true ]; then
    python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -make_cats
fi

if [ "$RUN_APER" = true ]; then
    srun --cpu-bind=cores python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_aper
fi

if [ "$RUN_SHIFTER" = true ]; then
    shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
    
    srun --cpu-bind=cores shifter --image docker:legacysurvey/legacypipe:DR10.3.4 \
        python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -img_source -use_sample shred
    
    srun --kill-on-bad-exit=1 --cpu-bind=cores shifter --image docker:legacysurvey/legacypipe:DR10.3.4 \
        python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -parent_galaxy -bkg_source -blend_remove_source -use_sample shred
fi

if [ "$RUN_COG" = true ]; then
    srun --cpu-bind=cores python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_cog
fi

