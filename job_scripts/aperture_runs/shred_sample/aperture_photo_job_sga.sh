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
#SBATCH --time=08:00:00
#SBATCH --job-name=sga_run
#SBATCH --output=aperture_shred_sga.log

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
SAMPLE="SGA"
MAKE_CATS=false      # set true/false
RUN_APER=true
RUN_COG=true
RUN_SHIFTER=true

# Match dwarf_photo_pipeline consolidated catalog for tractor incremental mode
END_NAME=""
OVERWRITE_PHOTOMETRY=true
TRACTOR_PHOTO_ARGS=(-end_name "$END_NAME")
if [ "$OVERWRITE_PHOTOMETRY" = true ]; then
    # -overwrite_photometry: skip the catalog-incremental TARGETID filter.
    # -overwrite: also force regeneration of per-object tractor model .npy files
    #   (filter_existing_sources skips objects whose models exist when this is
    #   off). A full overwrite rerun needs BOTH, else tractor reuses stale models
    #   with freshly regenerated aperture/COG outputs.
    TRACTOR_PHOTO_ARGS+=(-overwrite_photometry -overwrite)
fi

# Command-line args
BASE_ARGS="-sample $SAMPLE -min 0 -max 100000 -run_parr -ncores 128 -overwrite -nchunks 5 -no_cnn_cut -use_sample sga"

# Propagate the full-overwrite toggle to the photometry pipeline too. Without
# it, dwarf_photo_pipeline.py runs in incremental mode and exits early on every
# object already in the output catalog ("No new objects to process"), skipping
# make_cats/run_aper -- so the downstream tractor step then crashes on the
# missing source_cat_f.fits it expects.
if [ "$OVERWRITE_PHOTOMETRY" = true ]; then
    BASE_ARGS="$BASE_ARGS -overwrite_photometry"
fi

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
    
    srun --cpu-bind=cores shifter --env=PYTHONUSERBASE=$HOME/.local-legacypipe --image docker:legacysurvey/legacypipe:DR10.3.4 \
        python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -img_source -use_sample sga "${TRACTOR_PHOTO_ARGS[@]}"
    
    srun --kill-on-bad-exit=1 --cpu-bind=cores shifter --env=PYTHONUSERBASE=$HOME/.local-legacypipe --image docker:legacysurvey/legacypipe:DR10.3.4 \
        python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -parent_galaxy -bkg_source -blend_remove_source -use_sample sga "${TRACTOR_PHOTO_ARGS[@]}"
fi


if [ "$RUN_COG" = true ]; then
    srun --cpu-bind=cores python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_cog
fi
