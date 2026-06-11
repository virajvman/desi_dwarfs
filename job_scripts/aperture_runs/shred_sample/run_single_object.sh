set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# HDF5 cutout shard store: readers need file locking disabled on CFS/Lustre
export HDF5_USE_FILE_LOCKING=FALSE

# ------------------------------
# Configurable flags
# ------------------------------
SAMPLE="SGA"
if [ "$SAMPLE" = "SGA" ]; then
    SAMPLE_TYPE="sga"
else
    SAMPLE_TYPE="shred"
fi

MAKE_CATS=false      # set true/false
RUN_APER=true
RUN_COG=true
RUN_SHIFTER=true
TGID=39628516139993581

# Match dwarf_photo_pipeline consolidated catalog for tractor incremental mode.
# With -tgids, tractor_model.py skips photometry-catalog incremental filtering.
END_NAME=""
OVERWRITE_PHOTOMETRY=false
TRACTOR_PHOTO_ARGS=(-end_name "$END_NAME")
if [ "$OVERWRITE_PHOTOMETRY" = true ]; then
    TRACTOR_PHOTO_ARGS+=(-overwrite_photometry)
fi

# Command-line args
BASE_ARGS="-sample $SAMPLE -min 0 -max 100000 -run_parr -ncores 1 -overwrite -nchunks 1 -no_cnn_cut -use_sample $SAMPLE_TYPE -tgids $TGID"

# ------------------------------
# Run steps
# ------------------------------

if [ "$MAKE_CATS" = true ]; then
    python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -make_cats
fi

if [ "$RUN_APER" = true ]; then
    python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_aper
fi

if [ "$RUN_SHIFTER" = true ]; then
    shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
    
    shifter --env=PYTHONUSERBASE=$HOME/.local-legacypipe --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -img_source -use_sample $SAMPLE_TYPE "${TRACTOR_PHOTO_ARGS[@]}"
    
    shifter --env=PYTHONUSERBASE=$HOME/.local-legacypipe --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -parent_galaxy -bkg_source -blend_remove_source -use_sample $SAMPLE_TYPE -tgids $TGID "${TRACTOR_PHOTO_ARGS[@]}"

fi

if [ "$RUN_COG" = true ]; then
    python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_cog
fi
