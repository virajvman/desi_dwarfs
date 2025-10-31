set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# ------------------------------
# Configurable flags
# ------------------------------
SAMPLE="SGA"
if [ "$SAMPLE" = "SGA" ]; then
    SAMPLE_TYPE="sga"
else
    SAMPLE_TYPE="shred"
fi
    
RUN_APER=true
RUN_COG=true
TGID=39627931810534981

# Command-line args
BASE_ARGS="-sample $SAMPLE -min 0 -max 100000 -run_parr -ncores 1 -overwrite -nchunks 1 -no_cnn_cut -use_sample $SAMPLE_TYPE -tgids $TGID"

# ------------------------------
# Run steps
# ------------------------------

if [ "$RUN_APER" = true ]; then
    python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_aper
fi

shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4

# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -img_source -use_sample shred

shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample $SAMPLE -parent_galaxy -bkg_source -blend_remove_source -use_sample $SAMPLE_TYPE -tgids $TGID

if [ "$RUN_COG" = true ]; then
    python3 desi_dwarfs/code/dwarf_photo_pipeline.py $BASE_ARGS -run_cog
fi
