#!/bin/bash
set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -run_aper -no_cnn_cut -use_sample sga
# shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample SGA -parent_galaxy -use_sample sga
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -no_cnn_cut -use_sample sga -run_cog 

# -bkg_source -blend_remove_source
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 50 -run_parr -ncores 64 -overwrite -nchunks 1 -run_aper -no_cnn_cut -use_sample sga
# # shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
# # shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample SGA -parent_galaxy -bkg_source -blend_remove_source -use_sample sga -max_num 1000
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 50 -run_parr -ncores 64 -overwrite -nchunks 1 -no_cnn_cut -use_sample sga -run_cog
# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample SGA -img_source -use_sample sga

