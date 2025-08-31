
#!/bin/bash
set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample LOWZ,ELG,BGS_BRIGHT,BGS_FAINT -min 0 -max 200000 -run_parr -ncores 64 -overwrite -nchunks 5 -no_cnn_cut -use_sample shred -make_cats



shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample SGA -img_source -use_sample sga
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -run_aper -no_cnn_cut -use_sample sga
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample SGA -bkg_source -blend_remove_source -use_sample sga
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -run_aper -no_cnn_cut -use_sample sga -run_cog 
