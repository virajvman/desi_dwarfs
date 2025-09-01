
#!/bin/bash
set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4

# # shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_BRIGHT -img_source -use_sample shred
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -run_aper -no_cnn_cut -use_sample shred
# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_BRIGHT -parent_galaxy -bkg_source -blend_remove_source -use_sample shred
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -no_cnn_cut -use_sample shred -run_cog 

# # shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample ELG -img_source -use_sample shred
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -run_aper -no_cnn_cut -use_sample shred
# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample ELG -parent_galaxy -bkg_source -blend_remove_source -use_sample shred
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -no_cnn_cut -use_sample shred -run_cog 


# # # shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample LOWZ -img_source -use_sample shred
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample LOWZ -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 5 -run_aper -no_cnn_cut -use_sample shred
# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample LOWZ -parent_galaxy -bkg_source -blend_remove_source -use_sample shred
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample LOWZ -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 5 -no_cnn_cut -use_sample shred -run_cog 

# # shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_FAINT -img_source -use_sample shred
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 5 -run_aper -no_cnn_cut -use_sample shred
# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_FAINT -parent_galaxy -bkg_source -blend_remove_source -use_sample shred
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 5 -no_cnn_cut -use_sample shred -run_cog 


# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample SGA -img_source -use_sample sga
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -run_aper -no_cnn_cut -use_sample sga
# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample SGA -bkg_source -blend_remove_source -use_sample sga
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 100000 -run_parr -ncores 64 -overwrite -nchunks 20 -run_aper -no_cnn_cut -use_sample sga -run_cog 
