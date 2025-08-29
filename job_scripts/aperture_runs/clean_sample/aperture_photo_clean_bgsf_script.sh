cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_FAINT -img_source -use_sample clean
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 20000 -run_parr -ncores 64 -overwrite -nchunks 10 -run_aper -use_sample clean -no_cnn_cut
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_FAINT -bkg_source -blend_remove_source -use_sample clean
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 20000 -run_parr -ncores 64 -overwrite -nchunks 10 -run_aper -run_cog -use_sample clean -no_cnn_cut