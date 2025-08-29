cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT,BGS_FAINT,LOWZ,ELG -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 2 -use_sample clean -no_cnn_cut -make_cats
shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample clean -img_source
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 10 -run_aper -use_sample clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 10 -run_aper -use_sample clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample LOWZ -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 10 -run_aper -use_sample clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 10 -run_aper -use_sample clean -no_cnn_cut
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample clean -bkg_source -blend_remove_source
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 10 -run_aper -run_cog -use_sample clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 10 -run_aper -run_cog -use_sample clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample LOWZ -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 10 -run_aper  -run_cog -use_sample clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 10 -run_aper  -run_cog -use_sample clean -no_cnn_cut
