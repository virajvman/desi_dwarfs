cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample LOWZ -min 0 -max 500 -run_parr -ncores 128 -overwrite -nchunks 2 -run_aper -run_cog -use_clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 12000 -run_parr -ncores 128 -overwrite -nchunks 20 -run_aper -run_cog -use_clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 2400 -run_parr -ncores 128 -overwrite -nchunks 5 -run_aper -run_cog -use_clean -no_cnn_cut
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 24000 -run_parr -ncores 128 -overwrite -nchunks 20 -run_aper -run_cog -use_clean -no_cnn_cut

# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample LOWZ -min 0 -max 500 -run_parr -ncores 128 -overwrite -nchunks 2 -run_cog -use_clean -no_cnn_cut
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 12000 -run_parr -ncores 128 -overwrite -nchunks 20 -run_cog -use_clean -no_cnn_cut
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 2400 -run_parr -ncores 128 -overwrite -nchunks 5 -run_cog -use_clean -no_cnn_cut
# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 24000 -run_parr -ncores 128 -overwrite -nchunks 20 -run_cog -use_clean -no_cnn_cut

