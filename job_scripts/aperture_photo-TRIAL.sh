cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 100000 -run_parr -ncores 1 -overwrite -nchunks 1 -run_aper -no_cnn_cut -use_sample shred -run_cog -tgids 39633440412272553