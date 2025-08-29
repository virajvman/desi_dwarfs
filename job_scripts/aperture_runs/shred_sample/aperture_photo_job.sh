#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=06:00:00
#SBATCH --output=aperture_shred_job.log

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample LOWZ -min 0 -max 100000 -run_parr -ncores 128 -overwrite -nchunks 2 -run_aper -run_cog 
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 100000 -run_parr -ncores 128 -overwrite -nchunks 20 -run_aper -run_cog
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 100000 -run_parr -ncores 128 -overwrite -nchunks 5 -run_aper -run_cog
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 100000 -run_parr -ncores 128 -overwrite -nchunks 20 -run_aper -run_cog



# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 6000 -run_parr -ncores 64 -overwrite -nchunks 2 -use_sample shred -make_main_cats

python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG,LOWZ,BGS_BRIGHT,BGS_FAINT -min 0 -max 100000 -run_parr -ncores 1 -overwrite -nchunks 1 -tgids 39627785513211263 -make_cats

python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample SGA -min 0 -max 100000 -run_parr -ncores 32 -overwrite -nchunks 4 -make_cats -run_aper -use_sample sga