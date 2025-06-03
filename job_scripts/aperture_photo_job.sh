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

