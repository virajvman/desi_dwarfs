#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=aper_shred_job_bgsb.log

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT -min 0 -max 100000 -run_parr -ncores 32 -overwrite -nchunks 10 -run_aper -no_cnn_cut -use_sample shred

