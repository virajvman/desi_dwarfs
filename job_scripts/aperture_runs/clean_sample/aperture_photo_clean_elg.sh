#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --output=aperture_clean_job_elg.log

set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Avoid oversubscription in multi-threaded libraries
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
# shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample ELG -img_source -use_sample clean

python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 50000 -run_parr -ncores 64 -overwrite -nchunks 50 -run_aper -use_sample clean -no_cnn_cut

shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4

shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample ELG -img_source -use_sample clean

shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample ELG -bkg_source -blend_remove_source -parent_galaxy -use_sample clean

python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample ELG -min 0 -max 50000 -run_parr -ncores 64 -overwrite -nchunks 50 -run_cog -use_sample clean -no_cnn_cut

