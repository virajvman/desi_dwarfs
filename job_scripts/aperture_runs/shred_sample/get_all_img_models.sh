#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=get_all_imgs.log

set -e

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main

# Avoid oversubscription in multi-threaded libraries
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4

shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_BRIGHT -img_source -use_sample shred
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_FAINT -img_source -use_sample shred
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample ELG -img_source -use_sample shred
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample LOWZ -img_source -use_sample shred






