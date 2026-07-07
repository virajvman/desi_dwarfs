#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=25GB
#SBATCH --time=05:00:00
#SBATCH --output=run_rrdesi.log

source /global/common/software/desi/desi_environment.sh
cd /global/u1/v/virajvm/DESI2_LOWZ/quenched_fracs_nbs
python3 mock_rrdesi.py
