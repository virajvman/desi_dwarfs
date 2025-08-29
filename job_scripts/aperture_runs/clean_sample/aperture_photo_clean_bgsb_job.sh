#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --output=aperture_phot_clean_bgsb.log

./aperture_photo_clean_bgsb_script.sh