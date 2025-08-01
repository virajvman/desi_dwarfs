#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=clean_cutouts.log

N=1
mp=1

echo Starting at $(date)

cmd="sh cutouts_cnn_clean.sh $N $mp"
echo $cmd
$cmd

if [ $? -eq 0 ]; then
    echo SUCCESS: done at $(date)
else
    echo FAILED: done at $(date)
    exit 1
fi
