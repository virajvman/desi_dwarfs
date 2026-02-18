#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=get_imgs_general.log

N=1
mp=4

echo Starting at $(date)

cmd="sh cutouts_cnn_general.sh $N $mp"
echo $cmd
$cmd

if [ $? -eq 0 ]; then
    echo SUCCESS: done at $(date)
else
    echo FAILED: done at $(date)
    exit 1
fi
