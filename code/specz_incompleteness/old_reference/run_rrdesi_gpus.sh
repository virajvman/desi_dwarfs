#!/bin/bash -l

#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=run_rrdesi_gpus.log

source /global/common/software/desi/desi_environment.sh
srun -N 2 -n 8 -c 2 --gpu-bind=map_gpu:3,2,1,0 wrap_rrdesi -i /pscratch/sd/v/virajvm/list_coadds_0.ascii -o /pscratch/sd/v/virajvm/rrmock_reobs/  --gpu --rrdetails > redrock.log 2>&1