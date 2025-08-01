#!/bin/bash -l

#salloc -N 4 -C cpu -A desi -t 04:00:00 --qos interactive --image=dstndstn/cutouts:dvsro
#sh cutouts-fuji.sh 4 16 fuji healpix > cutouts-fuji-JOBID.log 2>&1 &

mpiscript=$HOME/DESI2_LOWZ/desi_dwarfs/job_scripts/image_cutouts/many_cutouts_clean.py

outdir_data=$PSCRATCH/redo_photometry_plots/all_good_cutouts

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

N=${1:-1}                  # nodes
mp=${2:-1}                 # number of multiprocessing cores


args="--outdir-data $outdir_data"

if [[ $mp != " " ]] && [[ $mp != "" ]] && [[ $mp != "-" ]]; then
    args=$args" --mp $mp"
fi

# compute the number of MPI tasks
ntasks=$((128 * $N / $mp)) # =512 if N=64; =32 if N=4

# Need to set the affinity correctly if using multiprocessing vs pure MPI.
# https://docs.nersc.gov/jobs/examples
# https://docs.nersc.gov/jobs/affinity
if [[ $mp > 1 ]]; then
    cpus_per_task=$(($mp * 2))
    cpu_bind="none"
else
    cpus_per_task=$((2 * 128 * $N / $ntasks))
    cpu_bind="cores"
fi

cmd="time srun --network=no_vni --nodes=$N --ntasks=$ntasks --cpus-per-task=$cpus_per_task --cpu-bind=$cpu_bind shifter $mpiscript $args"
echo $cmd
$cmd
