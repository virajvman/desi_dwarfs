#!/bin/bash -l

# Shell wrapper for many_cutouts_general.py
# Handles MPI task layout and CPU affinity on NERSC Perlmutter.
#
# Usage (called by get_imgs_{general,clean,shreds,sga}.sbatch):
#   sh cutouts_cnn_general.sh <N> <mp> <catalog_path> <outdir> \
#       [ra_col] [dec_col] [id_col] [cutout_size] [extra_args...]

mpiscript="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/many_cutouts_general.py"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
# Lustre + h5py: ranks write disjoint shards; readers open shards read-only
export HDF5_USE_FILE_LOCKING=FALSE

N=${1:-1}                  # nodes
mp=${2:-1}                 # multiprocessing workers per MPI rank
catalog_path=${3:?"ERROR: catalog_path (arg 3) is required"}
outdir_data=${4:?"ERROR: outdir_data (arg 4) is required"}
ra_col=${5:-RA}
dec_col=${6:-DEC}
id_col=${7:-TARGETID}
cutout_size=${8:-152}
shift 8 2>/dev/null
extra_args="$@"

mkdir -p "$outdir_data"

args="--catalog-path $catalog_path --outdir-data $outdir_data"
args="$args --ra-col $ra_col --dec-col $dec_col --id-col $id_col"
args="$args --cutout-size $cutout_size"

if [[ $mp != " " ]] && [[ $mp != "" ]] && [[ $mp != "-" ]]; then
    args="$args --mp $mp"
fi

if [[ -n "$extra_args" ]]; then
    args="$args $extra_args"
fi

# Compute number of MPI tasks (128 cores per Perlmutter CPU node)
ntasks=$((128 * $N / $mp))

# CPU affinity: when using multiprocessing, let the OS schedule freely;
# otherwise pin to cores.
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
