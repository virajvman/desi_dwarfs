#!/bin/bash
#SBATCH --account=desi_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=specz_redrock
#SBATCH --output=specz_redrock_%j.log
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=FAIL

# Run GPU redrock on a list of mock coadd files.
# Usage:  sbatch run_redrock.sbatch <input_list.ascii>
#
# REDROCK PINNING (load-bearing): the mocks must be processed with the SAME
# redrock version + galaxy templates as the iron/DR1 production, otherwise we
# model the failure modes of a different pipeline than the one that produced
# the catalog. 23.1 is the iron-era software release.
# TODO(viraj): confirm the exact iron production tag before the full run.

set -eo pipefail

INPUT_LIST="${1:?usage: sbatch run_redrock.sbatch <input_list.ascii>}"
OUTPUT_DIR="/pscratch/sd/v/virajvm/specz_incompleteness/redrock"

source /global/common/software/desi/desi_environment.sh 23.1

set -u

mkdir -p "${OUTPUT_DIR}"

srun -N 2 -n 8 -c 2 --gpu-bind=map_gpu:3,2,1,0 \
    wrap_rrdesi -i "${INPUT_LIST}" -o "${OUTPUT_DIR}" --gpu --rrdetails
