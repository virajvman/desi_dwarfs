#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=get_imgs_sga.log

# ============================================================
# SGA-matched sample cutouts -> per-brick HDF5 shards.
# Per-object cutout sizes from IMAGE_SIZE_PIX. Unlike the old
# many_cutouts_sga.py, invvar is now INCLUDED (2026-06 decision).
# Reruns are incremental.
# ============================================================
CATALOG_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_desi_SGA_matched_dwarfs_REPROCESS_V2.fits"
OUTDIR="/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts/sga"
RA_COL="RA"
DEC_COL="DEC"
ID_COL="TARGETID"
CUTOUT_SIZE=152   # fallback only; --size-col takes precedence

N=1       # number of nodes (must match #SBATCH --nodes)
mp=4      # multiprocessing workers per MPI rank

EXTRA_ARGS="--size-col IMAGE_SIZE_PIX"
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Starting at $(date)"

cmd="sh $SCRIPT_DIR/cutouts_cnn_general.sh $N $mp $CATALOG_PATH $OUTDIR $RA_COL $DEC_COL $ID_COL $CUTOUT_SIZE $EXTRA_ARGS"
echo $cmd
$cmd

if [ $? -eq 0 ]; then
    echo "SUCCESS: cutouts done at $(date)"
else
    echo "FAILED at $(date)"
    exit 1
fi
