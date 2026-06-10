#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=get_imgs_clean.log

# ============================================================
# CLEAN sample cutouts -> per-brick HDF5 shards.
# Fixed 152 px cutouts; reruns are incremental (existing shard
# groups and tombstoned objects are skipped automatically), so
# you can feed this any catalog of clean-sample objects.
# ============================================================
CATALOG_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v5.fits"
OUTDIR="$PSCRATCH/dwarf_cutouts/clean"
RA_COL="RA"
DEC_COL="DEC"
ID_COL="TARGETID"
CUTOUT_SIZE=152

N=1       # number of nodes (must match #SBATCH --nodes)
mp=4      # multiprocessing workers per MPI rank

# Extra flags forwarded to many_cutouts_general.py,
# e.g. "--retry-failed --timeout 180 --layer ls-dr10"
EXTRA_ARGS=""
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
