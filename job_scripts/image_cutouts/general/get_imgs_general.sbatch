#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=get_imgs_general.log

# ============================================================
# GENERAL template for a single ad-hoc catalog -- edit before
# submitting. For the standard clean+shreds+sga download use
# get_imgs_all.sbatch instead. Never run two cutout-writing jobs
# against the store at the same time.
#
# Phase 1 writes per-brick HDF5 shards into OUTDIR (incremental
# and size-aware: objects already stored at >= the requested size,
# or tombstoned in permanently_failed.csv, are skipped).
# ============================================================
CATALOG_PATH="/pscratch/sd/v/virajvm/agent_ai_data/download_round_2_catalog.fits"
# Unified store (must match cutout_store.CUTOUTS_DIR)
OUTDIR="/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts/images"
RA_COL="RA"
DEC_COL="DEC"
ID_COL="TARGETID"
CUTOUT_SIZE=152

N=1       # number of nodes (must match #SBATCH --nodes)
mp=4      # multiprocessing workers per MPI rank

# Extra flags forwarded to many_cutouts_general.py
# e.g. "--size-col IMAGE_SIZE_PIX --no-invvar --retry-failed --layer ls-dr10"
EXTRA_ARGS=""

# Phase 2 (optional): consolidate shards into a single fixed-size HDF5
# for ML training. Set H5_OUTPUT to a file path to enable; empty = skip.
H5_OUTPUT=""
EXTRA_COLS=""       # e.g. "Z,MAG_G,MAG_R,MAG_Z,IS_DWARF"
H5_NWORKERS=8       # parallel shard-reading workers for consolidation
# ============================================================

# NOTE: sbatch copies this script to a spool dir, so BASH_SOURCE
# does NOT point at the repo -- the script dir must be hardcoded.
SCRIPT_DIR="/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/job_scripts/image_cutouts/general"

echo "Starting at $(date)"

# --- Phase 1: generate cutouts (inside Shifter container via srun) ---
cmd="sh $SCRIPT_DIR/cutouts_cnn_general.sh $N $mp $CATALOG_PATH $OUTDIR $RA_COL $DEC_COL $ID_COL $CUTOUT_SIZE $EXTRA_ARGS"
echo $cmd
$cmd

if [ $? -eq 0 ]; then
    echo "Phase 1 SUCCESS: cutouts done at $(date)"
else
    echo "Phase 1 FAILED at $(date)"
    exit 1
fi

# --- Phase 2 (optional): consolidate shards into one fixed-size HDF5 ---
if [[ -n "$H5_OUTPUT" ]]; then
    echo "Starting Phase 2: HDF5 consolidation at $(date)"

    # Phase 2 runs OUTSIDE the container -- get h5py/astropy from the DESI env
    source /global/cfs/cdirs/desi/software/desi_environment.sh main
    export HDF5_USE_FILE_LOCKING=FALSE

    h5_args="--catalog-path $CATALOG_PATH --cutouts-dir $OUTDIR --output-h5 $H5_OUTPUT"
    h5_args="$h5_args --ra-col $RA_COL --dec-col $DEC_COL --id-col $ID_COL"
    h5_args="$h5_args --cutout-size $CUTOUT_SIZE --nworkers $H5_NWORKERS"
    h5_args="$h5_args --skip-missing"

    if [[ -n "$EXTRA_COLS" ]]; then
        h5_args="$h5_args --extra-cols $EXTRA_COLS"
    fi

    h5_cmd="python3 $SCRIPT_DIR/consolidate_to_h5.py $h5_args"
    echo $h5_cmd
    $h5_cmd

    if [ $? -eq 0 ]; then
        echo "Phase 2 SUCCESS: HDF5 written at $(date)"
    else
        echo "Phase 2 FAILED at $(date)"
        exit 1
    fi
fi

echo "All done at $(date)"
