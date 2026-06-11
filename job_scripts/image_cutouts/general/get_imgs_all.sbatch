#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --image=dstndstn/cutouts:dvsro3
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=get_imgs_all.log

# ============================================================
# UNIFIED cutout download: all sample classes into ONE per-brick
# HDF5 shard store. Phases run SEQUENTIALLY in one job -- never
# run two cutout-writing jobs against the store at the same time
# (each shard must have a single writer).
#
# The store is incremental and size-aware (bigger-size-wins):
# objects already stored at >= the requested size are skipped,
# so later phases automatically skip overlap with earlier ones.
# ============================================================

# NOTE: sbatch copies this script to a spool dir, so BASH_SOURCE
# does NOT point at the repo -- the script dir must be hardcoded.
SCRIPT_DIR="/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/job_scripts/image_cutouts/general"

# Unified store (must match cutout_store.CUTOUTS_DIR)
OUTDIR="/global/cfs/cdirs/desi/users/virajvm/dwarf_cutouts/images"

RA_COL="RA"
DEC_COL="DEC"
ID_COL="TARGETID"
CUTOUT_SIZE=152   # fallback only; --size-col takes precedence

N=1       # number of nodes (must match #SBATCH --nodes)
mp=4      # multiprocessing workers per MPI rank

# Per-phase enable flags + catalogs

RUN_SGA=false
SGA_CATALOG="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_desi_SGA_matched_dwarfs_REPROCESS_V2.fits"
SGA_ARGS="--size-col IMAGE_SIZE_PIX"

RUN_SHREDS=false
SHREDS_CATALOG="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_shreds_catalog_v5.fits"
SHREDS_ARGS="--size-col IMAGE_SIZE_PIX"

RUN_CLEAN=false
CLEAN_CATALOG="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v5_RUN_W_APER.fits"
CLEAN_ARGS="--size-col IMAGE_SIZE_PIX"

RUN_CLEAN_ALL=true
CLEAN_ALL_CATALOG="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_y1_dwarf_clean_catalog_v5.fits"
CLEAN_ALL_ARGS=""

# Extra flags applied to every phase, e.g. "--retry-failed --timeout 180"
EXTRA_ARGS=""
# ============================================================

echo "Starting at $(date)"

run_phase () {
    local name=$1 catalog=$2 phase_args=$3
    echo "=== Phase: $name ($(date)) ==="
    cmd="sh $SCRIPT_DIR/cutouts_cnn_general.sh $N $mp $catalog $OUTDIR $RA_COL $DEC_COL $ID_COL $CUTOUT_SIZE $phase_args $EXTRA_ARGS"
    echo $cmd
    $cmd
    if [ $? -ne 0 ]; then
        echo "Phase $name FAILED at $(date)"
        exit 1
    fi
    echo "Phase $name SUCCESS at $(date)"
}

if [ "$RUN_SGA" = true ];    then run_phase sga    "$SGA_CATALOG"    "$SGA_ARGS";    fi
if [ "$RUN_SHREDS" = true ]; then run_phase shreds "$SHREDS_CATALOG" "$SHREDS_ARGS"; fi
if [ "$RUN_CLEAN" = true ];  then run_phase clean  "$CLEAN_CATALOG"  "$CLEAN_ARGS";  fi
if [ "$RUN_CLEAN_ALL" = true ];  then run_phase clean_all  "$CLEAN_ALL_CATALOG"  "$CLEAN_ALL_ARGS";  fi



echo "All done at $(date)"
