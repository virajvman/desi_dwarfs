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

# ============================================================
# USER CONFIG -- edit these before submitting with sbatch
# ============================================================
CATALOG_PATH="/pscratch/sd/v/virajvm/agent_ai_data/download_round_2_catalog.fits"
OUTDIR="$PSCRATCH/agent_ai_data/images"
RA_COL="RA"
DEC_COL="DEC"
ID_COL="TARGETID"
CUTOUT_SIZE=152

N=1       # number of nodes
mp=4      # multiprocessing workers per MPI rank

# Optional: any extra flags forwarded directly to many_cutouts_general.py
# e.g. "--no-invvar --timeout 180 --max-retries 3 --layer ls-dr10"
EXTRA_ARGS=""

# Phase 2 (optional): consolidate individual FITS into a single HDF5 file.
# Set H5_OUTPUT to a file path to enable; leave empty to skip.
H5_OUTPUT=""
EXTRA_COLS=""       # e.g. "Z,MAG_G,MAG_R,MAG_Z,IS_DWARF"
H5_NWORKERS=8      # parallel FITS-reading workers for consolidation
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo Starting at $(date)

# --- Phase 1: generate cutouts (runs inside Shifter container via srun) ---
cmd="sh cutouts_cnn_general.sh $N $mp $CATALOG_PATH $OUTDIR $RA_COL $DEC_COL $ID_COL $CUTOUT_SIZE $EXTRA_ARGS"
echo $cmd
$cmd

if [ $? -eq 0 ]; then
    echo "Phase 1 SUCCESS: cutouts done at $(date)"
else
    echo "Phase 1 FAILED at $(date)"
    exit 1
fi

# --- Phase 2 (optional): consolidate FITS into HDF5 ---
if [[ -n "$H5_OUTPUT" ]]; then
    echo "Starting Phase 2: HDF5 consolidation at $(date)"

    h5_args="--catalog-path $CATALOG_PATH --image-dir $OUTDIR --output-h5 $H5_OUTPUT"
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
