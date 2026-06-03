#!/bin/bash
# Run custom FastSpecFit (stackfit) on the M* x H-alpha-EW 3-bin stacks
# produced by code/nebular_stuff/stack_mstar_haew_3bin.py.
#
# "Custom" = pass the same templates + emlines file used in the custom
# fastspec production run (job_scripts/fastspec/run_custom_fastspec_job.sh),
# plus Monte-Carlo error sampling. Run interactively on a CPU node or wrap
# in an sbatch header as needed.

# Match the production custom-fastspec run's module setup exactly
# (see job_scripts/fastspec/combine_custom_fastspec_cat.sh).
source /dvs_ro/common/software/desi/desi_environment.sh main-2.2.0
module swap desitarget/4.7.2
module load fastspecfit/3.4.1

# --- paths ------------------------------------------------------------------
STACK_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_haew_3bin"
templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits
emlinesfile=/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines.ecsv
NCORES=32

# Custom run needs the same external data dirs as the production fastspec job.
export DESI_SPECTRO_REDUX=/dvs_ro/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/dvs_ro/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/dvs_ro/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/dvs_ro/cfs/cdirs/desi/public/external/templates/fastspecfit

echo "Running custom FastSpecFit (stackfit) on M* x H-alpha-EW 3-bin stacks"
echo "Stack path : ${STACK_PATH}"
echo "Templates  : ${templates}"
echo "Emlines    : ${emlinesfile}"
echo ""

shopt -s nullglob
stack_files=("${STACK_PATH}"/stack_ALL_mstar_*.fits)
if [ ${#stack_files[@]} -eq 0 ]; then
    echo "ERROR: no stack_ALL_mstar_*.fits files found in ${STACK_PATH}"
    exit 1
fi

for f in "${stack_files[@]}"; do
    basename=$(basename "$f")
    # Don't re-fit already-produced output files.
    case "${basename}" in
        fastspec_*) continue ;;
    esac
    outfile="${STACK_PATH}/fastspec_${basename}"
    echo "Processing: ${basename}"
    stackfit "$f" -o "$outfile" \
        --mp ${NCORES} \
        --templates="${templates}" \
        --emlinesfile="${emlinesfile}" \
        --nmonte=100 \
        --vdisp-nominal 100 --vdisp-bounds 50 200
    echo "  -> Saved to fastspec_${basename}"
    echo ""
done

echo "Done!"
