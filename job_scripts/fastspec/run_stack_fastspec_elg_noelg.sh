#!/bin/bash
# Run custom FastSpecFit (stackfit) on the M* bin ELG / NO-ELG stacks.
#
# Config matches run_stack_fastspec_haew_5pct.sh and the production catalog run
# (run_custom_fastspec_job.sh): Chabrier 9.9.9 templates + the custom dwarfs
# emline-constraints YAML (narrow He II 4686) + custom emlines list, so all
# three fit contexts (per-object catalog, EW stacks, ELG/NO-ELG stacks) use
# identical fastspecfit code and constraints.
#
# NOTE: the ELG / NO-ELG stack *input files* themselves are still being
# finalized; the stack_mstar_{elg,noelg}_*.fits globs below are left as-is and
# will be updated in a later stage.

# ---- fastspecfit environment: DESI stack + HEAD editable checkout ----------
# Same HEAD checkout as the production run so stackfit runs the same 3.4.3-dev
# code as mpi-fastspecfit. The one-time editable install registers the HEAD
# `stackfit` entry point (and the correct output-header version). See the long
# comment in run_custom_fastspec_job.sh for the full rationale.
#
# One-time on NERSC, and again after each `git pull` of $FSF_SRC:
#   git clone https://github.com/desihub/fastspecfit ${FSF_SRC}   # stay on main
#   source /dvs_ro/common/software/desi/desi_environment.sh main
#   pip install --no-deps -e ${FSF_SRC}
FSF_SRC=/global/homes/v/virajvm/packages/fastspecfit

source /dvs_ro/common/software/desi/desi_environment.sh main
# Do NOT module load fastspecfit -- the editable HEAD checkout provides it.
export PYTHONPATH=${FSF_SRC}/py:$PYTHONPATH
export PATH=${FSF_SRC}/bin:$PATH

# Confirm stackfit runs the HEAD override and supports --constraintsfile.
fsf_file=$(python -c "import fastspecfit, os; print(os.path.dirname(fastspecfit.__file__))")
echo "fastspecfit imported from: ${fsf_file}"
case "${fsf_file}" in
    "${FSF_SRC}"/*) : ;;
    *) echo "ERROR: fastspecfit NOT imported from HEAD checkout ${FSF_SRC} (got ${fsf_file}). Aborting."
       exit 1 ;;
esac
if ! stackfit --help 2>&1 | grep -q -- '--constraintsfile'; then
    echo "ERROR: stackfit does not support --constraintsfile. Need fastspecfit 3.4.2+ / HEAD."
    exit 1
fi

# ---- paths ------------------------------------------------------------------
STACK_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar"
templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits
constraintsfile=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emline-constraints-dwarfs.yaml
emlinesfile=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines-dwarfs.ecsv
NCORES=32

for f in "${constraintsfile}" "${emlinesfile}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required input not found: ${f}"
        echo "       (constraints/emlines are shared with run_stack_fastspec_haew_5pct.sh)"
        exit 1
    fi
done

# Validate the custom dwarfs constraint + line list load and tie He II 4686 narrow.
python3 -c "
from astropy.table import Table
from fastspecfit.emlines import EmlineConstraints
lt = Table.read('${emlinesfile}', format='ascii.ecsv')
ec = EmlineConstraints('${constraintsfile}', lt)
print('OK -', len(lt), 'lines; heii_4686 sigma_max =', ec.line_bounds('heii_4686')[1], 'km/s')
"

# Same external data dirs as the production fastspec job.
export DESI_SPECTRO_REDUX=/dvs_ro/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/dvs_ro/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/dvs_ro/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/dvs_ro/cfs/cdirs/desi/public/external/templates/fastspecfit

echo "Running custom FastSpecFit (stackfit) on M* bin ELG / NO-ELG stacks"
echo "Stack path  : ${STACK_PATH}"
echo "Templates   : ${templates}"
echo "Constraints : ${constraintsfile}"
echo "Emlines     : ${emlinesfile}"

# --- ELG stacks ---
echo ""
echo "--- ELG stacks ---"
for f in ${STACK_PATH}/stack_mstar_elg_*.fits; do
    if [ -f "$f" ]; then
        basename=$(basename "$f")
        outfile="${STACK_PATH}/fastspec_${basename}"
        echo "Processing: ${basename}"
        stackfit "$f" -o "$outfile" \
            --mp ${NCORES} \
            --templates="${templates}" \
            --emlinesfile="${emlinesfile}" \
            --constraintsfile="${constraintsfile}" \
            --nmonte=100 \
            --vdisp-nominal 100 --vdisp-bounds 50 200
        echo "  -> Saved to fastspec_${basename}"
        echo ""
    fi
done

# --- NO-ELG stacks ---
echo ""
echo "--- NO-ELG stacks ---"
for f in ${STACK_PATH}/stack_mstar_noelg_*.fits; do
    if [ -f "$f" ]; then
        basename=$(basename "$f")
        outfile="${STACK_PATH}/fastspec_${basename}"
        echo "Processing: ${basename}"
        stackfit "$f" -o "$outfile" \
            --mp ${NCORES} \
            --templates="${templates}" \
            --emlinesfile="${emlinesfile}" \
            --constraintsfile="${constraintsfile}" \
            --nmonte=100 \
            --vdisp-nominal 100 --vdisp-bounds 50 200
        echo "  -> Saved to fastspec_${basename}"
        echo ""
    fi
done

echo "Done!"
