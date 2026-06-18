#!/bin/bash
# Run custom FastSpecFit (stackfit) on the M* bin ELG / NO-ELG stacks, for the
# CONTINUUM (r-band) normalized product. Identical to
# run_stack_fastspec_elg_noelg.sh except STACK_PATH points at the contnorm
# directory; the per-bin FITS filenames are the same stem
# (stack_mstar_{elg,noelg}_{mlo}_{mhi}.fits), so the two products coexist in
# separate directories.
#
# Config matches run_stack_fastspec_haew_5pct.sh and the production catalog run
# (run_custom_fastspec_job.sh): Chabrier 9.9.9 templates + the custom dwarfs
# emline-constraints YAML (narrow He II 4686) + custom emlines list, so all
# fit contexts use the identical fastspecfit version and constraints.
#
# Inputs are produced by
#   code/stacking_analysis/stack_mstar_elg_vs_noelg.py --norm continuum
# (1 mean row + 200 bootstrap rows; Scholte propagated-ivar error model, same
# as the haew_5pct product). Stale fastspec_* outputs are removed below before
# refitting so a dropped/renamed bin never lingers.

# ---- fastspecfit environment ------------------------------------------------
# Same fastspecfit/3.4.3 module as the production run, so stackfit runs the same
# version as mpi-fastspecfit (incl. the narrow final-pass free_sigma:false +
# doublet-locking line-fit changes). 3.4.3 gives stackfit --constraintsfile.
export FASTSPECFIT_VERSION=3.4.3
source /dvs_ro/common/software/desi/desi_environment.sh main
module load fastspecfit/${FASTSPECFIT_VERSION}

# Confirm stackfit is available and supports --constraintsfile.
if ! command -v stackfit &>/dev/null; then
    echo "ERROR: stackfit not on PATH after module load. Aborting."
    exit 1
fi
if ! stackfit --help 2>&1 | grep -q -- '--constraintsfile'; then
    echo "ERROR: stackfit does not support --constraintsfile. Need fastspecfit/3.4.2+."
    exit 1
fi

# ---- paths ------------------------------------------------------------------
STACK_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_contnorm"
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

echo "Running custom FastSpecFit (stackfit) on M* bin ELG / NO-ELG stacks (continuum-normalized)"
echo "Stack path  : ${STACK_PATH}"
echo "Templates   : ${templates}"
echo "Constraints : ${constraintsfile}"
echo "Emlines     : ${emlinesfile}"

# Remove previous fastspec_* outputs so a dropped/renamed bin never lingers.
shopt -s nullglob
n_old=0
for f in "${STACK_PATH}"/fastspec_stack_mstar_elg_*.fits \
         "${STACK_PATH}"/fastspec_stack_mstar_noelg_*.fits; do
    rm -f "$f"
    n_old=$((n_old + 1))
done
echo "Removed ${n_old} previous fastspec_stack_mstar_{elg,noelg}_*.fits files"

# --- ELG stacks ---
echo ""
echo "--- ELG stacks ---"
for f in ${STACK_PATH}/stack_mstar_elg_*.fits; do
    if [ -f "$f" ]; then
        basename=$(basename "$f")
        case "${basename}" in
            fastspec_*) continue ;;
        esac
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
        case "${basename}" in
            fastspec_*) continue ;;
        esac
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
