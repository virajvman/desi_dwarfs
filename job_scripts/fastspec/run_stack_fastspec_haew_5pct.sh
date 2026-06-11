#!/bin/bash
# Run custom FastSpecFit (stackfit) on stacked dwarf spectra from
# code/nebular_stuff/stack_mstar_haew_5pct.py:
#
#   1. EW-binned stacks in STACK_PATH/  (M* x H-alpha EW, bootstrap rows)
#   2. Mass-only stacks in STACK_PATH/mstar_only/  (M* only, single row)
#
# Mass bins: 0.5 dex from log M*=6 to 8; 0.25 dex from 8 to 9.25.
# EW bins (<30, 30-100, >100 Angstrom) apply to product 1 only (N>=50).
#
# "Custom" = Chabrier 9.9.9 templates + narrow He II lambda4686 via a custom
# emline-constraints YAML (see CONSTRAINTS setup below).
#
# Run interactively on a CPU node or wrap in an sbatch header as needed.

# --- module setup (match production custom-fastspec run) -------------------
# 3.4.3 matches run_custom_fastspec_job.sh so the stacks are fit with the same
# version as the per-object catalog (incl. the narrow final-pass free_sigma:false
# + doublet-locking line-fit changes).
export FASTSPECFIT_VERSION=3.4.3
source /dvs_ro/common/software/desi/desi_environment.sh main
module load fastspecfit/${FASTSPECFIT_VERSION}

# Sanity check: stackfit must expose --constraintsfile (added in 3.4.2)
if ! stackfit --help 2>&1 | grep -q constraintsfile; then
    echo "ERROR: stackfit does not support --constraintsfile. Need fastspecfit/3.4.2+."
    exit 1
fi

# --- paths ------------------------------------------------------------------
STACK_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_haew_5pct"
MSTAR_ONLY_PATH="${STACK_PATH}/mstar_only"
templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits

# Fiducial line list is bundled with fastspecfit/3.4.2 — no custom emlines.ecsv needed.
# (Your old data_metal/emlines.ecsv matched fiducial; drop --emlinesfile.)
#
# CONSTRAINTS setup — narrow He II lambda4686
# -------------------------------------------
# 1. Copy bundled YAML once (do NOT edit the module install in place):
#
#    CONSTRAINTS_SRC=$(python -c "from importlib.resources import files; \
#        print(files('fastspecfit') / 'data/emline-constraints.yaml')")
#    mkdir -p ${PSCRATCH}/fastspecfit/config
#    cp "${CONSTRAINTS_SRC}" ${PSCRATCH}/fastspecfit/config/emline-constraints-dwarfs.yaml
#
# 2. Edit emline-constraints-dwarfs.yaml:
#    a) In global.free_lines: REMOVE the line "heii_4686"
#       (leave heii_1640 — out of range at z<0.5 for dwarf stacks)
#    b) In profiles.narrow_only.kinematic_groups[narrow_balmer].members: ADD heii_4686
#    c) In profiles.narrow_broad.kinematic_groups[narrow_all].members:   ADD heii_4686
#    Effect: He II 4686 tied to narrow Halpha kinematics, sigma max 750 km/s
#            (instead of broad QSO defaults: sigma max 10000 km/s).
#

constraintsfile=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emline-constraints-dwarfs.yaml
emlinesfile=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines-dwarfs.ecsv

if [[ ! -f "${constraintsfile}" ]]; then
    echo "ERROR: constraints file not found: ${constraintsfile}"
    echo "       Follow the CONSTRAINTS setup comments at the top of this script."
    exit 1
fi

if [[ ! -f "${emlinesfile}" ]]; then
    echo "ERROR: emlines file not found: ${emlinesfile}"
    exit 1
fi

python3 -c "
from astropy.table import Table
from fastspecfit.emlines import EmlineConstraints
lt = Table.read('${emlinesfile}', format='ascii.ecsv')
ec = EmlineConstraints('${constraintsfile}', lt)
print('OK —', len(lt), 'lines; heii_4686 sigma_max =', ec.line_bounds('heii_4686')[1], 'km/s')
"

NCORES=32

# Same external data dirs as the production fastspec job.
export DESI_SPECTRO_REDUX=/dvs_ro/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/dvs_ro/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/dvs_ro/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/dvs_ro/cfs/cdirs/desi/public/external/templates/fastspecfit

echo "Running custom FastSpecFit (stackfit) on stacked dwarf spectra"
echo "EW-binned path : ${STACK_PATH}"
echo "Mass-only path : ${MSTAR_ONLY_PATH}"
echo "Templates      : ${templates}"
echo "Constraints    : ${constraintsfile}"
echo "Emlines        : ${emlinesfile}"
echo ""

run_stackfit_in_dir() {
    local dir="$1"
    local label="$2"

    echo "========== ${label}: ${dir} =========="

    if [[ ! -d "${dir}" ]]; then
        echo "WARNING: directory not found: ${dir}; skipping."
        echo ""
        return 0
    fi

    shopt -s nullglob
    local n_old=0
    for f in "${dir}"/fastspec_stack_ALL_mstar_*.fits; do
        rm -f "$f"
        n_old=$((n_old + 1))
    done
    echo "Removed ${n_old} previous fastspec_stack_ALL_mstar_*.fits files"

    local stack_files=("${dir}"/stack_ALL_mstar_*.fits)
    if [ ${#stack_files[@]} -eq 0 ]; then
        echo "WARNING: no stack_ALL_mstar_*.fits files found in ${dir}; skipping."
        echo ""
        return 0
    fi

    local n_fit=0
    for f in "${stack_files[@]}"; do
        local basename
        basename=$(basename "$f")
        case "${basename}" in
            fastspec_*) continue ;;
        esac
        local outfile="${dir}/fastspec_${basename}"
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
        n_fit=$((n_fit + 1))
    done

    echo "${label}: fitted ${n_fit} stack file(s) in ${dir}"
    echo ""
}

run_stackfit_in_dir "${STACK_PATH}" "EW-binned"
run_stackfit_in_dir "${MSTAR_ONLY_PATH}" "Mass-only"

echo "Done!"
