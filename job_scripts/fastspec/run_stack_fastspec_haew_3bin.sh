#!/bin/bash
# Run custom FastSpecFit (stackfit) on the M* x H-alpha-EW 3-bin stacks
# produced by code/nebular_stuff/stack_mstar_haew_3bin.py.
#
# "Custom" = Chabrier 9.9.9 templates + narrow He II lambda4686 via a custom
# emline-constraints YAML (see CONSTRAINTS setup below).
#
# Run interactively on a CPU node or wrap in an sbatch header as needed.

# --- module setup (match production custom-fastspec run) -------------------
source /dvs_ro/common/software/desi/desi_environment.sh main
module load fastspecfit/3.4.2

# Sanity check: stackfit must expose --constraintsfile (added in 3.4.2)
if ! stackfit --help 2>&1 | grep -q constraintsfile; then
    echo "ERROR: stackfit does not support --constraintsfile. Need fastspecfit/3.4.2+."
    exit 1
fi

# --- paths ------------------------------------------------------------------
STACK_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar_haew_3bin"
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
# 3. Validate before a long run:
#    python -c "
#    from fastspecfit.linetable import LineTable
#    from fastspecfit.emlines import EmlineConstraints
#    path = '${PSCRATCH}/fastspecfit/config/emline-constraints-dwarfs.yaml'
#    ec = EmlineConstraints(path, LineTable().table)
#    _, smax, _, _, _ = ec.line_bounds('heii_4686')
#    print(f'heii_4686 sigma_max = {smax} km/s')   # expect 750.0
#    "
#
# 4. After one stackfit output, spot-check science:
#    python -c "
#    import fitsio
#    d = fitsio.read('OUTFILE', ext='FASTSPEC')
#    print('HEII_4686_SIGMA =', d['HEII_4686_SIGMA'][0])  # should be <~750 km/s if detected
#    "
#    Also check primary header records your YAML:
#    python -c "
#    import fitsio
#    hdr = fitsio.read_header('OUTFILE')
#    print([k for k in hdr.keys() if 'CONSTRAINT' in k.upper()])
#    "
#
constraintsfile=${PSCRATCH}/fastspecfit/config/emline-constraints-dwarfs.yaml
if [[ ! -f "${constraintsfile}" ]]; then
    echo "ERROR: constraints file not found: ${constraintsfile}"
    echo "       Follow the CONSTRAINTS setup comments at the top of this script."
    exit 1
fi

NCORES=32

# Same external data dirs as the production fastspec job.
export DESI_SPECTRO_REDUX=/dvs_ro/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/dvs_ro/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/dvs_ro/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/dvs_ro/cfs/cdirs/desi/public/external/templates/fastspecfit

echo "Running custom FastSpecFit (stackfit) on M* x H-alpha-EW 3-bin stacks"
echo "Stack path    : ${STACK_PATH}"
echo "Templates     : ${templates}"
echo "Constraints   : ${constraintsfile}"
echo "Emlines       : bundled fiducial (default)"
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
        --constraintsfile="${constraintsfile}" \
        --nmonte=100 \
        --vdisp-nominal 100 --vdisp-bounds 50 200
    echo "  -> Saved to fastspec_${basename}"
    echo ""
done

echo "Done!"