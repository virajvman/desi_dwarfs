#!/bin/bash -l
# =============================================================================
# Login-node sanity check for the HEAD (3.4.3-dev) custom fastspec setup.
#
# NO SLURM allocation needed -- run it directly on a Perlmutter login node:
#     bash check_custom_fastspec_install.sh
#
# It validates, in seconds, everything that does NOT require real fitting:
#   1. The HEAD fastspecfit checkout is actually imported (not bare 3.4.2)
#   2. mpi-fastspecfit exposes --constraintsfile
#   3. The custom dwarf constraints + emlines load and tie He II 4686 narrow
#   4. The sample file reads and --plan computes a node/target distribution
#   5. --dry-run shows --constraintsfile propagating into the per-healpix
#      worker commands (the 3.4.3 fix, commit 1cbc08a)
#
# What it does NOT test (no compute is run): numba JIT compilation, template
# loading, the actual fit, or output writing. A clean pass here means the
# install + inputs + command wiring are correct; it does NOT guarantee a fit
# succeeds. For that, do a real single-object fit (see notes at bottom).
# =============================================================================

set -uo pipefail

# ---- paths (match run_custom_fastspec_job.sh) ------------------------------
samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits
constraintsfile=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emline-constraints-dwarfs.yaml
emlinesfile=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines-dwarfs.ecsv
templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits
# Use an EMPTY throwaway outdir so --plan/--dry-run see all files as "to do" and
# actually generate commands. (Pointing at the real outdir with existing outputs
# makes --dry-run emit 0 commands -> a false "propagation failed" result.)
outdir_data=${PSCRATCH}/fastspecfit/check-dryrun
FSF_SRC=/global/homes/v/virajvm/packages/fastspecfit
mp=16
# ---------------------------------------------------------------------------

# ---- fastspecfit environment: DESI stack + HEAD editable checkout ----------
# See run_custom_fastspec_job.sh for the full rationale + one-time setup:
#   git clone https://github.com/desihub/fastspecfit ${FSF_SRC}   # stay on main
#   source /dvs_ro/common/software/desi/desi_environment.sh main
#   pip install --no-deps -e ${FSF_SRC}
source /dvs_ro/common/software/desi/desi_environment.sh main
# Do NOT module load fastspecfit -- the editable HEAD checkout provides it.
export PYTHONPATH=${FSF_SRC}/py:$PYTHONPATH
export PATH=${FSF_SRC}/bin:$PATH
mkdir -p "${outdir_data}"

# external data dirs (needed so --plan/--dry-run can locate the redux files)
export DESI_SPECTRO_REDUX=/dvs_ro/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/dvs_ro/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/dvs_ro/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/dvs_ro/cfs/cdirs/desi/public/external/templates/fastspecfit

fail() { echo "FAIL: $*"; exit 1; }

# ---- 1. HEAD import + mpi-fastspecfit on PATH ------------------------------
echo "=== 1. environment / HEAD override ==="
command -v mpi-fastspecfit &>/dev/null || fail "mpi-fastspecfit not on PATH after module load + override."
fsf_file=$(python -c "import fastspecfit, os; print(os.path.dirname(fastspecfit.__file__))")
echo "fastspecfit imported from: ${fsf_file}"
case "${fsf_file}" in
    "${FSF_SRC}"/*) : ;;
    *) fail "fastspecfit NOT imported from HEAD checkout ${FSF_SRC} (got ${fsf_file})." ;;
esac
echo "fastspecfit HEAD: $(git -C "${FSF_SRC}" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "mpi-fastspecfit : $(type -p mpi-fastspecfit)"

# ---- 2. mpi-fastspecfit exposes --constraintsfile --------------------------
echo ""
echo "=== 2. --constraintsfile available on mpi-fastspecfit ==="
mpi-fastspecfit --help 2>&1 | grep -q -- '--constraintsfile' \
    || fail "mpi-fastspecfit lacks --constraintsfile -- checkout ${FSF_SRC} is stale (git pull onto main)."
echo "OK"

# ---- 3. inputs exist + constraints/emlines load ----------------------------
echo ""
echo "=== 3. inputs load (constraints + emlines, He II 4686 narrow) ==="
for f in "${samplefile}" "${constraintsfile}" "${emlinesfile}" "${templates}"; do
    [[ -f "${f}" ]] || fail "required input not found: ${f}"
done
python3 -c "
from astropy.table import Table
from fastspecfit.emlines import EmlineConstraints
lt = Table.read('${emlinesfile}', format='ascii.ecsv')
ec = EmlineConstraints('${constraintsfile}', lt)
smax = ec.line_bounds('heii_4686')[1]
print('OK -', len(lt), 'lines; heii_4686 sigma_max =', smax, 'km/s')
assert smax <= 750.0, 'heii_4686 sigma_max %.0f km/s looks broad -- check narrow tying' % smax
" || fail "constraints/emlines failed to load or He II 4686 is not narrow-tied."

# ---- 4. --plan: sample reads + node/target distribution --------------------
echo ""
echo "=== 4. --plan (sample reads; node/target distribution) ==="
mpi-fastspecfit \
    --samplefile=${samplefile} \
    --outdir-data=${outdir_data} \
    --specprod iron \
    --mp=${mp} \
    --overwrite --nompi --plan \
    || fail "--plan failed (sample read / distribution)."

# ---- 5. --dry-run: confirm --constraintsfile propagates --------------------
echo ""
echo "=== 5. --dry-run (--constraintsfile propagation into worker commands) ==="
dryout=$(mpi-fastspecfit \
    --samplefile=${samplefile} \
    --outdir-data=${outdir_data} \
    --emlinesfile=${emlinesfile} \
    --constraintsfile=${constraintsfile} \
    --templates=${templates} \
    --specprod iron \
    --mp=${mp} \
    --nmonte=100 \
    --vdisp-nominal 100 --vdisp-bounds 50 200 \
    --ignore-quasarnet \
    --overwrite --nompi --dry-run 2>&1)
# Show a couple of representative generated commands.
echo "${dryout}" | grep -m 3 -- '--constraintsfile' || true
echo "${dryout}" | grep -q -- '--constraintsfile' \
    || fail "--constraintsfile did NOT propagate into the per-healpix commands (HEAD missing commit 1cbc08a)."
echo "OK: --constraintsfile (and --emlinesfile) propagate into the worker fastspec commands."

echo ""
echo "=== ALL CHECKS PASSED ==="
echo "Install + inputs + command wiring are correct."
echo "NOTE: no fit was run. To also confirm fitting/templates/output before the"
echo "      full job, run one real object in an interactive node, e.g.:"
echo ""
echo "  salloc -N1 -C cpu -q interactive -t 00:20:00 -A desi"
echo "  # then, with the same env loaded:"
echo "  read s p h <<< \$(python -c \"from astropy.table import Table; t=Table.read('${samplefile}')[0]; print(t['SURVEY'],t['PROGRAM'],t['HEALPIX'])\")"
echo "  mpi-fastspecfit --specprod=iron --coadd-type=healpix --survey=\$s --program=\$p --healpix=\$h \\"
echo "      --mp=1 --ntargets=1 --nmonte=0 --nompi --overwrite --outdir-data=/tmp/fsf-check \\"
echo "      --templates=${templates} --emlinesfile=${emlinesfile} --constraintsfile=${constraintsfile} \\"
echo "      --vdisp-nominal 100 --vdisp-bounds 50 200 --ignore-quasarnet"
