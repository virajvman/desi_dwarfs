#!/bin/bash -l
#SBATCH --job-name=fastspec_incr_8k
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/fastspec_incr_%j.out
#SBATCH --error=logs/fastspec_incr_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=virajvm@stanford.edu
#
# Incremental FastSpecFit: fit only TARGETIDs missing from the merged catalog,
# then vstack new rows onto fastspec-iron-dr1-dwarfs.fits.
#
# The fit writes to a SEPARATE scratch tree (incremental_outdir), so the canonical
# 450k tree is never modified and only the missing TARGETIDs are fit (no neighbor
# re-fitting, no deletes).
#
# Operational checklist:
#   1. Dry-run (prep only, inspect the manifest):
#        DRY_RUN=1 sbatch job_scripts/fastspec/run_incremental_fastspec_job.sh
#      Then read the manifest:
#        cat .../desi_dr1_dwarfs_fastspec_incremental.fits.manifest.json
#      Sanity-check n_missing_targetids (~8k) and n_healpix_new_region (informational).
#   2. Full run:
#        sbatch job_scripts/fastspec/run_incremental_fastspec_job.sh
#   3. Verify coverage (e.g. consolidate_photometry report_fastspec_coverage).
#   4. If satisfied, replace the canonical merged catalog:
#        mv fastspec-iron-dr1-dwarfs-v2.fits fastspec-iron-dr1-dwarfs.fits
#      or re-run combine with --replace-original.
#
# Combine-only after a crashed fit (fit finished, combine did not):
#        sbatch job_scripts/fastspec/combine_incremental_fastspec_cat.sh

# ---- knobs -----------------------------------------------------------------
# Set DRY_RUN=1 to only run prepare_incremental_fastspec_sample.py (no fit/combine).
DRY_RUN=${DRY_RUN:-0}

N=2
mp=16

dwarf_catalog=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits
dwarf_hdu=MAIN
fastspec_merged=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/iron/catalogs/fastspec-iron-dr1-dwarfs.fits
fastspec_hdu=3
out_sample=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs_fastspec_incremental.fits
manifest=${out_sample}.manifest.json
out_merged=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/iron/catalogs/fastspec-iron-dr1-dwarfs-v2.fits

constraintsfile=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emline-constraints-dwarfs.yaml
emlinesfile=/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines-dwarfs.ecsv
templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits
# Canonical 450k tree -- READ ONLY here (prep uses it only to flag new-region healpix).
outdir_data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/
# Scratch tree the incremental fit writes to (and combine reads from). Starts empty,
# so no healpix is ever skipped and the canonical tree is never modified.
incremental_outdir=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_incremental_run/
# ---------------------------------------------------------------------------

# NOTE: sbatch copies this script to a spool dir; BASH_SOURCE does NOT point at the repo.
REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code"

export FASTSPECFIT_VERSION=3.4.3
source /dvs_ro/common/software/desi/desi_environment.sh main
module load fastspecfit/${FASTSPECFIT_VERSION}

if ! command -v mpi-fastspecfit &>/dev/null; then
    echo "ERROR: mpi-fastspecfit not on PATH after module load. Aborting."
    module avail fastspecfit 2>&1
    exit 1
fi

echo "=== Loaded modules ==="
module list 2>&1 | grep -E 'fastspecfit|desiutil|desispec|desitarget|speclite|specsim'
echo "======================"

export DESI_SPECTRO_REDUX=/dvs_ro/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/dvs_ro/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/dvs_ro/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/dvs_ro/cfs/cdirs/desi/public/external/templates/fastspecfit
export NUMBA_CACHE_DIR=${PSCRATCH}/fastspecfit/numba-cache/${FASTSPECFIT_VERSION}
mkdir -p "${NUMBA_CACHE_DIR}" logs

# ---- Step 1: prep incremental sample --------------------------------------
PREP_ARGS=(
    --dwarf-catalog="${dwarf_catalog}"
    --dwarf-hdu="${dwarf_hdu}"
    --fastspec-merged="${fastspec_merged}"
    --fastspec-hdu="${fastspec_hdu}"
    --outdir-data="${outdir_data}"
    --incremental-outdir="${incremental_outdir}"
    --out-sample="${out_sample}"
    --manifest="${manifest}"
)
if (( DRY_RUN )); then
    PREP_ARGS+=(--dry-run)
fi

python3 "${CODE_DIR}/prepare_incremental_fastspec_sample.py" "${PREP_ARGS[@]}"
prep_status=$?
if (( prep_status != 0 )); then
    exit "${prep_status}"
fi

n_missing=$(python3 -c "import json; print(json.load(open('${manifest}'))['n_missing_targetids'])")
if (( n_missing == 0 )); then
    echo "0 missing TARGETIDs; nothing to fit or combine."
    exit 0
fi

if (( DRY_RUN )); then
    echo "DRY_RUN=1: inspect manifest at ${manifest} then re-submit without DRY_RUN."
    exit 0
fi

if [[ ! -f "${out_sample}" ]]; then
    echo "ERROR: incremental sample missing at ${out_sample}"
    exit 1
fi

# ---- Step 2: mpi-fastspecfit into the scratch tree (no --overwrite) ---------
# The scratch tree starts empty, so every healpix is fit and the canonical 450k
# tree is never touched. No delete step is needed.
mkdir -p "${incremental_outdir}"
mpiscript=$(type -p mpi-fastspecfit)
echo "mpi-fastspecfit: ${mpiscript}"

echo "Warming up Numba cache on one rank..."
read wu_survey wu_program wu_healpix <<< $(python3 -c "
from astropy.table import Table; t = Table.read('${out_sample}')[0]
print(t['SURVEY'], t['PROGRAM'], t['HEALPIX'])")
srun --nodes=1 --ntasks=1 --cpus-per-task=2 --cpu-bind=cores \
    ${mpiscript} \
    --specprod=iron --coadd-type=healpix \
    --survey=${wu_survey} --program=${wu_program} --healpix=${wu_healpix} \
    --mp=1 --ntargets=1 --nmonte=0 --nompi \
    --outdir-data=/tmp/fastspecfit-warmup --overwrite \
    --templates=${templates} \
    --emlinesfile=${emlinesfile} \
    --constraintsfile=${constraintsfile} \
    --vdisp-nominal 100 --vdisp-bounds 50 200 --ignore-quasarnet
echo "Warm-up complete."

ntasks=$(( 128 * N / mp ))
if (( mp > 1 )); then
    cpus_per_task=$(( mp * 2 ))
    cpu_bind="none"
else
    cpus_per_task=2
    cpu_bind="cores"
fi

echo "Launching incremental fit: nodes=${N} ntasks=${ntasks} mp=${mp}"

time srun --nodes=${N} --ntasks=${ntasks} \
          --cpus-per-task=${cpus_per_task} --cpu-bind=${cpu_bind} \
    ${mpiscript} \
        --samplefile="${out_sample}" \
        --outdir-data="${incremental_outdir}" \
        --emlinesfile="${emlinesfile}" \
        --constraintsfile="${constraintsfile}" \
        --templates="${templates}" \
        --specprod iron \
        --mp=${mp} \
        --nmonte=100 \
        --vdisp-nominal 100 --vdisp-bounds 50 200 \
        --ignore-quasarnet

# ---- Step 3: vstack incremental rows onto merged catalog --------------------
python3 "${CODE_DIR}/combine_incremental_fastspec.py" \
    --manifest="${manifest}" \
    --fastspec-merged="${fastspec_merged}" \
    --out-merged="${out_merged}"

echo "Done. Review ${out_merged} then replace ${fastspec_merged} if satisfied."
