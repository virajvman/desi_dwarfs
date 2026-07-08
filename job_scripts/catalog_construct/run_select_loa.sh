#!/bin/bash -l
#
# Stage 1 (loa): select BGS_BRIGHT/BGS_FAINT/LOW_Z from the loa healpix zall
# catalog (iron format, single file), dereddens g/r/z from EBV+PHOTSYS, computes
# the fiducial stellar mass (get_stellar_mass_mia / de los Reyes+2024 Eq.13), and
# keeps only DWARFS (LOGM_M24 < 9.25). One pass over zall-pix-loa.fits; single
# process (no multiprocessing needed).
#
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --job-name=loa_select
#SBATCH --output=loa_select_%j.log

# -e and pipefail are safe throughout. nounset (-u) is enabled only AFTER the
# DESI environment is loaded: conda activation and desi_environment.sh reference
# unset vars (e.g. DESI_ROOT) and would abort the job under -u.
set -eo pipefail

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------ config ------------------------------ #
REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
GROUP="pix"                                   # loa is healpix -> pix
INDIR="/global/cfs/cdirs/desi/spectro/redux/loa/zcatalog/v1"
OUTDIR="/pscratch/sd/v/virajvm/loa"
DWARF_CAT="${OUTDIR}/loa_${GROUP}_bgs_lowz_dwarfs.fits"
# -------------------------------------------------------------------- #

# Science + dwarf cuts written out explicitly for reproducibility (these are
# also the script defaults). Bump --zmax / --logmstar-max later to widen.
python3 "${REPO_ROOT}/code/catalog_construct/select_loa_dwarfs.py" \
    --input-dir "${INDIR}" \
    --specprod loa \
    --group "${GROUP}" \
    --output "${DWARF_CAT}" \
    --zmin 0.001 \
    --zmax 0.2 \
    --deltachi2-min 40 \
    --logmstar-max 9.25 \
    --require-galaxy \
    --primary-only

echo "Stage 1 (loa) done -> ${DWARF_CAT}"
