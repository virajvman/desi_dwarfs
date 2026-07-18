#!/bin/bash -l
#
# Stage 1: select BGS_BRIGHT/BGS_FAINT/LOW_Z from the matterhorn zall catalog.
# Reads the ~30 GB base file once; single process (no multiprocessing needed).
#
#SBATCH --account=desi
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=mh_select
#SBATCH --output=mh_select_%j.log

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
GROUP="pix"                                   # pix | tilecumulative
OUTDIR="/pscratch/sd/v/virajvm/matterhorn"
RAW_CAT="${OUTDIR}/matterhorn_${GROUP}_bgs_lowz_raw.fits"
# -------------------------------------------------------------------- #

# Science cuts written out explicitly for reproducibility (these are also the
# script defaults). Bump --zmax later to widen the redshift range.
python3 "${REPO_ROOT}/code/catalog_construct/select_matterhorn_bgs_lowz.py" \
    --group "${GROUP}" \
    --output "${RAW_CAT}" \
    --zmin 0.001 \
    --zmax 0.2 \
    --deltachi2-min 40 \
    --require-galaxy \
    --primary-only

echo "Stage 1 done -> ${RAW_CAT}"
