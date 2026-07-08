#!/bin/bash -l
#
# Stage 2 (loa): cross-match the stage-1 loa dwarf selection to the Legacy
# Surveys DR9 Tractor catalogs (gather_tractorphot), attach FRACFLUX +
# MW_TRANSMISSION (+ FRACMASKED/FRACIN/RCHISQ/NOBS/FIBERFLUX), add dereddened
# mags, apply the FRACFLUX cut (all of g,r,z < 0.35), and write the final
# cleaned catalog. Parallelizes over bricks with Python multiprocessing.
#
# The stage-1 output already carries the exact join keys
# (RELEASE/BRICKID/BRICK_OBJID/BRICKNAME/PHOTSYS) that crossmatch_tractorphot.py
# needs, so the SAME stage-2 script used for matterhorn is reused unchanged --
# loa has no public lsdr9-photometry VAC, and DR9 is the targeting imaging for
# loa's BGS/LOW_Z, so the DR9 sweeps are the correct FRACFLUX source.
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
#SBATCH --time=05:00:00
#SBATCH --job-name=loa_xmatch
#SBATCH --output=loa_xmatch_%j.log

# -e and pipefail are safe throughout. nounset (-u) is enabled only AFTER the
# DESI environment is loaded: conda activation and desi_environment.sh reference
# unset vars (e.g. DESI_ROOT) and would abort the job under -u.
set -eo pipefail

cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
set -u

# One thread per BLAS op; parallelism comes from the script's process Pool.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ------------------------------ config ------------------------------ #
REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
GROUP="pix"                                   # must match stage 1
OUTDIR="/pscratch/sd/v/virajvm/loa"
DWARF_CAT="${OUTDIR}/loa_${GROUP}_bgs_lowz_dwarfs.fits"          # stage-1 output
CLEAN_CAT="${OUTDIR}/loa_${GROUP}_bgs_lowz_dwarfs_clean.fits"    # FRACFLUX-cut subset
PHOT_CAT="${OUTDIR}/loa_${GROUP}_bgs_lowz_dwarfs_phot.fits"      # all dwarfs + photometry
NPROC=128                                     # physical cores on a Perlmutter CPU node
# Elise's DR9 LOW_Z target catalogs (photometry source for LOW_Z-only objects).
NORTH_TARGETS="/pscratch/sd/v/virajvm/target/dr9_north_lowz_targets_no_rfib_cut.fits"
SOUTH_TARGETS="/pscratch/sd/v/virajvm/target/dr9_south_lowz_targets_no_rfib_cut_dec20.fits"
# DR9 is loa's main-survey targeting imaging; pass a dr10 dir here to override.
# LSDIR="/global/cfs/cdirs/desi/external/legacysurvey/dr9"
# -------------------------------------------------------------------- #

# Run python directly (NOT srun): this is a single-node multiprocessing job, so
# the Pool needs unbound access to all cores on the batch node.
python3 "${REPO_ROOT}/code/catalog_construct/crossmatch_tractorphot.py" \
    --input "${DWARF_CAT}" \
    --output "${CLEAN_CAT}" \
    --phot-output "${PHOT_CAT}" \
    --north-targets "${NORTH_TARGETS}" \
    --south-targets "${SOUTH_TARGETS}" \
    --fracflux-max 0.35 \
    --nproc "${NPROC}"

echo "Stage 2 (loa) done -> ${PHOT_CAT} (full) and ${CLEAN_CAT} (cleaned)"
