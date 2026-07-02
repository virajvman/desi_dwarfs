#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --job-name=scarlet_driver
#SBATCH --output=run_scarlet_driver.log

# ============================================================
# Stage 1 (FIT) + Stage 2 (CONSOLIDATE) of the SCARLET photometry pipeline.
# Sibling of job_scripts/aperture_runs/*'s single-node, ntasks=1,
# cpus-per-task=256 pattern (NOT the MPI pattern used by the PSF-download job
# -- scarlet_photo.driver is a single Python process with its own internal
# multiprocessing.Pool, no MPI needed).
#
# WHY --cpus-per-task=256 but --ncores 128 below: a Perlmutter CPU node has
# 128 PHYSICAL cores / 256 LOGICAL (2-way SMT/hyperthreaded) cores. SLURM's
# scheduler allocates by logical-CPU count, so --cpus-per-task=256 is what
# actually claims the WHOLE node exclusively (matches every other job in
# job_scripts/aperture_runs/, e.g. aperture_photo_job_bgsb.sh). The Python
# worker count is kept at 128 (physical cores) because SCARLET fits are
# floating-point/compute-bound (proxmin/ADMM linear algebra, convolutions) --
# two hyperthreads on the same physical core share its FPU/execution units,
# so running 256-way doesn't add real throughput and can cause cache/FPU
# contention that slows individual workers down instead.
#
# PREREQUISITE: the PSF store must already be populated for every object in
# CATALOG_PATH (see job_scripts/image_cutouts/psf/get_psfs_general.sbatch) --
# scarlet_photo.driver reads PSFs but does not build them.
#
# NOTE ON $0 / BASH_SOURCE: sbatch copies this script to a spool dir before
# running it, so BASH_SOURCE does NOT point at the repo -- REPO_ROOT below is
# hardcoded, per CLAUDE.md.
# ============================================================

set -eo pipefail

REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
CODE_DIR="${REPO_ROOT}/code"

# Confirmed 2026-07-XX: scarlet was pip-installed (--no-build-isolation --user,
# see git history / conversation log for why --no-build-isolation is needed --
# scarlet's setup.py lazy-imports pybind11/peigen at C++-extension compile
# time, which pip's isolated build sandbox never sees otherwise) directly on
# top of the shared desi_environment.sh env -- NOT a separate conda/venv.
# `python -c "import scarlet"` succeeded in exactly this environment.
source /global/cfs/cdirs/desi/software/desi_environment.sh main

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# HDF5 cutout/PSF/bundle stores: readers/writers need file locking disabled on
# CFS/Lustre (same as every other job script in this repo).
export HDF5_USE_FILE_LOCKING=FALSE

# ------------------------------
# Configurable flags
# ------------------------------
CATALOG_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_photometry/iron_BGS_BRIGHT_shreds_catalog_w_aper_mags.fits"
NCORES=128
FIXED_BOX_SIZE=350          # locked config (see DESIGN.md) -- catalog's native
                             # IMAGE_SIZE_PIX varies and is always >= this
LIMIT=""                    # e.g. "--limit 300" for a validation batch first
                             # (recommended before the first full run -- see
                             # the walltime note below)
RUN_CONSOLIDATE=true        # chain stage 2 right after stage 1 (cheap,
                             # numpy/h5py-only, no scarlet needed -- set false
                             # to review stage-1 output before packing it)

# ============================================================
# ROUGH WALLTIME GUIDANCE (from local testing, 2026-07-02, under the FINAL
# locked config -- fit_lsb=False, detect_scale=1, wavelet_K=5, GMM grouping,
# fixed_box_size=350): 6 hand-picked local test objects fit in ~155s
# wall-clock run 6-way parallel (~1 core each) -- i.e. per-object fit time is
# roughly 150-200s on that small sample. This has NOT been validated at
# NERSC-node core speed or against the real production sample's actual
# crowding/component-count distribution (denser fields will be slower) --
# treat it as a first-order estimate, not a guarantee.
#
#   wall_time ≈ ceil(N_objects / NCORES) * ~150-200s
#
#   Example: N_objects=6000, NCORES=128 -> ceil(6000/128)=47 batches
#            * ~150-200s ≈ 7050-9400s ≈ ~2.0-2.6 hours
#
# RECOMMENDATION: run a validation batch first (set LIMIT="--limit 300"
# above, short --time, e.g. 00:30:00) to get a REAL per-object timing
# distribution on this catalog before committing walltime for the full run.
# --time=06:00:00 above is a generous first-attempt guess (~2-3x the point
# estimate for 6000 objects) -- tighten or loosen after the validation batch.
#
# Re-submittable: driver.py skips objects whose fragment already exists
# unless --overwrite is passed, so if this job times out partway through, a
# clean re-submission just resumes from where it left off (no --overwrite
# needed, and adding it would force a wasteful full re-fit).
# ============================================================

echo "Starting at $(date)"
cd "$CODE_DIR"

python -c "import scarlet; print('scarlet OK:', scarlet.__file__)"

srun --cpu-bind=cores python -m scarlet_photo.driver \
    --input-catalog "$CATALOG_PATH" \
    --fixed-box-size "$FIXED_BOX_SIZE" \
    --ncores "$NCORES" \
    $LIMIT

if [ "$RUN_CONSOLIDATE" = true ]; then
    echo "Stage 1 done at $(date); starting consolidate"
    srun --cpu-bind=cores python -m scarlet_photo.consolidate \
        --input-catalog "$CATALOG_PATH" \
        --ncores 64
fi

echo "All done at $(date)"
