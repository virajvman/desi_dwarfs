#!/bin/bash -l
#
# Login-node submitter: fire the four shred-sample aperture/COG photometry jobs,
# then chain the reconstructed-cube consolidation pass with an afterok
# dependency so it runs only once all four have finished successfully.
#
# The consolidation MUST be a single job over all four samples (the reconstructed
# store is unified by brick, and one brick can hold objects from multiple
# samples -- single-writer-per-shard). This script enforces that ordering.
#
# Usage:   ./submit_shred_photo_and_consolidate.sh
#
# Only calls sbatch (never sources the DESI env), so -u is safe here.
set -euo pipefail

# Hardcoded canonical paths (SLURM copies scripts to spool; see CLAUDE.md).
REPO_ROOT="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs"
APER_DIR="${REPO_ROOT}/job_scripts/aperture_runs"
SHRED_DIR="${APER_DIR}/shred_sample"
CONSOLIDATE_JOB="${APER_DIR}/consolidate_reconstructed_job.sh"

# Photometry jobs (re)run before consolidation: the four shred samples plus
# SGA. All must run with the new code so their catalogs carry PSFSIZE_* and
# APER_RADEC_CEN_*, which the consolidation reads.
PHOTO_JOBS=(
    "${SHRED_DIR}/aperture_photo_job_bgsb.sh"
    "${SHRED_DIR}/aperture_photo_job_bgsf.sh"
    "${SHRED_DIR}/aperture_photo_job_elg.sh"
    "${SHRED_DIR}/aperture_photo_job_lowz.sh"
    "${SHRED_DIR}/aperture_photo_job_sga.sh"
)

echo "Submitting shred photometry jobs..."
JOB_IDS=()
for job in "${PHOTO_JOBS[@]}"; do
    if [ ! -f "$job" ]; then
        echo "ERROR: job script not found: $job" >&2
        exit 1
    fi
    jid=$(sbatch --parsable "$job")
    JOB_IDS+=("$jid")
    echo "  submitted $(basename "$job") -> job $jid"
done

# Build the afterok dependency string: afterok:<id1>:<id2>:...
DEP="afterok"
for jid in "${JOB_IDS[@]}"; do
    DEP="${DEP}:${jid}"
done

echo "Submitting consolidation job (dependency: ${DEP})..."
cons_jid=$(sbatch --parsable --dependency="${DEP}" "$CONSOLIDATE_JOB")
echo "  submitted $(basename "$CONSOLIDATE_JOB") -> job $cons_jid"

echo
echo "Done. Photometry jobs: ${JOB_IDS[*]}"
echo "Consolidation job:    ${cons_jid} (runs after all photometry jobs succeed)"
echo "Track with:  squeue -u \$USER"
