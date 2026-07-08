#!/bin/bash -l
#
# submit_loa_catalog.sh — submit the two-stage loa dwarf catalog build as a
# SLURM afterok dependency chain.
#
# Run this on a Perlmutter LOGIN node (do NOT sbatch this file; it is a
# submitter, not a batch job):
#
#     ./submit_loa_catalog.sh
#
# Stage 1  run_select_loa.sh             reads zall-pix-loa.fits (iron-format
#                                        healpix), selects BGS_BRIGHT/BGS_FAINT/
#                                        LOW_Z, dereddens, computes the fiducial
#                                        stellar mass and keeps dwarfs
#                                        (LOGM_M24 < 9.25)   -> ..._dwarfs.fits
# Stage 2  run_crossmatch_loa.sh         cross-matches to DR9 Tractor for
#                                        FRACFLUX + MW_TRANSMISSION, adds
#                                        dereddened mags, applies the FRACFLUX
#                                        cut               -> ..._dwarfs_clean.fits
#
# Both jobs enter the queue immediately, but Stage 2 is held by SLURM and only
# starts once Stage 1 finishes SUCCESSFULLY (exit 0) -- it reads Stage 1's
# output. That is the afterok dependency. Re-running just Stage 2 (e.g. after
# tweaking the FRACFLUX threshold) then never repeats the Stage 1 selection.

set -euo pipefail

SCRIPT_DIR="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/job_scripts/catalog_construct"

SELECT_JOB="${SCRIPT_DIR}/run_select_loa.sh"
XMATCH_JOB="${SCRIPT_DIR}/run_crossmatch_loa.sh"

# Stage 1: selection + dwarf cut. --parsable makes sbatch print just the job id.
jid1=$(sbatch --parsable "${SELECT_JOB}")
echo "Submitted stage 1 (select loa dwarfs): job ${jid1}"

# Stage 2: Tractor cross-match, gated on stage 1 succeeding.
#   afterok:<jid1>          -> start only if stage 1 exits 0
#   --kill-on-invalid-dep   -> if stage 1 FAILS, cancel stage 2 automatically
jid2=$(sbatch --parsable \
    --dependency=afterok:"${jid1}" \
    --kill-on-invalid-dep=yes \
    "${XMATCH_JOB}")
echo "Submitted stage 2 (crossmatch loa):    job ${jid2}  (depends on afterok:${jid1})"

echo
echo "Watch the chain with:  squeue --me"
