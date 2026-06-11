#!/bin/bash -l
#
# submit_make_cat.sh — submit the two-stage dwarf-catalog build as a SLURM
# dependency chain.
#
# Run this on a Perlmutter LOGIN node (do NOT sbatch this file itself; it is a
# submitter, not a batch job):
#
#     ./submit_make_cat.sh
#
# Stage 1  run_consolidate.sh    builds the multi-extension catalog
#                                /pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits
#                                (MAIN, ZCAT, TRACTOR, FASTSPEC, REPROCESS_PHOTO,
#                                 SPECTRA_TEMPLATE, IMG_SSL extensions).
# Stage 2  run_nebular_props.sh  appends the SPEC_DERIVED HDU to that SAME file.
#
# Both jobs enter the queue immediately ("at the same time"), but Stage 2 is
# held by SLURM and only starts once Stage 1 finishes SUCCESSFULLY (exit 0),
# because it edits the file Stage 1 produces. That is the afterok dependency.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONSOLIDATE_JOB="${SCRIPT_DIR}/run_consolidate.sh"
NEBULAR_JOB="${SCRIPT_DIR}/run_nebular_props.sh"

# Stage 1: consolidation. --parsable makes sbatch print just the numeric job id.
jid1=$(sbatch --parsable "${CONSOLIDATE_JOB}")
echo "Submitted stage 1 (consolidate):   job ${jid1}"

# Stage 2: nebular properties, gated on stage 1 succeeding.
#   afterok:<jid1>          -> start only if stage 1 exits 0
#   --kill-on-invalid-dep   -> if stage 1 FAILS, cancel stage 2 automatically
#                              instead of leaving it stuck as
#                              DependencyNeverSatisfied.
jid2=$(sbatch --parsable \
    --dependency=afterok:"${jid1}" \
    --kill-on-invalid-dep=yes \
    "${NEBULAR_JOB}")
echo "Submitted stage 2 (nebular props): job ${jid2}  (depends on afterok:${jid1})"

echo
echo "Watch the chain with:  squeue --me"
