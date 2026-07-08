#!/bin/bash -l
#
# submit_template_scan.sh — submit the two-stage spectral-template scan as a
# SLURM dependency chain.
#
# Run this on a Perlmutter LOGIN node (do NOT sbatch this file itself; it is a
# submitter, not a batch job):
#
#     ./submit_template_scan.sh
#
# Stage 1  scan_nnmf_ntemplates.sh   trains the NNMF grid (n = 1..20) from
#                                     scratch, fits it to the train + validation
#                                     halves, and writes:
#                                       templates_ntemp{n}.npy
#                                       hcoeffs_{valid,train}_ntemp{n}.npy
#                                       ntemplate_scan_summary.csv   (Panel 1)
#                                     Environment: DESI stack + cupy (GPU).
# Stage 2  scan_nnmf_pca_grid.sh     rebuilds each n_nmf residual from the saved
#                                     coefficients, runs residual PCA, and writes
#                                     the combined (n_nmf x n_pca) reduced-chi^2
#                                     grid + noise-floor crossings:
#                                       nnmf_pca_grid_summary.npz    (hist2d panels)
#                                     Environment: standalone pytorch module.
#
# The two stages use DIFFERENT environments (cupy/DESI vs the bare pytorch
# module), so they must be separate jobs. Both enter the queue immediately, but
# Stage 2 is held by SLURM until Stage 1 finishes SUCCESSFULLY (exit 0), because
# it reads the templates + coefficients Stage 1 produces. That is the afterok
# dependency.

set -euo pipefail

SCRIPT_DIR="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/job_scripts"

NNMF_JOB="${SCRIPT_DIR}/scan_nnmf_ntemplates.sh"
GRID_JOB="${SCRIPT_DIR}/scan_nnmf_pca_grid.sh"

# Stage 1: NNMF grid. --parsable makes sbatch print just the numeric job id.
jid1=$(sbatch --parsable "${NNMF_JOB}")
echo "Submitted stage 1 (nnmf grid):     job ${jid1}"

# Stage 2: combined grid, gated on stage 1 succeeding.
#   afterok:<jid1>          -> start only if stage 1 exits 0
#   --kill-on-invalid-dep   -> if stage 1 FAILS, cancel stage 2 automatically
#                              instead of leaving it stuck as
#                              DependencyNeverSatisfied.
jid2=$(sbatch --parsable \
    --dependency=afterok:"${jid1}" \
    --kill-on-invalid-dep=yes \
    "${GRID_JOB}")
echo "Submitted stage 2 (nnmf+pca grid): job ${jid2}  (depends on afterok:${jid1})"

echo
echo "Watch the chain with:  squeue --me"
