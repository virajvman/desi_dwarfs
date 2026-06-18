#!/bin/bash -l

#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=virajvm@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --job-name=run_stack_elg_noelg_contnorm
#SBATCH --output=run_stack_elg_noelg_contnorm.log

# CONTINUUM (r-band) normalized ELG vs non-ELG stacking pipeline (2 stages).
# Sibling of run_stack_elg_noelg.sh (the Halpha-normalized product); kept
# independent so either product can be (re)run on its own. Writes to a SEPARATE
# directory (stack_files/mstar_contnorm/) with the same per-bin filenames, so the
# two products never clobber each other.
#
# Normalization: each spectrum is divided by a continuum r-band flux scalar
# derived from the rest-frame, emission-removed model magnitude
# MAG_R_SDSS_Z0_MODEL_NOEMI (F_r = 10^(-0.4*(mag - 22.5))), giving a
# luminosity-weighted stack. The per-galaxy scalar cancels in line ratios; it
# only reweights galaxies in the mean stack vs the Halpha (SF-weighted) product.
#
# Mass bins: 0.25 dex from log M*=6.0 to 9.5. Each (sample, mass) cell stacks
# when the catalog count >= 50; output FITS carry 1 mean row + 200 bootstrap rows
# (Scholte propagated-ivar error model, identical to the Halpha product).
#
# Direct-method metallicity (stage 3 of the Halpha pipeline) is intentionally
# NOT run here.
#
# Submit:   sbatch job_scripts/fastspec/run_stack_elg_noelg_contnorm.sh
# Monitor:  squeue --me   |   tail -f run_stack_elg_noelg_contnorm.log
# (Or run the body directly on an exclusive interactive Perlmutter CPU node.)
#
# Stage 2 (run_stack_fastspec_elg_noelg_contnorm.sh) sets up its OWN FastSpecFit
# environment in a child shell (`bash ...`), so its `module load fastspecfit`
# does not leak into stage 1, which uses the main DESI environment.

set -e

source /global/cfs/cdirs/desi/software/desi_environment.sh main

cd /global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs

# Unbuffered Python so the .log shows live per-bin progress -- SLURM block-buffers
# stdout, so a killed job otherwise loses all its progress prints. One BLAS thread
# per process: stage 1's bootstrap coadds run on a process pool (BOOT_NJOBS) and
# stage 2's stackfit --mp forks workers; in both cases we want a single thread per
# process to avoid CPU oversubscription and the fork-with-threaded-BLAS hang.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== [1/2] Bootstrap stacking, continuum-normalized (stack_mstar_elg_vs_noelg.py --norm continuum) ==="
python3 code/stacking_analysis/stack_mstar_elg_vs_noelg.py --norm continuum --resume

echo "=== [2/2] FastSpecFit on continuum-normalized ELG / NO-ELG stacks ==="
bash job_scripts/fastspec/run_stack_fastspec_elg_noelg_contnorm.sh

echo "=== Pipeline complete ==="
