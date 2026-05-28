#!/bin/bash -l
#SBATCH --job-name=fastspec_dwarfs_400k
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=10
#SBATCH --time=06:00:00
#SBATCH --output=logs/fastspec_400k_%j.out
#SBATCH --error=logs/fastspec_400k_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=virajvm@stanford.edu

# ---- knobs -----------------------------------------------------------------
# Pick mp from the avg-targets/healpix measurement (see README.sample step 2):
#   per-object mode (avg >= mp):   mp=64 or mp=128
#   per-file   mode (avg <  mp):   mp=16 (gives many MPI ranks)
N=10
mp=16                                # change after measuring avg targets/healpix
samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits
emlinesfile=/global/homes/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emlines.ecsv
templates=/global/cfs/cdirs/desi/users/dscholte/data/ohno/templates/9.9.9/ftemplates-chabrier-9.9.9.fits
outdir_data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/
# ---------------------------------------------------------------------------

# Match the recommended stack in etc/fastspecfit-env.sh
export FASTSPECFIT_VERSION=3.4.1
export DESITARGET_VERSION=4.7.2
source /dvs_ro/common/software/desi/desi_environment.sh main-2.2.0
module swap desitarget/${DESITARGET_VERSION}
module load fastspecfit/${FASTSPECFIT_VERSION}

# ADD THIS BLOCK ---->
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

# Shared numba JIT cache (must be on a path visible from every compute node)
export NUMBA_CACHE_DIR=${PSCRATCH}/fastspecfit/numba-cache/${FASTSPECFIT_VERSION}
mkdir -p "${NUMBA_CACHE_DIR}" logs

mpiscript=$(type -p mpi-fastspecfit)

# --- Numba warm-up on a single rank (populates NUMBA_CACHE_DIR) -------------
echo "Warming up Numba cache on one rank..."
read wu_survey wu_program wu_healpix <<< $(python -c "
from astropy.table import Table; t = Table.read('${samplefile}')[0]
print(t['SURVEY'], t['PROGRAM'], t['HEALPIX'])")
srun --nodes=1 --ntasks=1 --cpus-per-task=2 --cpu-bind=cores \
    ${mpiscript} \
    --specprod=iron --coadd-type=healpix \
    --survey=${wu_survey} --program=${wu_program} --healpix=${wu_healpix} \
    --mp=1 --ntargets=1 --nmonte=0 --nompi \
    --outdir-data=/tmp/fastspecfit-warmup --overwrite \
    --templates=${templates} --emlinesfile=${emlinesfile} \
    --vdisp-nominal 100 --vdisp-bounds 50 200 --ignore-quasarnet
echo "Warm-up complete."

# --- Production run ---------------------------------------------------------
ntasks=$(( 128 * N / mp ))
if (( mp > 1 )); then
    cpus_per_task=$(( mp * 2 ))
    cpu_bind="none"
else
    cpus_per_task=2
    cpu_bind="cores"
fi

echo "Launching: nodes=${N} ntasks=${ntasks} mp=${mp} cpus_per_task=${cpus_per_task}"

time srun --nodes=${N} --ntasks=${ntasks} \
          --cpus-per-task=${cpus_per_task} --cpu-bind=${cpu_bind} \
    ${mpiscript} \
        --samplefile=${samplefile} \
        --outdir-data=${outdir_data} \
        --emlinesfile=${emlinesfile} \
        --templates=${templates} \
        --specprod iron \
        --mp=${mp} \
        --nmonte=100 \
        --vdisp-nominal 100 --vdisp-bounds 50 200 \
        --ignore-quasarnet