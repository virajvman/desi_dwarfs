#!/bin/bash -l
#SBATCH --job-name=combine_dwarfs
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=logs/combine_fastspec_%j.out
#SBATCH --error=logs/combine_fastspec_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=virajvm@stanford.edu

# Merge the per-healpix fastspec catalogs produced by run_custom_fastspec_job.sh.
# Must use the SAME fastspecfit code as the production run so the merged schema
# matches the per-object outputs -- hence the same HEAD-source override below
# (see the long comment in run_custom_fastspec_job.sh for why 3.4.2 alone is
# insufficient). Merge itself needs no constraints/emlines/templates.
FSF_SRC=/global/homes/v/virajvm/packages/fastspecfit

source /dvs_ro/common/software/desi/desi_environment.sh main
module swap desitarget/4.7.2
module load fastspecfit/3.4.2
export PYTHONPATH=${FSF_SRC}/py:$PYTHONPATH
export PATH=${FSF_SRC}/bin:$PATH

# Fail fast if anything's off
if ! command -v mpi-fastspecfit &>/dev/null; then
    echo "ERROR: mpi-fastspecfit not on PATH after module load + override. Aborting."
    module avail fastspecfit 2>&1
    exit 1
fi

fsf_file=$(python -c "import fastspecfit, os; print(os.path.dirname(fastspecfit.__file__))")
echo "fastspecfit imported from: ${fsf_file}"
case "${fsf_file}" in
    "${FSF_SRC}"/*) : ;;
    *) echo "ERROR: fastspecfit NOT imported from HEAD checkout ${FSF_SRC} (got ${fsf_file}). Aborting."
       exit 1 ;;
esac

echo "=== Loaded modules ==="
module list 2>&1 | grep -E 'fastspecfit|desiutil|desispec|desitarget|speclite|desiconda'
echo "fastspecfit HEAD: $(git -C "${FSF_SRC}" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "======================"

export DESI_SPECTRO_REDUX=/dvs_ro/cfs/cdirs/desi/spectro/redux
export DUST_DIR=/dvs_ro/cfs/cdirs/cosmo/data/dust/v0_1
export FPHOTO_DIR=/dvs_ro/cfs/cdirs/desi/external/legacysurvey/dr9
export FTEMPLATES_DIR=/dvs_ro/cfs/cdirs/desi/public/external/templates/fastspecfit

mkdir -p logs

time srun -n 1 -c 128 --cpu-bind=none mpi-fastspecfit \
    --samplefile=/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits \
    --outdir-data=/pscratch/sd/v/virajvm/desi_dwarf_catalogs/fastspecfit_custom_run/ \
    --specprod iron \
    --merge --merge-suffix dr1-dwarfs \
    --mp 32
