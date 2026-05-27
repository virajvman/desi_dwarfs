#! /bin/bash

source /global/common/software/desi/desi_environment.sh main
cd /global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/code

python filter_qn_overrides_fastspec.py \
    --samplefile /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits \
    --outfile    /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs_qnsafe.fits \
    --specprod   iron \
    --nproc      32 \
    --report     /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs_qn_dropped.fits