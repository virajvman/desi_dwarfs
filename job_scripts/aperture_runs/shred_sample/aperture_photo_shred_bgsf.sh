cd /global/u1/v/virajvm/DESI2_LOWZ
source /global/cfs/cdirs/desi/software/desi_environment.sh main
shifterimg pull docker:legacysurvey/legacypipe:DR10.3.4
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_FAINT -img_source -use_sample shred
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 100 -run_parr -ncores 4 -overwrite -nchunks 1 -run_aper -use_sample shred -no_cnn_cut
shifter --image docker:legacysurvey/legacypipe:DR10.3.4 python3 desi_dwarfs/code/tractor_model.py -sample BGS_FAINT -bkg_source -blend_remove_source -use_sample shred
python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_FAINT -min 0 -max 100 -run_parr -ncores 4 -overwrite -nchunks 1 -run_aper -run_cog -use_sample shred -no_cnn_cut

# python3 desi_dwarfs/code/dwarf_photo_pipeline.py -sample BGS_BRIGHT,BGS_FAINT,LOWZ,ELG -min 0 -max 50000 -run_parr -ncores 128 -overwrite -nchunks 2 -use_sample clean -no_cnn_cut -make_cats


# !/bin/bash -l

# SBATCH --account=desi
# SBATCH --qos=regular
# SBATCH --constraint=cpu
# SBATCH --mail-user=virajvm@stanford.edu
# SBATCH --mail-type=ALL
# SBATCH --nodes=1
# SBATCH --mem=128GB
# SBATCH --time=02:00:00
# SBATCH --output=aperture_phot_clean_bgsf.log