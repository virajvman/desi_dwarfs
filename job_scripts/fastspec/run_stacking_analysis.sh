source /global/cfs/cdirs/desi/software/desi_environment.sh main

cd /global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs
python3 code/nebular_stuff/stack_mstar_haew_5pct.py
bash job_scripts/fastspec/run_stack_fastspec_haew_5pct.sh
python3 code/nebular_stuff/stack_direct_metallicity.py --line-flux-type BOXFLUX --density-diagnostic SII