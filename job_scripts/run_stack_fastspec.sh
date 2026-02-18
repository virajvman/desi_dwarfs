#!/bin/bash

source /global/cfs/cdirs/desi/software/desi_environment.sh main
module load fastspecfit/main

STACK_PATH="/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/stack_files/mstar"
NCORES=32

echo "Running FastSpecFit on mstar bin stacks"
echo "Stack path: ${STACK_PATH}"

# --- ELG stacks ---
echo ""
echo "--- ELG stacks ---"
for f in ${STACK_PATH}/stack_mstar_elg_*.fits; do
    if [ -f "$f" ]; then
        basename=$(basename "$f")
        outfile="${STACK_PATH}/fastspec_${basename}"
        echo "Processing: ${basename}"
        stackfit "$f" -o "$outfile" --mp ${NCORES}
        echo "  -> Saved to fastspec_${basename}"
        echo ""
    fi
done

# --- NO-ELG stacks ---
echo ""
echo "--- NO-ELG stacks ---"
for f in ${STACK_PATH}/stack_mstar_noelg_*.fits; do
    if [ -f "$f" ]; then
        basename=$(basename "$f")
        outfile="${STACK_PATH}/fastspec_${basename}"
        echo "Processing: ${basename}"
        stackfit "$f" -o "$outfile" --mp ${NCORES}
        echo "  -> Saved to fastspec_${basename}"
        echo ""
    fi
done

echo "Done!"