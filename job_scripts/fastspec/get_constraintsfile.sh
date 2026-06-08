#!/bin/bash

source /global/cfs/cdirs/desi/software/desi_environment.sh main
module load fastspecfit/3.4.3

CONSTRAINTS_SRC=$(python -c "from importlib.resources import files; print(files('fastspecfit') / 'data/emline-constraints.yaml')")
CONSTRAINTS_DST="/global/u1/v/virajvm/DESI2_LOWZ/desi_dwarfs/data/data_metal/emline-constraints-dwarfs.yaml"

if [[ ! -f "${CONSTRAINTS_SRC}" ]]; then
    echo "ERROR: bundled constraints file not found: ${CONSTRAINTS_SRC}"
    exit 1
fi

mkdir -p "$(dirname "${CONSTRAINTS_DST}")"
cp "${CONSTRAINTS_SRC}" "${CONSTRAINTS_DST}"

echo "Copied:"
echo "  from: ${CONSTRAINTS_SRC}"
echo "  to:   ${CONSTRAINTS_DST}"
echo ""
echo "Next: edit ${CONSTRAINTS_DST}"
echo "  1) Remove heii_4686 from global.free_lines"
echo "  2) Add heii_4686 to narrow_balmer members (narrow_only profile)"
echo "  3) Add heii_4686 to narrow_all members (narrow_broad profile)"