"""
Append spectroscopically derived nebular properties to a consolidated dwarf
catalog produced by ``consolidate_photometry.py``.

The script reads the ``MAIN``, ``FASTSPEC`` (a.k.a. ``DWARF_CATALOG_SPEC_HDU``),
and ``TRACTOR`` extensions of the multi-extension FITS catalog and writes a
fresh ``SPEC_DERIVED`` HDU containing, at present:

    TARGETID
    LOG_SFR_HALPHA, LOG_SFR_HALPHA_ERR
    LOG_MSTAR_24_FIBER
    LOG_HALPHA_SFR_FIBER
    Z_GAS_R23_N2

This script is the place where future spectroscopically derived nebular
properties (AV from Balmer decrements, direct-method metallicity, etc.)
should be added.

The operation is idempotent: re-running replaces any existing
``SPEC_DERIVED`` HDU. Existing HDUs (including ``FASTSPEC``) are preserved
bit-for-bit via a temp-file + ``os.replace`` swap.

Usage:
    python add_nebular_props.py /path/to/desi_dr1_dwarf_catalog.fits
"""

import argparse
import os
import sys

# ``code/nebular_stuff/`` is a flat folder of scripts (no __init__.py) and the
# rest of the project uses cwd-style imports (e.g. ``from sfr_and_metallicity
# import ...``). Make ``nebular_stuff/`` importable here so the user can invoke
# this script from anywhere without setting PYTHONPATH manually.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NEBULAR_DIR = os.path.join(_THIS_DIR, "nebular_stuff")
if _NEBULAR_DIR not in sys.path:
    sys.path.insert(0, _NEBULAR_DIR)

from code.desi_lowz_funcs import compute_separations
from sfr_and_metallicity import build_spec_derived_hdu


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Append a SPEC_DERIVED HDU (Halpha SFR, fiber Mstar/SFR, "
            "strong-line metallicity) to a consolidated dwarf catalog."
        ),
    )
    parser.add_argument(
        "catalog_path",
        help="Path to the multi-extension dwarf catalog FITS file.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.catalog_path):
        parser.error(f"catalog_path does not exist: {args.catalog_path}")

    build_spec_derived_hdu(args.catalog_path, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

    TODO: need to find AV per object. For now just take Halpha, Hbeta and Hgamma 
    if it exists and then fit AV to it assuming case b etc.

    Need to check in the objects where we have high enough SNR to do ne and te compute_separations
    if the ha/hb ratio changes a lot or not ... 
