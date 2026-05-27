"""
Append spectroscopically derived nebular properties to a consolidated dwarf
catalog produced by ``consolidate_photometry.py``.

The script reads the ``MAIN``, ``FASTSPEC`` (a.k.a. ``DWARF_CATALOG_SPEC_HDU``),
and ``TRACTOR`` extensions of the multi-extension FITS catalog and writes a
fresh ``SPEC_DERIVED`` HDU containing:

    TARGETID
    LOG_SFR_HALPHA, LOG_SFR_HALPHA_ERR
    LOG_MSTAR_24_FIBER
    LOG_HALPHA_SFR_FIBER
    Z_GAS_R23_N2

    DELTA_MAG_{G,R}_BASS2DECAM       (north-masked; previously in FASTSPEC)
    DELTA_MAG_{G,R}_NEB
    DELTA_MAG_{G,R}_DECAM2SDSS
    DELTA_MAG_{G,R}_KCORR

    TE_NE_OII, TE_T_OIII, TE_AV,
    TE_LOG_O2_ABUND, TE_LOG_O3_ABUND, TE_12_LOG_OH
        (each with _LO / _HI / _ERR siblings)
    TE_N_RATIOS, TE_FIT_SUCCESS

    MAG_{G,R}_DECAM_MODEL_NOEMI      (previously in FASTSPEC)
    MAG_{G,R}_DECAM_MODEL_WEMI
    MAG_{G,R}_BASS_MODEL_WEMI
    MAG_{G,R}_SDSS_MODEL_NOEMI
    MAG_{G,R}_SDSS_Z0_MODEL_NOEMI

The direct-method nebular fits (``TE_*``) are run only on rows passing
``line_snr_mask([HALPHA, HBETA, HGAMMA, OIII_4363, OIII_5007, OII_3726,
OII_3729], snr_val=5, min_lines=7)`` -- all other rows have NaN / False / 0
fills so the row order matches MAIN exactly.

The ``MAG_*_MODEL_*`` columns are matched by TARGETID to the pre-computed
``model_photometry_diffs_{gal_type}.fits`` tables; unmatched rows are NaN.

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
from sfr_and_metallicity import (
    build_spec_derived_hdu,
    add_model_photometry_to_spec_derived,
)


# ---------------------------------------------------------------------------
# Hard-coded run-time knobs (same style as the boolean/string toggles at the
# top of ``consolidate_photometry.py``'s ``__main__`` block).
# ---------------------------------------------------------------------------

# Number of parallel worker processes for the per-row direct-method fits.
# Set to the number of cores you allocated (e.g., 64 or 128 on Perlmutter
# CPU nodes). Do NOT use N_JOBS > 1 on login nodes.
N_JOBS = 64

# Fitting method for compute_direct_metallicities:
#   "ultranest" : nested sampling (matches Scholte+2026), slow but reliable
#   "mle"       : L-BFGS-B + Hessian errors, fast but less reliable
TE_METHOD = "ultranest"

# UltraNest min_num_live_points (ignored for "mle").
TE_MIN_NUM_LIVE_POINTS = 400

# Line-SNR gating for the direct-method fits. Only rows with at least
# TE_MIN_LINES of these lines at SNR >= TE_SNR_VAL get a TE_* fit.
TE_LINE_NAMES = ["HALPHA", "HBETA", "HGAMMA",
                 "OIII_4363", "OIII_5007",
                 "OII_3726", "OII_3729"]
TE_SNR_VAL = 3
TE_MIN_LINES = 7


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Append a SPEC_DERIVED HDU (Halpha SFR, fiber Mstar/SFR, "
            "strong-line metallicity, DELTA_MAG photometric corrections, "
            "direct-method nebular properties, fastspec MAG_*_MODEL_* "
            "model magnitudes) to a consolidated dwarf catalog."
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

    build_spec_derived_hdu(
        args.catalog_path,
        verbose=not args.quiet,
        n_jobs=N_JOBS,
        te_method=TE_METHOD,
        min_num_live_points=TE_MIN_NUM_LIVE_POINTS,
        te_line_names=TE_LINE_NAMES,
        te_snr_val=TE_SNR_VAL,
        te_min_lines=TE_MIN_LINES,
    )

    add_model_photometry_to_spec_derived(
        args.catalog_path,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
