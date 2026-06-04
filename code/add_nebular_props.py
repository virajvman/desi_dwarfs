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
OII_3729], snr_val=5, min_lines=7)`` with per-line flux > 1 in FastSpec
units (1e-17 erg/cm2/s) -- all other rows have NaN / False / 0 fills so
the row order matches MAIN exactly.

The ``MAG_*_MODEL_*`` columns are matched by TARGETID to the pre-computed
``model_photometry_diffs_{gal_type}.fits`` tables; unmatched rows are NaN.

The operation is idempotent: re-running replaces any existing
``SPEC_DERIVED`` HDU. Existing HDUs (including ``FASTSPEC``) are preserved
bit-for-bit via a temp-file + ``os.replace`` swap.

The direct-method fit uses one of two strategies (set via
``TE_USE_INFORMATIVE_PRIORS`` or ``--use-informative-priors``):
    Plan A (default): single-stage joint 5-parameter UltraNest fit.
    Plan B: two-stage informative-prior fit -- fit ne/Te/Av first, then the
        abundances using the Stage-1 posteriors as priors. Bounds the
        pathological posteriors that make the single-stage fit slow.

The per-row direct-method UltraNest fits are cached by TARGETID. Plan A/B and
OII/SII density diagnostic each use a separate cache file (e.g.
``te_fit_cache_ultranest.fits`` for OII Plan A,
``te_fit_cache_ultranest_sii.fits`` for SII Plan A). Subsequent runs reuse cached rows and only fit TARGETIDs
that are new (or whose cached row has ``fit_success=False`` / NaN
``twelve_log_OH``, which are always retried). The cache is cumulative across
catalog versions; pass ``--overwrite-te-cache`` to force a fresh UltraNest
fit for every TARGETID in the current ``te_mask`` and upsert the results into
the cache file.

Usage:
    python add_nebular_props.py /path/to/desi_dr1_dwarf_catalog.fits \\
        --line-flux-type BOXFLUX
    python add_nebular_props.py /path/to/desi_dr1_dwarf_catalog.fits \\
        --line-flux-type BOXFLUX --overwrite-te-cache
    python add_nebular_props.py /path/to/desi_dr1_dwarf_catalog.fits \\
        --line-flux-type FLUX --use-informative-priors
    python add_nebular_props.py /path/to/desi_dr1_dwarf_catalog.fits \\
        --line-flux-type BOXFLUX --density-diagnostic SII
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

from desi_lowz_funcs import compute_separations
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
N_JOBS = 128

# UltraNest min_num_live_points for compute_direct_metallicities.
TE_MIN_NUM_LIVE_POINTS = 400

# Line-SNR gating for the direct-method fits. Only rows with at least
# TE_MIN_LINES of these lines at SNR >= TE_SNR_VAL get a TE_* fit.
_TE_LINE_NAMES_BASE = ["HALPHA", "HBETA", "HGAMMA",
                       "OIII_4363", "OIII_5007",
                       "OII_3726", "OII_3729"]
TE_SNR_VAL = 3

# Density diagnostic for the direct-method fit: 'OII' ([O II] 3726/3729) or
# 'SII' ([S II] 6716/6731). CLI --density-diagnostic overrides this constant.
TE_DENSITY_DIAGNOSTIC = "OII"

# UltraNest run() termination guards. Bound pathological fits so a few hard
# objects can't stall the whole parallel batch (see nebular_stuff/
# collaborator_code.py, which uses the same guards for its abundance stage).
TE_SAMPLER_KWARGS = {"frac_remain": 0.01, "max_iters": 40000, "max_ncalls": int(1e5)}

# Direct-method fit strategy:
#   False -> single-stage joint 5-parameter fit (Plan A, default).
#   True  -> two-stage informative-prior fit (Plan B): fit ne/Te/Av first,
#            then the abundances using the Stage-1 posteriors as priors.
# The CLI flag --use-informative-priors forces this on regardless of the
# constant. The two methods use separate TE-fit cache files so their cached
# results never mix.
TE_USE_INFORMATIVE_PRIORS = False


def _te_line_gating(density_diagnostic):
    """Line-SNR mask lines and min_lines for the direct-method TE fit."""
    names = list(_TE_LINE_NAMES_BASE)
    if density_diagnostic == "SII":
        names.extend(["SII_6716", "SII_6731"])
    return names, len(names)


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
        "--line-flux-type",
        required=True,
        choices=("FLUX", "BOXFLUX"),
        help=(
            "FastSpec line-flux family for nebular calculations "
            "(FLUX Gaussian or BOXFLUX boxcar). Required."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--overwrite-te-cache",
        action="store_true",
        help=(
            "Refit every TARGETID in the current te_mask with UltraNest even "
            "if a usable cache row exists, and upsert the new results into "
            "the cache file. Cache rows for TARGETIDs outside the current "
            "te_mask are left untouched."
        ),
    )
    parser.add_argument(
        "--use-informative-priors",
        action="store_true",
        help=(
            "Use the two-stage informative-prior direct-method fit (Plan B): "
            "fit ne/Te/Av first, then the abundances using the Stage-1 "
            "posteriors as priors. Overrides the TE_USE_INFORMATIVE_PRIORS "
            "module constant. Uses a separate TE-fit cache file."
        ),
    )
    parser.add_argument(
        "--density-diagnostic",
        choices=("OII", "SII"),
        default=None,
        help=(
            "Low-ionization doublet constraining electron density in the "
            "direct-method fit: OII ([O II] 3726/3729, default) or SII "
            "([S II] 6716/6731). Overrides TE_DENSITY_DIAGNOSTIC. Uses a "
            "separate TE-fit cache file from the other diagnostic."
        ),
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.catalog_path):
        parser.error(f"catalog_path does not exist: {args.catalog_path}")

    use_informative_priors = (
        args.use_informative_priors or TE_USE_INFORMATIVE_PRIORS
    )
    density_diagnostic = args.density_diagnostic or TE_DENSITY_DIAGNOSTIC
    te_line_names, te_min_lines = _te_line_gating(density_diagnostic)

    build_spec_derived_hdu(
        args.catalog_path,
        args.line_flux_type,
        verbose=not args.quiet,
        n_jobs=N_JOBS,
        min_num_live_points=TE_MIN_NUM_LIVE_POINTS,
        te_line_names=te_line_names,
        te_snr_val=TE_SNR_VAL,
        te_min_lines=te_min_lines,
        sampler_kwargs=TE_SAMPLER_KWARGS,
        use_informative_priors=use_informative_priors,
        density_diagnostic=density_diagnostic,
        overwrite_te_cache=args.overwrite_te_cache,
    )

    add_model_photometry_to_spec_derived(
        args.catalog_path,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

    #TODO: use BOXFLUX as fiducial flux value for stuff. Make note that is it likely biased for 20% of the toime
