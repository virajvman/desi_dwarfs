"""
Identify dwarf galaxies in QSO, MWS, and SCND samples.

This script cross-matches objects from these samples against an existing
dwarf catalog and applies cleaning cuts to find potential new dwarf candidates.
Distances, LOGM_M24_FIDU, sweeps/bright-star/NOBS steps align with the primary
INT_V2 pipeline in construct_dwarf_galaxy_catalogs.py; output is written for
nebular correction (iron_other_qso_scnd_candidates_INT_V2.fits) and legacy
hidden_dwarf_candidates_qso_mws_scnd.fits.

Run with ``main(run_neb_correction=True)`` or ``python construct_other_dwarf_catalog.py --run-neb``
(after INT_V2) to build ``iron_other_qso_scnd_candidates_INT_V2_NEBCORR.fits`` for
``consolidate_photometry.process_qso_scnd`` (requires ``fastspec_funcs``).
"""

import os
import sys
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from easyquery import Query

# Repo code (independent_distances, etc.)
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# Setup paths for desi_lowz_funcs (NERSC layout)
rootdir = '/global/u1/v/virajvm/'
sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ'))

from desi_lowz_funcs import (
    save_table,
    get_useful_cat_colms,
    _n_or_more_lt,
    get_remove_flag,
    add_sweeps_column,
    get_stellar_mass_mia,
)

from independent_distances import update_distance_catalog

from construct_dwarf_galaxy_catalogs import (
    read_tractorphot,
    get_final_catalogs,
    get_nam_distances,
    bright_star_filter,
    run_nebular_correction_int_v2,
)


def load_main_dwarf_catalog(filename):
    """Load and filter the main dwarf catalog."""
    main_dwarf_cat = Table.read(filename, hdu="MAIN")

    # Select only objects in BGS, ELG, LOWZ
    samp_mask = (
        (main_dwarf_cat["SAMPLE"] == "BGS_BRIGHT") |
        (main_dwarf_cat["SAMPLE"] == "BGS_FAINT") |
        (main_dwarf_cat["SAMPLE"] == "LOWZ") |
        (main_dwarf_cat["SAMPLE"] == "ELG")
    )
    main_dwarf_cat = main_dwarf_cat[samp_mask]

    return main_dwarf_cat


def load_zpix_catalog(filename):
    """Load zpix catalog with quality cuts."""
    zpix_iron = Table.read(filename)
    zpix_iron = zpix_iron[(zpix_iron["DELTACHI2"] > 40) & (zpix_iron["ZWARN"] == 0)]
    return zpix_iron


def get_sample_masks(zpix_cat):
    """Get boolean masks for QSO, MWS, and SCND samples."""
    desi_tgt = zpix_cat['DESI_TARGET']

    masks = {
        'QSO': (desi_tgt & 2**2) != 0,
        'SCND': (desi_tgt & 2**62) != 0,
        'MWS': (desi_tgt & 2**61) != 0,
    }

    return masks


def remove_known_dwarfs(zpix_sub_cat, main_dwarf_cat, match_radius_arcsec=1.0):
    """
    Remove objects that are already in the main dwarf catalog.
    Match by TARGETID or by position within match_radius.

    Returns mask of objects NOT in the main dwarf catalog.
    """
    targetid_match = np.isin(zpix_sub_cat['TARGETID'], main_dwarf_cat['TARGETID'])

    coords_zpix = SkyCoord(ra=np.array(zpix_sub_cat['RA'].data)*u.deg, dec=np.array(zpix_sub_cat['DEC'].data)*u.deg)
    coords_dwarf = SkyCoord(ra=np.array(main_dwarf_cat['RA'].data)*u.deg, dec=np.array(main_dwarf_cat['DEC'].data)*u.deg)

    idx, sep2d, _ = coords_zpix.match_to_catalog_sky(coords_dwarf)
    position_match = sep2d < match_radius_arcsec * u.arcsec

    not_in_dwarf_cat = ~(targetid_match | position_match)

    return not_in_dwarf_cat


def apply_maskbit_cuts(catalog):
    """Apply maskbit cleaning cuts."""
    remove_queries = [
        "(MASKBITS >> 1) % 2 > 0",   # Bit 1
        "(MASKBITS >> 5) % 2 > 0",   # Bit 5
        "(MASKBITS >> 6) % 2 > 0",   # Bit 6
        "(MASKBITS >> 7) % 2 > 0",   # Bit 7
        "(MASKBITS >> 13) % 2 > 0",  # Bit 13
    ]

    good_mask = get_remove_flag(catalog, remove_queries) == 0
    return good_mask


def apply_pmra_cuts(catalog):
    """Apply pmra cleaning cuts."""

    pmra_snr = np.abs(catalog["PMRA"].data) * np.sqrt(catalog["PMRA_IVAR"].data)
    pmdec_snr = np.abs(catalog["PMDEC"].data) * np.sqrt(catalog["PMDEC_IVAR"].data)
    signi_pm = (pmra_snr > 2) | (pmdec_snr > 2)

    print(f"{np.sum(signi_pm)}/{len(catalog)} objects have significant proper motion.")

    return ~signi_pm


def apply_shred_cuts(catalog):
    """
    Remove shredded objects with FRACFLUX > 0.2 in 2 or more bands.
    """
    fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]
    shred_queries = [Query(_n_or_more_lt(fracflux_grz, 2, 0.2))]

    shred_mask = get_remove_flag(catalog, shred_queries) == 0
    return ~shred_mask


def apply_rchisq_cuts(catalog, rchisq_cut=10):
    """Remove objects with poor model fits (RCHISQ > cut in any band)."""
    poor_fit = (
        (catalog["RCHISQ_G"] > rchisq_cut) |
        (catalog["RCHISQ_R"] > rchisq_cut) |
        (catalog["RCHISQ_Z"] > rchisq_cut)
    )
    good_mask = ~poor_fit
    return good_mask


def apply_sigma_cuts(catalog, nsigma_bands=2, nsigma_thresh=5):
    """
    Require robust detection (SIGMA_GOOD > threshold) in at least nsigma_bands bands.
    """
    sigma_grz = [f"SIGMA_GOOD_{b}" for b in "GRZ"]
    nsigma_queries = [_n_or_more_lt(sigma_grz, nsigma_bands, nsigma_thresh)]

    good_mask = get_remove_flag(catalog, nsigma_queries) == 0
    return good_mask


def compute_distances_and_velocities(catalog, verbose=True):
    """
    NAM + independent distance updates; fiducial column is DIST_MPC_FIDU.
    Matches primary construct_dwarf_galaxy_catalogs.py (compute_other_nam=True).
    """
    if verbose:
        print("Computing NAM distances, velocities, and independent distance updates...")

    catalog = get_nam_distances(catalog, compute_other_nam=True)
    catalog, _, _, _ = update_distance_catalog(
        catalog,
        size_col="SHAPE_R",
        dist_col="DIST_MPC_FIDU",
        keep_lumi_dist_orig=True
    )
    return catalog


def compute_logm_m24_fidu(catalog, verbose=True):
    """
    Pre-nebular LOGM_M24_FIDU (Mia prescription), same as primary INT_V2.
    Fills LOGM_M24_FIDU for z < 0.5; leaves -99 otherwise. No row filtering here.
    """
    if verbose:
        print("Computing LOGM_M24_FIDU (pre-nebular, primary prescription)...")

    gr_colors = catalog["MAG_G"] - catalog["MAG_R"]
    zred_mask = catalog["Z"] < 0.5
    n_valid = int(np.sum(zred_mask))

    if verbose:
        print(f"  {n_valid}/{len(catalog)} objects have z < 0.5 for stellar mass estimates")

    catalog["LOGM_M24_FIDU"] = -99.0 * np.ones(len(catalog), dtype=np.float64)

    if n_valid > 0:
        mstars = get_stellar_mass_mia(
            gr_colors[zred_mask].data,
            catalog["MAG_G"][zred_mask].data,
            catalog["Z_CMB"][zred_mask].data,
            d_in_mpc=catalog["DIST_MPC_FIDU"][zred_mask].data,
            input_zred=False,
        )
        catalog["LOGM_M24_FIDU"][zred_mask] = mstars

    return catalog


def process_sample(sample_name, zpix_sub_cat, main_dwarf_cat,
                   compute_nam_dists=True, get_color_mstar=True, verbose=True):
    """
    Process a single sample (QSO, MWS, or SCND) to find potential dwarf candidates.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {sample_name} sample")
        print(f"{'='*60}")
        print(f"Initial number of objects: {len(zpix_sub_cat)}")

    not_in_dwarf = remove_known_dwarfs(zpix_sub_cat, main_dwarf_cat)
    zpix_sub_cat = zpix_sub_cat[not_in_dwarf]

    if verbose:
        print(f"After removing known dwarfs: {len(zpix_sub_cat)}")

    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining for {sample_name}")
        return None

    zpix_trac = read_tractorphot(zpix_sub_cat, verbose=verbose)
    zpix_trac = get_useful_cat_colms(zpix_trac)

    zpix_sub_cat, zpix_trac = get_final_catalogs(zpix_sub_cat, zpix_trac, sample_name)

    if verbose:
        max_ra_diff = np.max(np.abs(zpix_trac["RA"] - zpix_sub_cat["RA"]))
        print(f"Maximum RA difference: {max_ra_diff}")

    maskbit_good = apply_maskbit_cuts(zpix_trac)
    not_star_mask = apply_pmra_cuts(zpix_trac)

    if verbose:
        print(f"Fraction passing maskbit cuts: {np.sum(maskbit_good)/len(maskbit_good):.3f}")

    zpix_sub_cat = zpix_sub_cat[maskbit_good & not_star_mask]
    zpix_trac = zpix_trac[maskbit_good & not_star_mask]

    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after maskbit cuts for {sample_name}")
        return None

    shred_good = apply_shred_cuts(zpix_sub_cat)

    if verbose:
        print(f"Fraction passing shred cuts: {np.sum(shred_good)/len(shred_good):.3f}")

    zpix_sub_cat = zpix_sub_cat[shred_good]
    zpix_trac = zpix_trac[shred_good]

    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after shred cuts for {sample_name}")
        return None

    rchisq_good = apply_rchisq_cuts(zpix_trac)

    if verbose:
        print(f"Fraction passing RCHISQ cuts: {np.sum(rchisq_good)/len(rchisq_good):.3f}")

    zpix_sub_cat = zpix_sub_cat[rchisq_good]
    zpix_trac = zpix_trac[rchisq_good]

    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after RCHISQ cuts for {sample_name}")
        return None

    sigma_good = apply_sigma_cuts(zpix_sub_cat)

    if verbose:
        print(f"Fraction passing sigma cuts: {np.sum(sigma_good)/len(sigma_good):.3f}")

    zpix_sub_cat = zpix_sub_cat[sigma_good]
    zpix_trac = zpix_trac[sigma_good]

    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after sigma cuts for {sample_name}")
        return None

    if verbose:
        print(f"\nObjects passing legacy cleaning cuts: {len(zpix_sub_cat)}")

    if compute_nam_dists:
        zpix_sub_cat = compute_distances_and_velocities(zpix_sub_cat, verbose=verbose)

    if get_color_mstar:
        zpix_sub_cat = compute_logm_m24_fidu(zpix_sub_cat, verbose=verbose)

    zpix_sub_cat = add_sweeps_column(zpix_sub_cat)
    zpix_sub_cat = bright_star_filter(zpix_sub_cat)

    nobs_mask = (
        (zpix_sub_cat["NOBS_G"] > 0) &
        (zpix_sub_cat["NOBS_R"] > 0) &
        (zpix_sub_cat["NOBS_Z"] > 0)
    )
    if verbose:
        print(f"Fraction passing NOBS cut: {np.sum(nobs_mask)/len(nobs_mask):.3f}")
    zpix_sub_cat = zpix_sub_cat[nobs_mask]

    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after NOBS cut for {sample_name}")
        return None

    zpix_sub_cat['ORIGIN_SAMPLE'] = sample_name

    if verbose:
        print(f"\nFinal number of {sample_name} dwarf candidates: {len(zpix_sub_cat)}")

    return zpix_sub_cat


def main(
    compute_nam_dists=True,
    get_color_mstar=True,
    run_neb_correction=True,
    ncore_neb=16,
    overwrite_neb=True,
):
    """
    Main function to identify dwarf galaxies in QSO/MWS/SCND target classes.

    Parameters
    ----------
    compute_nam_dists : bool
        Whether to compute NAM distances and velocities
    get_color_mstar : bool
        Whether to compute LOGM_M24_FIDU (pre-nebular)
    run_neb_correction : bool
        If True, after writing INT_V2 run ``run_nebular_correction_int_v2`` for OTHER
        (needs ``fastspec_funcs`` on ``sys.path``).
    ncore_neb : int
        Workers passed to ``compute_photometry_catalog`` when ``run_neb_correction``.
    overwrite_neb : bool
        If False, reuse cached ``model_photometry_diffs_OTHER.fits`` when TARGETIDs match.
    """

    dwarf_catalog_file = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"
    zpix_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/zdownselect-pix-iron.fits"
    output_dir = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs"

    print("Loading main dwarf catalog...")
    main_dwarf_cat = load_main_dwarf_catalog(dwarf_catalog_file)
    print(f"Main dwarf catalog: {len(main_dwarf_cat)} objects")

    print("\nLoading zpix catalog...")
    zpix_iron = load_zpix_catalog(zpix_file)
    print(f"zpix catalog (after quality cuts): {len(zpix_iron)} objects")

    sample_masks = get_sample_masks(zpix_iron)

    for sample_name, mask in sample_masks.items():
        print(f"\n{sample_name}: {np.sum(mask)} objects")

    all_candidates = []

    for sample_name, mask in sample_masks.items():
        zpix_sub_cat = zpix_iron[mask]

        candidates = process_sample(
            sample_name,
            zpix_sub_cat,
            main_dwarf_cat,
            compute_nam_dists=compute_nam_dists,
            get_color_mstar=get_color_mstar,
            verbose=True
        )

        if candidates is not None and len(candidates) > 0:
            candidates.write(os.path.join(output_dir, f"{sample_name}_temp.fits"), overwrite=True)
            all_candidates.append(candidates)

    if all_candidates:
        combined_candidates = vstack(all_candidates)
        print(f"\n{'='*60}")
        print(f"Total dwarf candidates from QSO/MWS/SCND: {len(combined_candidates)}")
        print(f"{'='*60}")

        for sample_name in ['QSO', 'MWS', 'SCND']:
            n_sample = np.sum(combined_candidates['ORIGIN_SAMPLE'] == sample_name)
            print(f"  {sample_name}: {n_sample} candidates")

        output_file = os.path.join(output_dir, "hidden_dwarf_candidates_qso_mws_scnd.fits")
        save_table(combined_candidates, output_file)
        print(f"\nSaved candidates to: {output_file}")

        int_v2_name = os.path.join(output_dir, "iron_other_qso_scnd_candidates_INT_V2.fits")
        save_table(combined_candidates, int_v2_name)
        print(f"Saved INT_V2-compatible table for nebular step: {int_v2_name}")

        other_base = "iron_other_qso_scnd_candidates.fits"
        if run_neb_correction:
            run_nebular_correction_int_v2(
                output_dir,
                other_base,
                "OTHER",
                ncore_neb=ncore_neb,
                overwrite=overwrite_neb,
            )

        return combined_candidates

    print("\nNo dwarf candidates found in QSO/MWS/SCND samples.")
    return None


if __name__ == "__main__":
    main()
