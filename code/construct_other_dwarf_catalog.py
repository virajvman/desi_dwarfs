"""
Identify dwarf galaxies hiding in QSO, MWS, and SCND samples.

This script cross-matches objects from these samples against an existing
dwarf catalog and applies cleaning cuts to find potential new dwarf candidates.
"""

import os
import sys
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from easyquery import Query

# Setup paths
rootdir = '/global/u1/v/virajvm/'
sys.path.append(os.path.join(rootdir, 'DESI2_LOWZ'))

from desi_lowz_funcs import (
    save_table, 
    get_useful_cat_colms, 
    _n_or_more_gt, 
    _n_or_more_lt, 
    get_remove_flag,
    match_c_to_catalog, 
    get_stellar_mass, 
    get_stellar_mass_mia, 
    calc_normalized_dist    
)

from construct_dwarf_galaxy_catalogs import read_tractorphot, get_final_catalogs, get_nam_distances, bright_star_filter


def load_main_dwarf_catalog(filename):
    """Load and filter the main dwarf catalog."""
    main_dwarf_cat = Table.read(filename, hdu="MAIN")
    # main_dwarf_cat = main_dwarf_cat[main_dwarf_cat["DWARF_MASKBIT"] == 0]
    
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
    # Match by TARGETID
    targetid_match = np.isin(zpix_sub_cat['TARGETID'], main_dwarf_cat['TARGETID'])
    
    # Match by position
    coords_zpix = SkyCoord(ra=np.array(zpix_sub_cat['RA'].data)*u.deg, dec=np.array(zpix_sub_cat['DEC'].data)*u.deg)
    coords_dwarf = SkyCoord(ra=np.array(main_dwarf_cat['RA'].data)*u.deg, dec=np.array(main_dwarf_cat['DEC'].data)*u.deg)
    
    idx, sep2d, _ = coords_zpix.match_to_catalog_sky(coords_dwarf)
    position_match = sep2d < match_radius_arcsec * u.arcsec
    
    # Objects to keep are those NOT matched by either method
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


def apply_shred_cuts(catalog):
    """
    Remove shredded objects with FRACFLUX > 0.2 in 2 or more bands.
    """
    fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]
    # Objects with fracflux < 0.2 in at least 2 bands are good
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
    # Want objects with SIGMA_GOOD >= threshold in at least nsigma_bands bands
    # Using _n_or_more_lt to find objects that DON'T meet this criterion
    nsigma_queries = [_n_or_more_lt(sigma_grz, nsigma_bands, nsigma_thresh)]
    
    good_mask = get_remove_flag(catalog, nsigma_queries) == 0
    return good_mask


def compute_distances_and_velocities(catalog, verbose=True):
    """
    Compute velocities in different reference frames and distance columns.
    The fiducial distance column is DIST_MPC_FIDU.
    """
    if verbose:
        print("Computing NAM distances and velocities...")
    
    catalog = get_nam_distances(catalog, compute_other_nam=False)
    return catalog


def compute_stellar_masses(catalog, verbose=True):
    """
    Compute optical color-based stellar masses.
    These prescriptions only work for z < 0.5 galaxies.
    
    Computes:
    - LOGM_SAGA_VCMB: SAGA prescription using r-band and VCMB distance
    - LOGM_SAGA_FIDU: SAGA prescription using r-band and fiducial distance
    - LOGM_M24_VCMB: M24 (Mia) prescription using g-band
    """
    if verbose:
        print("Computing optical color-based stellar masses...")
    
    # g-r colors
    gr_colors = catalog["MAG_G"] - catalog["MAG_R"]
    
    # Only valid for z < 0.5
    zred_mask = catalog["Z"] < 0.5
    n_valid = np.sum(zred_mask)
    
    if verbose:
        print(f"  {n_valid}/{len(catalog)} objects have z < 0.5 for stellar mass estimates")
    
    # Initialize columns with placeholder values
    catalog["LOGM_M24_VCMB"] = -99.0 * np.ones(len(catalog))
    
    if n_valid > 0:        
        # M24 (Mia) prescription using g-band magnitude
        mstars_M24_VCMB = get_stellar_mass_mia(
            gr_colors[zred_mask].data, 
            catalog["MAG_G"][zred_mask].data,
            catalog["Z_CMB"][zred_mask].data
        )
        
        # Add the stellar masses
        catalog["LOGM_M24_VCMB"][zred_mask] = mstars_M24_VCMB

    mstar_mask = (catalog["LOGM_M24_VCMB"] > 5) & (catalog["LOGM_M24_VCMB"] < 9.25)

    if verbose:
        print(f"  {np.sum(mstar_mask)}/{len(catalog)} objects have 5 < Mstar < 9.25!")
    
    return catalog[mstar_mask]


def add_supplementary_info(catalog, verbose=True):
    """
    Add supplementary information to the catalog:
    - Sweep file information
    - Nearby bright star contamination flags
    """
    if verbose:
        print("Adding bright star contamination flags...")
    catalog = bright_star_filter(catalog)
    
    return catalog


def process_sample(sample_name, zpix_sub_cat, main_dwarf_cat, 
                   compute_nam_dists=True, get_color_mstar=True, verbose=True):
    """
    Process a single sample (QSO, MWS, or SCND) to find potential dwarf candidates.
    
    Parameters
    ----------
    sample_name : str
        Name of the sample ('QSO', 'MWS', or 'SCND')
    zpix_sub_cat : astropy.table.Table
        Subset of zpix catalog for this sample
    main_dwarf_cat : astropy.table.Table
        Main dwarf catalog to cross-match against
    compute_nam_dists : bool
        Whether to compute NAM distances and velocities
    get_color_mstar : bool
        Whether to compute color-based stellar masses
    verbose : bool
        Print progress information
        
    Returns
    -------
    dwarf_candidates : astropy.table.Table
        Table of potential dwarf candidates passing all cuts
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {sample_name} sample")
        print(f"{'='*60}")
        print(f"Initial number of objects: {len(zpix_sub_cat)}")
    
    # Step 1: Remove objects already in the main dwarf catalog
    not_in_dwarf = remove_known_dwarfs(zpix_sub_cat, main_dwarf_cat)
    zpix_sub_cat = zpix_sub_cat[not_in_dwarf]
    
    if verbose:
        print(f"After removing known dwarfs: {len(zpix_sub_cat)}")
    
    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining for {sample_name}")
        return None
    
    # Step 2: Read tractor photometry
    zpix_trac = read_tractorphot(zpix_sub_cat, verbose=verbose)
    zpix_trac = get_useful_cat_colms(zpix_trac)
    
    # Step 3: Combine catalogs
    zpix_sub_cat, zpix_trac = get_final_catalogs(zpix_sub_cat, zpix_trac, sample_name)
    
    if verbose:
        max_ra_diff = np.max(np.abs(zpix_trac["RA"] - zpix_sub_cat["RA"]))
        print(f"Maximum RA difference: {max_ra_diff}")
    
    # Step 4: Apply maskbit cuts
    maskbit_good = apply_maskbit_cuts(zpix_trac)
    
    if verbose:
        print(f"Fraction passing maskbit cuts: {np.sum(maskbit_good)/len(maskbit_good):.3f}")
    
    zpix_sub_cat = zpix_sub_cat[maskbit_good]
    zpix_trac = zpix_trac[maskbit_good]
    
    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after maskbit cuts for {sample_name}")
        return None
    
    # Step 5: Apply shred cuts
    shred_good = apply_shred_cuts(zpix_sub_cat)
    
    if verbose:
        print(f"Fraction passing shred cuts: {np.sum(shred_good)/len(shred_good):.3f}")
    
    zpix_sub_cat = zpix_sub_cat[shred_good]
    zpix_trac = zpix_trac[shred_good]
    
    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after shred cuts for {sample_name}")
        return None
    
    # Step 6: Apply RCHISQ cuts
    rchisq_good = apply_rchisq_cuts(zpix_trac)
    
    if verbose:
        print(f"Fraction passing RCHISQ cuts: {np.sum(rchisq_good)/len(rchisq_good):.3f}")
    
    zpix_sub_cat = zpix_sub_cat[rchisq_good]
    zpix_trac = zpix_trac[rchisq_good]
    
    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after RCHISQ cuts for {sample_name}")
        return None
    
    # Step 7: Apply sigma detection cuts
    sigma_good = apply_sigma_cuts(zpix_sub_cat)
    
    if verbose:
        print(f"Fraction passing sigma cuts: {np.sum(sigma_good)/len(sigma_good):.3f}")
    
    zpix_sub_cat = zpix_sub_cat[sigma_good]
    zpix_trac = zpix_trac[sigma_good]
    
    if len(zpix_sub_cat) == 0:
        print(f"No objects remaining after sigma cuts for {sample_name}")
        return None
    
    if verbose:
        print(f"\nObjects passing all cleaning cuts: {len(zpix_sub_cat)}")
    
    # Step 8: Compute NAM distances and velocities
    if compute_nam_dists:
        zpix_sub_cat = compute_distances_and_velocities(zpix_sub_cat, verbose=verbose)
    
    # Step 9: Compute color-based stellar masses
    if get_color_mstar:
        zpix_sub_cat = compute_stellar_masses(zpix_sub_cat, verbose=verbose)
    
    # Step 10: Add supplementary information (sweeps, bright stars)
    zpix_sub_cat = add_supplementary_info(zpix_sub_cat, verbose=verbose)
    
    # Add sample identifier
    zpix_sub_cat['ORIGIN_SAMPLE'] = sample_name
    
    if verbose:
        print(f"\nFinal number of {sample_name} dwarf candidates: {len(zpix_sub_cat)}")
    
    return zpix_sub_cat


def main(compute_nam_dists=True, get_color_mstar=True):
    """
    Main function to identify hidden dwarf galaxies.
    
    Parameters
    ----------
    compute_nam_dists : bool
        Whether to compute NAM distances and velocities
    get_color_mstar : bool
        Whether to compute color-based stellar masses
    """
    
    # File paths
    dwarf_catalog_file = "/pscratch/sd/v/virajvm/desi_dwarf_catalogs/dr1/v1.0/desi_dr1_dwarf_catalog.fits"
    zpix_file = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/zdownselect-pix-iron.fits"
    output_dir = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs"
    
    print("Loading main dwarf catalog...")
    main_dwarf_cat = load_main_dwarf_catalog(dwarf_catalog_file)
    print(f"Main dwarf catalog: {len(main_dwarf_cat)} objects")
    
    print("\nLoading zpix catalog...")
    zpix_iron = load_zpix_catalog(zpix_file)
    print(f"zpix catalog (after quality cuts): {len(zpix_iron)} objects")
    
    # Get sample masks
    sample_masks = get_sample_masks(zpix_iron)
    
    for sample_name, mask in sample_masks.items():
        print(f"\n{sample_name}: {np.sum(mask)} objects")
    
    # Process each sample
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

        candidates.write(output_dir+f"/{sample_name}_temp.fits",overwrite=True)
        
        if candidates is not None and len(candidates) > 0:
            all_candidates.append(candidates)
    
    # Combine all candidates
    if all_candidates:
        combined_candidates = vstack(all_candidates)
        print(f"\n{'='*60}")
        print(f"Total dwarf candidates from QSO/MWS/SCND: {len(combined_candidates)}")
        print(f"{'='*60}")
        
        # Summary by sample
        for sample_name in ['QSO', 'MWS', 'SCND']:
            n_sample = np.sum(combined_candidates['ORIGIN_SAMPLE'] == sample_name)
            print(f"  {sample_name}: {n_sample} candidates")
        
        # Save results
        output_file = os.path.join(output_dir, "hidden_dwarf_candidates_qso_mws_scnd.fits")
        save_table(combined_candidates, output_file)
        print(f"\nSaved candidates to: {output_file}")
        
        return combined_candidates
    else:
        print("\nNo dwarf candidates found in QSO/MWS/SCND samples.")
        return None


if __name__ == "__main__":
    candidates = main()