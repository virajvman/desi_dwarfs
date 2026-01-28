'''
In this script, we will find other DESI DR1 fibers that are associated with the parent galaxy.

code to make the minimal desi dr1 redshift catalog
# zpix_iron = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/zdownselect-pix-iron.fits")
# zpix_minimal = zpix_iron["TARGETID","RA","DEC","Z","DELTACHI2"]
# print(len(zpix_minimal))
# zpix_minimal = zpix_minimal[ zpix_minimal["DELTACHI2"].data > 40 ]
# print(len(zpix_minimal))
# zpix_minimal.write("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/zminimal-pix-iron.fits",overwrite=True)
zpix_iron  =  Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/zminimal-pix-iron.fits")

'''

from desi_lowz_funcs import find_objects_nearby
from astropy.table import Table, vstack
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

def is_zred_consistent(ref_z, cat, delta_zred=250 / 300000):
    """
    Check whether catalog redshifts are consistent with a reference redshift.

    Parameters
    ----------
    ref_z : float
        Reference redshift.
    cat : astropy.table.Table
        Catalog containing column 'Z'.
    delta_zred : float, optional
        Redshift tolerance (default corresponds to 250 km/s).

    Returns
    -------
    mask : np.ndarray (bool)
        Boolean mask of redshift-consistent objects.
    """
    return np.abs(cat["Z"] - ref_z) < delta_zred


def find_associated_tgids(dwarf_cat):
    # Read catalogs
    zpix_iron  = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/zminimal-pix-iron.fits")
    
    # Convert positions to SkyCoord (RA, DEC in degrees)
    dwarf_coords = SkyCoord(ra=dwarf_cat["RA"], dec=dwarf_cat["DEC"], unit=u.deg)
    zpix_coords = SkyCoord(ra=zpix_iron["RA"], dec=zpix_iron["DEC"], unit=u.deg)
    
    # Maximum search radius (2 x R50 in arcsec -> deg)
    max_radius_deg = (2.0 * np.max(dwarf_cat["R50_R"]) / 3600.0) * u.deg
    
    # Run search_around_sky once for all dwarfs
    # Correct: search around dwarfs
    idx_zpix, idx_dwarf, sep2d, _ = dwarf_coords.search_around_sky(zpix_coords, seplimit=max_radius_deg)

    print(np.max(idx_dwarf), np.max(idx_zpix))
    
    # --- per-dwarf radius in degrees ---
    dwarf_radii = (2.0 * dwarf_cat["R50_R"] / 3600.0) * u.deg
    
    # --- vectorized masks ---
    mask_radius = sep2d < dwarf_radii[idx_dwarf]
    mask_z      = np.abs(zpix_iron["Z"][idx_zpix] - dwarf_cat["Z"][idx_dwarf]) < 250 / 300000
    mask_not_self = zpix_iron["TARGETID"][idx_zpix] != dwarf_cat["TARGETID"][idx_dwarf]
    
    # combine masks
    mask = mask_radius & mask_z & mask_not_self
    
    # keep only matched indices
    idx_dwarf_matched = idx_dwarf[mask]
    idx_zpix_matched  = idx_zpix[mask]
    
    # --- group matches by dwarf index using np.unique + np.split ---
    if idx_dwarf_matched.size == 0:
        results = [np.array([], dtype=int) for _ in range(len(dwarf_cat))]
    else:
        # sort by dwarf index so grouping works
        order = np.argsort(idx_dwarf_matched)
        idx_dwarf_sorted = idx_dwarf_matched[order]
        idx_zpix_sorted  = idx_zpix_matched[order]
    
        # unique dwarfs and how many matches each
        unique_dwarfs, counts = np.unique(idx_dwarf_sorted, return_counts=True)
    
        # split zpix indices for each dwarf
        split_at = np.cumsum(counts)[:-1]
        groups = np.split(idx_zpix_sorted, split_at)
    
        # prepare final results array, same order as dwarf_cat
        results = [np.array([], dtype=int) for _ in range(len(dwarf_cat))]
        for uidx, grp in zip(unique_dwarfs, groups):
            results[uidx] = np.array(zpix_iron["TARGETID"][grp], dtype=int)
    
    # # now `results[i]` corresponds to `dwarf_cat[i]` and contains the associated TARGETIDs
    dwarf_cat["ASSOCIATED_TARGETIDS"] = results

    return dwarf_cat


def get_dwarf_primary(dwarf_cat):
    """
    Identify a single primary TARGETID per dwarf galaxy.

    Logic:
    - Consider the dwarf TARGETID plus all ASSOCIATED_TARGETIDS
    - Restrict to TARGETIDs that appear elsewhere in the catalog
    - Choose the brightest (minimum MAG_R_TARGET)
    - If no associates exist, the dwarf itself is primary
    """

    # Build fast lookup tables
    tid_to_index = {
        tid: i for i, tid in enumerate(dwarf_cat["TARGETID"])
    }

    primary_ids = np.full(len(dwarf_cat), -1, dtype=np.int64)

    for i, row in enumerate(dwarf_cat):
        tids = [row["TARGETID"]]
        tids.extend(row["ASSOCIATED_TARGETIDS"])

        valid_rows = [
            tid_to_index[tid]
            for tid in tids
            if tid in tid_to_index
        ]

        if len(valid_rows) == 0:
            primary_ids[i] = row["TARGETID"]
            continue

        mags = dwarf_cat["MAG_R_TARGET"][valid_rows]
        best = valid_rows[np.argmin(mags)]
        primary_ids[i] = dwarf_cat["TARGETID"][best]

    dwarf_cat["DWARF_PRIMARY_TARGETID"] = primary_ids

    #now with this dwarf primary targetid
    is_primary = (dwarf_cat["TARGETID"] == dwarf_cat["DWARF_PRIMARY_TARGETID"])

    dwarf_cat["DWARF_PRIMARY"] = is_primary

    return dwarf_cat


def get_associated_tgid_info(tgid_main, main, zpix_iron):
    '''
    Function used for plotting 
    '''
    # get the main row
    main_row = main[main["TARGETID"] == tgid_main]
    if len(main_row) == 0:
        raise ValueError(f"TARGETID {tgid_main} not found in main catalog.")
    
    # get associated IDs
    assoc_tgids = main_row["ASSOCIATED_TARGETIDS"][0]  # usually this is an array
    all_tgids = np.concatenate([[tgid_main], assoc_tgids])

    sample_info = []
    sample_ra = []
    sample_dec = []

    for tgidi in all_tgids:
        cat_i = main[main["TARGETID"] == tgidi]
        if len(cat_i) == 0:
            # fallback to zpix_iron
            zpix_i = zpix_iron[zpix_iron["TARGETID"] == tgidi]
            if len(zpix_i) == 0:
                raise ValueError("ISSUE!!")
            else:
                sample_info.append("OTHER")
                sample_ra.append(float(zpix_i["RA"][0]))
                sample_dec.append(float(zpix_i["DEC"][0]))
        else:
            # convert to normal Python types
            sample_info.append(str(cat_i["SAMPLE"][0]))
            sample_ra.append(float(cat_i["RA_TARGET"][0]))
            sample_dec.append(float(cat_i["DEC_TARGET"][0]))

    cutout_ra = float(main_row["RA"][0])
    cutout_dec = float(main_row["DEC"][0])
    gal_r50 = float(main_row["R50_R"][0])

    return cutout_ra, cutout_dec, gal_r50, sample_info, sample_ra, sample_dec

        